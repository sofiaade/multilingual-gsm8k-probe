"""
Multilingual GSM8K Probe
=========================
Research question:
    Does fine-tuning a language model for mathematical reasoning in English
    improve its reasoning in low-resource languages like Yoruba and Igbo?

Motivation:
    Most RLHF and fine-tuning research is conducted on English benchmarks.
    But if mathematical reasoning is learned as a general capability (not as
    language-specific pattern matching), the gains should transfer cross-lingually —
    even to languages the model has seen little training data for.

    This probe tests that hypothesis specifically for West African languages,
    motivated by the practical goal of making capable AI more accessible to
    the ~100M Yoruba speakers and ~45M Igbo speakers who are largely underserved
    by current NLP research.

Method:
    1. Take N problems from GSM8K (English ground truths known)
    2. Translate problem statements to Yoruba and Igbo via API
    3. Evaluate the same model on: English, Yoruba, Igbo versions
    4. Compare accuracy and analyze failure patterns
    5. Report: does math fine-tuning help cross-lingual reasoning?

Supported languages:
    - en: English (baseline)
    - yo: Yoruba (~100M speakers, Nigeria, Benin, Togo)
    - ig: Igbo (~45M speakers, southeastern Nigeria)
    - ha: Hausa (~70M speakers, northern Nigeria, Niger)
    - sw: Swahili (for comparison — higher-resource African language)

Translation:
    Uses Google Translate API (free tier via deep_translator).
    Results are cached to avoid re-translating across runs.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language Configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "en": {"name": "English",  "iso": "en", "translator_code": "en"},
    "yo": {"name": "Yoruba",   "iso": "yo", "translator_code": "yo"},
    "ig": {"name": "Igbo",     "iso": "ig", "translator_code": "ig"},
    "ha": {"name": "Hausa",    "iso": "ha", "translator_code": "ha"},
    "sw": {"name": "Swahili",  "iso": "sw", "translator_code": "sw"},
}

PROMPT_TEMPLATES = {
    "en": "Solve this math problem step by step. Write your final answer after '####'.\n\nProblem: {question}\n\nSolution:",
    "yo": "Yanju ìdánwò ìṣirò yìí ìgbésẹ̀ nípa ìgbésẹ̀. Kọ ìdáhùn rẹ lẹ́yìn '####'.\n\nÌdánwò: {question}\n\nÌdáhùn:",
    "ig": "Dozie nsogbu mgbakọ a usoro n'usoro. Dee ọzụzụ gị ka '####'.\n\nNsogbu: {question}\n\nNzaghachi:",
    "ha": "Warware wannan tambayar lissafi mataki-mataki. Rubuta amsar ka bayan '####'.\n\nTambaya: {question}\n\nAmsa:",
    "sw": "Tatua tatizo hili la hesabu hatua kwa hatua. Andika jibu lako baada ya '####'.\n\nTatizo: {question}\n\nSuluhisho:",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TranslatedProblem:
    """A GSM8K problem with translations into multiple languages."""
    idx:          int
    question_en:  str
    answer:       str   # Always the English numeric answer
    translations: dict  # {lang_code: translated_question}


@dataclass
class ProbeResult:
    """Result for one (problem, language) pair."""
    idx:         int
    lang:        str
    question:    str
    answer:      str
    output:      str
    extracted:   Optional[str]
    correct:     bool
    latency_s:   float


@dataclass
class LanguageReport:
    """Aggregate results for one language."""
    lang:           str
    lang_name:      str
    num_samples:    int
    num_correct:    int
    accuracy:       float
    extraction_rate: float
    avg_latency_s:  float
    sample_results: list[ProbeResult] = field(default_factory=list)


@dataclass
class MultilingualReport:
    """Full report across all languages and models."""
    model_name:  str
    languages:   list[str]
    num_samples: int
    results:     dict  # {lang: LanguageReport}
    timestamp:   str

    def accuracy_table(self) -> str:
        """Format accuracy results as a markdown table."""
        lines = ["| Language | Code | Accuracy | Correct/Total |"]
        lines += ["|---|---|---|---|"]
        for lang, report in self.results.items():
            lines.append(
                f"| {report.lang_name} | `{lang}` "
                f"| {report.accuracy:.1%} "
                f"| {report.num_correct}/{report.num_samples} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Translation (with caching)
# ---------------------------------------------------------------------------

class TranslationCache:
    """
    Disk-based cache for translations.
    Avoids re-calling the API across runs.
    Cache format: {lang: {original_text: translated_text}}
    """

    def __init__(self, cache_path: str = "data/translation_cache.json"):
        self.path = Path(cache_path)
        self.cache = {}
        if self.path.exists():
            with open(self.path) as f:
                self.cache = json.load(f)
            logger.info(f"Loaded translation cache from {self.path}")

    def get(self, lang: str, text: str) -> Optional[str]:
        return self.cache.get(lang, {}).get(text)

    def set(self, lang: str, text: str, translation: str):
        if lang not in self.cache:
            self.cache[lang] = {}
        self.cache[lang][text] = translation
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)


def translate_text(
    text: str,
    target_lang: str,
    cache: TranslationCache,
    sleep_between: float = 0.3,
) -> str:
    """
    Translate text to target_lang using Google Translate (via deep_translator).

    Falls back to the original text if translation fails.
    Results are cached — repeated calls are free.

    Install: pip install deep-translator
    """
    if target_lang == "en":
        return text  # Already English

    # Check cache first
    cached = cache.get(target_lang, text)
    if cached:
        return cached

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target=target_lang)
        result = translator.translate(text)
        cache.set(target_lang, text, result)
        time.sleep(sleep_between)  # Rate limiting
        return result
    except Exception as e:
        logger.warning(f"Translation failed for lang={target_lang}: {e}")
        return text   # Fall back to English


def translate_dataset(
    problems: list[dict],
    target_langs: list[str],
    cache: TranslationCache,
) -> list[TranslatedProblem]:
    """
    Translate all problems into all target languages.
    Respects the cache — only translates what's new.
    """
    translated = []
    for i, item in enumerate(tqdm(problems, desc="Translating problems")):
        question_en = item["question"]
        answer      = item["answer"].split("####")[-1].strip()
        translations = {"en": question_en}

        for lang in target_langs:
            if lang == "en":
                continue
            translations[lang] = translate_text(question_en, lang, cache)

        translated.append(TranslatedProblem(
            idx=i, question_en=question_en, answer=answer, translations=translations
        ))

    return translated


# ---------------------------------------------------------------------------
# Answer Extraction (language-agnostic)
# ---------------------------------------------------------------------------

import re

def extract_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer — works regardless of surrounding language.
    Numbers are the same across languages, so we look for digits after ####.
    """
    if not text:
        return None

    # #### pattern is language-agnostic — we use it in all prompts
    m = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Last number in text as fallback
    numbers = re.findall(r"([+-]?\d[\d,]*(?:\.\d+)?)", text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(predicted: Optional[str], ground_truth: str) -> bool:
    def norm(s):
        if not s:
            return ""
        s = s.replace(",", "").strip()
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return s
    return norm(predicted) == norm(ground_truth)


# ---------------------------------------------------------------------------
# Probe Runner
# ---------------------------------------------------------------------------

class MultilingualProbe:
    """
    Evaluates a model's mathematical reasoning across multiple languages.

    Designed to answer: "Does math fine-tuning (e.g., via GRPO) improve
    reasoning in low-resource languages, not just English?"

    Args:
        model:       HuggingFace causal LM
        tokenizer:   HuggingFace tokenizer
        model_name:  Identifier for reports
        device:      "cuda" or "cpu"
    """

    def __init__(self, model, tokenizer, model_name: str, device: str = "cuda"):
        self.model      = model.to(device)
        self.tok        = tokenizer
        self.model_name = model_name
        self.device     = device

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()

    def _format_prompt(self, question: str, lang: str) -> str:
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])
        return template.format(question=question)

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> tuple[str, float]:
        enc = self.tok(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        t0 = time.time()
        out = self.model.generate(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            max_new_tokens = max_new_tokens,
            do_sample      = False,   # Greedy for evaluation
            pad_token_id   = self.tok.pad_token_id,
        )
        latency = time.time() - t0

        completion = self.tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        return completion, latency

    def run(
        self,
        translated_problems: list[TranslatedProblem],
        languages: list[str],
        max_new_tokens: int = 256,
    ) -> MultilingualReport:
        """
        Evaluate the model on each language for all problems.

        Args:
            translated_problems: Output of translate_dataset()
            languages:           Which languages to evaluate (e.g., ["en", "yo", "ig"])
            max_new_tokens:      Max generation length

        Returns:
            MultilingualReport with per-language accuracy breakdown
        """
        import datetime

        all_results = {lang: [] for lang in languages}

        for problem in tqdm(translated_problems, desc="Running probe"):
            for lang in languages:
                question = problem.translations.get(lang, problem.question_en)
                prompt   = self._format_prompt(question, lang)

                output, latency = self._generate(prompt, max_new_tokens)
                extracted       = extract_answer(output)
                correct         = answers_match(extracted, problem.answer)

                all_results[lang].append(ProbeResult(
                    idx=problem.idx, lang=lang, question=question,
                    answer=problem.answer, output=output,
                    extracted=extracted, correct=correct, latency_s=latency,
                ))

        # Build per-language reports
        lang_reports = {}
        for lang, results in all_results.items():
            n       = len(results)
            correct = sum(r.correct for r in results)
            ext     = sum(r.extracted is not None for r in results)
            lang_reports[lang] = LanguageReport(
                lang=lang,
                lang_name=SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
                num_samples=n,
                num_correct=correct,
                accuracy=correct / n if n > 0 else 0.0,
                extraction_rate=ext / n if n > 0 else 0.0,
                avg_latency_s=sum(r.latency_s for r in results) / n if n > 0 else 0.0,
                sample_results=results,
            )

        return MultilingualReport(
            model_name=self.model_name,
            languages=languages,
            num_samples=len(translated_problems),
            results=lang_reports,
            timestamp=datetime.datetime.now().isoformat(),
        )

    def print_report(self, report: MultilingualReport):
        print("\n" + "="*60)
        print(f"  Multilingual GSM8K Probe")
        print(f"  Model: {report.model_name}")
        print(f"  N: {report.num_samples} problems per language")
        print("="*60)
        print(report.accuracy_table())
        print()

        # Key finding
        if "en" in report.results:
            en_acc = report.results["en"].accuracy
            for lang, r in report.results.items():
                if lang != "en":
                    diff = r.accuracy - en_acc
                    direction = "↑ better" if diff > 0.01 else ("↓ worse" if diff < -0.01 else "≈ same")
                    print(f"  {r.lang_name} vs English: {diff:+.1%} ({direction})")
        print()

    def save_report(self, report: MultilingualReport, output_dir: str = "results"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"{report.model_name.replace('/', '_')}_{report.timestamp[:10]}.json"

        # Serialize, excluding raw outputs to keep file size manageable
        summary = {
            "model_name":  report.model_name,
            "languages":   report.languages,
            "num_samples": report.num_samples,
            "timestamp":   report.timestamp,
            "results": {
                lang: {
                    "lang_name":      r.lang_name,
                    "accuracy":       r.accuracy,
                    "num_correct":    r.num_correct,
                    "num_samples":    r.num_samples,
                    "extraction_rate": r.extraction_rate,
                    "avg_latency_s":  r.avg_latency_s,
                    # Include first 10 samples for qualitative analysis
                    "sample_results": [asdict(s) for s in r.sample_results[:10]],
                }
                for lang, r in report.results.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {path}")
        return path
