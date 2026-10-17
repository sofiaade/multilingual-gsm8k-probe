"""
Run the Multilingual GSM8K Probe.

Compare a base model vs. a GRPO fine-tuned model across English, Yoruba, and Igbo.

Run (demo with GPT-2):
    python scripts/run_probe.py \
        --base_model gpt2 \
        --num_samples 50 \
        --langs en yo ig

Run (research):
    python scripts/run_probe.py \
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --finetuned_model path/to/grpo-checkpoint \
        --num_samples 200 \
        --langs en yo ig ha sw
"""

import argparse
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from probe.multilingual_probe import (
    MultilingualProbe,
    TranslationCache,
    translate_dataset,
    SUPPORTED_LANGUAGES,
)
from visualizer.plots_multilingual import plot_multilingual_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",      default="gpt2")
    parser.add_argument("--finetuned_model", default=None,  help="Optional: compare against fine-tuned")
    parser.add_argument("--num_samples",     type=int, default=50)
    parser.add_argument("--langs",           nargs="+", default=["en", "yo", "ig"])
    parser.add_argument("--cache_dir",       default="data/translation_cache.json")
    parser.add_argument("--output_dir",      default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load and translate data ---
    print(f"\nLoading {args.num_samples} GSM8K test problems...")
    dataset  = load_dataset("openai/gsm8k", "main", split="test")
    dataset  = dataset.select(range(min(args.num_samples, len(dataset))))
    problems = [{"question": d["question"], "answer": d["answer"]} for d in dataset]

    print(f"Translating to: {args.langs}")
    cache = TranslationCache(args.cache_dir)
    translated = translate_dataset(problems, args.langs, cache)
    print(f"Translation complete ({len(translated)} problems × {len(args.langs)} languages)")

    # --- Evaluate models ---
    all_reports = {}

    for model_label, model_name in [
        ("base", args.base_model),
        ("finetuned", args.finetuned_model),
    ]:
        if model_name is None:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_label} ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        probe  = MultilingualProbe(model, tokenizer, model_name=f"{model_label}_{model_name}", device=device)
        report = probe.run(translated, languages=args.langs)

        probe.print_report(report)
        probe.save_report(report, args.output_dir)
        all_reports[model_label] = report

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Comparison plot ---
    if len(all_reports) > 1:
        try:
            plot_multilingual_comparison(
                all_reports["base"],
                all_reports["finetuned"],
                save_path=f"{args.output_dir}/multilingual_comparison.png",
                show=False,
            )
            print(f"\nComparison plot saved to {args.output_dir}/multilingual_comparison.png")
        except ImportError:
            pass  # matplotlib optional

    # --- Print summary table ---
    print("\n" + "="*60)
    print("SUMMARY: Accuracy by Language")
    print("="*60)
    for model_label, report in all_reports.items():
        print(f"\n{model_label.upper()}: {report.model_name}")
        print(report.accuracy_table())


if __name__ == "__main__":
    main()
