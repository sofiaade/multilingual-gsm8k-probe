# Multilingual GSM8K Probe

Does fine-tuning a language model for mathematical reasoning in English improve its reasoning in **Yoruba, Igbo, and Hausa** — West African languages that are severely underrepresented in NLP research?

This probe tests that research question empirically, motivated by the practical goal of making mathematical AI capabilities more accessible to speakers of West African languages (~200M+ speakers combined).

---

## Research Question

> If a model learns to reason mathematically (not just pattern-match English math text), the capability should transfer across languages — even low-resource ones the model has seen little training data for.

**Hypothesis:** GRPO fine-tuning on English GSM8K → accuracy gains in Yoruba and Igbo, not just English.

**Null hypothesis:** Gains are English-specific (language-surface-level, not reasoning-level).

---

## Languages

| Language | Code | Speakers | Family | Notes |
|---|---|---|---|---|
| English | `en` | 1.5B | Indo-European | Baseline |
| Yoruba | `yo` | ~100M | Niger-Congo | Nigeria, Benin, Togo |
| Igbo | `ig` | ~45M | Niger-Congo | Southeastern Nigeria |
| Hausa | `ha` | ~70M | Afro-Asiatic | Northern Nigeria, Niger |
| Swahili | `sw` | ~200M | Niger-Congo | East Africa (higher-resource control) |

---

## Quickstart

```bash
pip install -r requirements.txt

# Demo run (GPT-2, 50 samples, English + Yoruba + Igbo)
python scripts/run_probe.py \
    --base_model gpt2 \
    --num_samples 50 \
    --langs en yo ig

# Research run (compare base vs. GRPO fine-tuned)
python scripts/run_probe.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --finetuned_model path/to/grpo-checkpoint \
    --num_samples 200 \
    --langs en yo ig ha sw
```

---

## Translation

Problems are translated using Google Translate (free via `deep-translator`). Translations are **cached to disk** in `data/translation_cache.json` — subsequent runs reuse cached translations.

```bash
pip install deep-translator
```

**Limitation:** Google Translate quality for Yoruba and Igbo is imperfect. For research publication, consider human-verified translations or a dedicated NMT system.

---

## Output

The probe saves a JSON report per model:
```json
{
  "model_name": "base_gpt2",
  "results": {
    "en": {"accuracy": 0.04, "num_correct": 2, "num_samples": 50},
    "yo": {"accuracy": 0.02, "num_correct": 1, "num_samples": 50},
    "ig": {"accuracy": 0.02, "num_correct": 1, "num_samples": 50}
  }
}
```

And generates a grouped bar chart comparing base vs. fine-tuned accuracy per language.

---

## Expected Results (Hypothesis)

| Model | English | Yoruba | Igbo | Interpretation |
|---|---|---|---|---|
| Base | X% | ≈X% | ≈X% | Baseline |
| GRPO fine-tuned | X+N% | X+M% | X+M% | Transfer if M > 0 |

If **M > 0**: supports reasoning transfer hypothesis.
If **M ≈ 0**: suggests English-surface fine-tuning, not deep reasoning.

*Fill this table with your actual results.*

---

## Citation

If you use this probe in research, please cite:

> [Your name]. "Multilingual Transfer of Mathematical Reasoning: A Probe on West African Languages." [Year].

And the original GSM8K dataset:
```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
