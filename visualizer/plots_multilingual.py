"""
Multilingual Results Visualization
====================================
Plots comparing model performance across languages.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_multilingual_comparison(base_report, finetuned_report, save_path=None, show=True):
    """
    Grouped bar chart: base vs. fine-tuned accuracy per language.

    The key research finding: does GRPO fine-tuning help Yoruba and Igbo,
    or does it only help English?
    """
    languages  = base_report.languages
    lang_names = [base_report.results[l].lang_name for l in languages]

    base_acc     = [base_report.results[l].accuracy     for l in languages]
    finetuned_acc = [finetuned_report.results[l].accuracy for l in languages]

    x      = np.arange(len(languages))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, base_acc,      width, label="Base model",       color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, finetuned_acc, width, label="GRPO fine-tuned",  color="#2ecc71", alpha=0.85)

    # Annotate with delta
    for i, (b, f) in enumerate(zip(base_acc, finetuned_acc)):
        delta = f - b
        color = "#27ae60" if delta > 0 else "#e74c3c"
        ax.text(x[i] + width/2, f + 0.01, f"{delta:+.1%}",
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lang_names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy on GSM8K")
    ax.set_title(
        f"Multilingual GSM8K: Base vs. GRPO Fine-tuned\n"
        f"(Does math fine-tuning transfer across languages?)"
    )
    ax.legend()
    ax.set_ylim(0, min(1.0, max(base_acc + finetuned_acc) + 0.15))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add language family annotations
    ax.axvspan(-0.5, 0.5, alpha=0.04, color="blue",  label="Indo-European")
    ax.axvspan(0.5,  len(languages) - 0.5, alpha=0.04, color="green", label="Niger-Congo")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_accuracy_radar(report, save_path=None, show=True):
    """
    Radar/spider chart showing accuracy across all languages for one model.
    """
    languages  = report.languages
    lang_names = [report.results[l].lang_name for l in languages]
    accuracies = [report.results[l].accuracy   for l in languages]

    # Close the radar loop
    N      = len(languages)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    accuracies_plot = accuracies + accuracies[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, accuracies_plot, "o-", linewidth=2, color="#3498db")
    ax.fill(angles, accuracies_plot, alpha=0.2, color="#3498db")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(lang_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"{report.model_name}\nAccuracy by Language", pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
