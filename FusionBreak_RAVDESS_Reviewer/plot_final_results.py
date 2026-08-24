from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "final" / "final_results.json"
FIGURES = ROOT / "figures"


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)

    expanded = data["expanded_results"]
    labels = ["Extra Trees", "HistGradientBoosting"]
    targeted = [100 * row["targeted_rate"] for row in expanded]
    random_rates = [100 * row["random_rate"] for row in expanded]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar([i - width / 2 for i in x], targeted, width, label="YASMIN")
    ax.bar([i + width / 2 for i in x], random_rates, width, label="Random schedules")
    ax.set_ylabel("Alarm suppression rate (%)")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 75)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "expanded_suppression_rates.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "expanded_suppression_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    confirm = data["confirmatory_results"]
    d8 = 100 * confirm[0]["targeted_rate"]
    d4 = 100 * confirm[2]["targeted_rate"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.bar(["d = 8", "d = 4"], [d8, d4])
    ax.set_ylabel("Targeted suppression rate (%)")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_displacement_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "paired_displacement_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures under {FIGURES}")


if __name__ == "__main__":
    main()
