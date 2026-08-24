#!/usr/bin/env python3
"""
Audit and aggregate the clean RAVDESS accuracies already stored in the
12 final USENIX JSON files.

This does NOT rerun models. The final RAVDESS result JSONs already contain
clean_test.audio_acc, clean_test.video_acc, clean_test.fusion_acc, and the
all-three-correct eligibility statistics.

Place this script in the directory containing:
    usenix_final_allmethods_v1_ravdess_seed42_*.json
    usenix_final_allmethods_v1_ravdess_seed43_*.json
    usenix_final_allmethods_v1_ravdess_seed44_*.json

Run:
    python audit_ravdess_clean_accuracies.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")

# Values currently stated in the paper for RAVDESS eligibility.
PAPER_EXPECTED = {
    "eligible_count_mean": 33.25,
    "eligible_count_sd": 4.11,
    "eligible_rate_mean_percent": 15.39,
    "eligible_rate_sd_percent": 1.91,
    "eligible_rate_min_percent": 13.89,
    "eligible_rate_max_percent": 18.52,
}


def sample_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def percent_summary(values: list[float]) -> dict[str, float]:
    s = sample_summary(values)
    return {k: 100.0 * v for k, v in s.items()}


def rounded_equal(actual: float, expected: float, digits: int = 2) -> bool:
    return round(actual, digits) == round(expected, digits)


def main() -> None:
    root = Path(__file__).resolve().parent

    # Final RAVDESS runners save their JSONs under:
    #   <project-root>/fusion_gap_ravdess_multimodal_outputs/logs
    # rather than directly in the project root.
    candidate_dirs = [
        root / "fusion_gap_ravdess_multimodal_outputs" / "logs",
        root,
    ]

    paths = []
    result_dir = None
    for d in candidate_dirs:
        if not d.exists():
            continue
        found = sorted(d.glob("usenix_final_allmethods_v1_ravdess_seed*.json"))
        if found:
            paths = found
            result_dir = d
            break

    # Last-resort recursive search, useful if the output folder was moved.
    if not paths:
        paths = sorted(root.rglob("usenix_final_allmethods_v1_ravdess_seed*.json"))
        if paths:
            result_dir = paths[0].parent

    if not paths:
        raise RuntimeError(
            "Could not find any final RAVDESS result JSONs. Expected files like "
            "'usenix_final_allmethods_v1_ravdess_seed42_concat.json' under "
            f"'{root / 'fusion_gap_ravdess_multimodal_outputs' / 'logs'}'."
        )

    print(f"Reading final RAVDESS JSONs from: {result_dir}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset") != "RAVDESS":
            continue

        seed = int(data["seed"])
        arch = str(data["fusion_architecture"])
        key = (seed, arch)
        if seed not in SEEDS or arch not in ARCHITECTURES:
            continue
        if key in seen:
            raise RuntimeError(f"Duplicate RAVDESS configuration: {key}")
        seen.add(key)

        clean = data["clean_test"]
        elig = data["eligibility"]

        if int(clean["samples"]) != int(elig["test_samples"]):
            raise RuntimeError(f"{path.name}: clean/test sample count mismatch.")
        if int(clean["all_three_correct_count"]) != int(
            elig["all_three_correct_count"]
        ):
            raise RuntimeError(f"{path.name}: eligibility count mismatch.")
        if abs(
            float(clean["all_three_correct_rate"])
            - float(elig["all_three_correct_rate"])
        ) > 1e-12:
            raise RuntimeError(f"{path.name}: eligibility rate mismatch.")

        rows.append(
            {
                "seed": seed,
                "fusion_architecture": arch,
                "test_samples": int(clean["samples"]),
                "audio_acc": float(clean["audio_acc"]),
                "video_acc": float(clean["video_acc"]),
                "fusion_acc": float(clean["fusion_acc"]),
                "all_three_correct_count": int(
                    clean["all_three_correct_count"]
                ),
                "all_three_correct_rate": float(
                    clean["all_three_correct_rate"]
                ),
            }
        )

    expected_keys = {
        (seed, arch) for seed in SEEDS for arch in ARCHITECTURES
    }
    missing = sorted(expected_keys - seen)
    extra = sorted(seen - expected_keys)
    if missing or extra or len(rows) != 12:
        raise RuntimeError(
            f"Expected exactly 12 final configurations. "
            f"Missing={missing}, extra={extra}, found={len(rows)}"
        )

    rows.sort(key=lambda r: (r["seed"], r["fusion_architecture"]))

    audio = [r["audio_acc"] for r in rows]
    video = [r["video_acc"] for r in rows]
    fusion = [r["fusion_acc"] for r in rows]
    eligible_count = [float(r["all_three_correct_count"]) for r in rows]
    eligible_rate = [r["all_three_correct_rate"] for r in rows]

    aggregate = {
        "clean_test_accuracy_percent": {
            "audio": percent_summary(audio),
            "video": percent_summary(video),
            "fusion": percent_summary(fusion),
        },
        "eligibility_count": sample_summary(eligible_count),
        "eligibility_rate_percent": percent_summary(eligible_rate),
    }

    ec = aggregate["eligibility_count"]
    er = aggregate["eligibility_rate_percent"]

    checks = {
        "eligible_count_mean": rounded_equal(
            ec["mean"], PAPER_EXPECTED["eligible_count_mean"]
        ),
        "eligible_count_sd": rounded_equal(
            ec["sd"], PAPER_EXPECTED["eligible_count_sd"]
        ),
        "eligible_rate_mean_percent": rounded_equal(
            er["mean"], PAPER_EXPECTED["eligible_rate_mean_percent"]
        ),
        "eligible_rate_sd_percent": rounded_equal(
            er["sd"], PAPER_EXPECTED["eligible_rate_sd_percent"]
        ),
        "eligible_rate_min_percent": rounded_equal(
            er["min"], PAPER_EXPECTED["eligible_rate_min_percent"]
        ),
        "eligible_rate_max_percent": rounded_equal(
            er["max"], PAPER_EXPECTED["eligible_rate_max_percent"]
        ),
    }

    # Unimodal checkpoints are shared across architectures within a seed.
    per_seed_unimodal = []
    for seed in SEEDS:
        seed_rows = [r for r in rows if r["seed"] == seed]
        a = {round(r["audio_acc"], 15) for r in seed_rows}
        v = {round(r["video_acc"], 15) for r in seed_rows}
        if len(a) != 1 or len(v) != 1:
            raise RuntimeError(
                f"Seed {seed}: unimodal accuracies unexpectedly differ "
                "across fusion architectures."
            )
        per_seed_unimodal.append(
            {
                "seed": seed,
                "audio_acc": seed_rows[0]["audio_acc"],
                "video_acc": seed_rows[0]["video_acc"],
            }
        )

    aggregate["unimodal_unique_seed_accuracy_percent"] = {
        "audio": percent_summary(
            [r["audio_acc"] for r in per_seed_unimodal]
        ),
        "video": percent_summary(
            [r["video_acc"] for r in per_seed_unimodal]
        ),
    }

    result = {
        "dataset": "RAVDESS",
        "configs": 12,
        "paper_eligibility_consistency_passed": all(checks.values()),
        "paper_checks": checks,
        "per_config": rows,
        "per_seed_unimodal": per_seed_unimodal,
        "aggregate": aggregate,
    }

    out_path = (result_dir or root) / "usenix_clean_accuracies_audit_v1_ravdess.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    a = aggregate["clean_test_accuracy_percent"]["audio"]
    v = aggregate["clean_test_accuracy_percent"]["video"]
    f = aggregate["clean_test_accuracy_percent"]["fusion"]

    print("RAVDESS clean test accuracy across the 12 paper configurations")
    print(f"Audio : {a['mean']:.2f} +/- {a['sd']:.2f}%")
    print(f"Video : {v['mean']:.2f} +/- {v['sd']:.2f}%")
    print(f"Fusion: {f['mean']:.2f} +/- {f['sd']:.2f}%")
    print()
    print(
        f"Eligibility count: {ec['mean']:.2f} +/- {ec['sd']:.2f}"
    )
    print(
        f"Eligibility rate : {er['mean']:.2f} +/- {er['sd']:.2f}% "
        f"(range {er['min']:.2f}%-{er['max']:.2f}%)"
    )
    print(
        "Paper eligibility consistency:",
        "PASS" if all(checks.values()) else "FAIL",
    )
    if not all(checks.values()):
        print("Checks:", checks)
        raise SystemExit(1)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
