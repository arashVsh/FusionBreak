#!/usr/bin/env python3
"""
Compute CLEAN test accuracies for the exact CREMA-D configurations used in
the final FusionBreak USENIX experiments.

This script:
  * DOES NOT train models.
  * DOES NOT run adversarial attacks.
  * Reuses the same dataset splitting, packed-cache preprocessing, model
    classes, checkpoint loader, and pinned configuration as the final
    controlled CREMA-D experiment.
  * Evaluates seeds 42, 43, 44 and the four paper fusion architectures.
  * Verifies the all-three-correct eligibility counts against the completed
    attack results used in the paper. If these counts differ, the script stops
    rather than reporting clean accuracies from a different experimental state.

Place this file in the CREMA-D project root beside:
    run_cremad_usenix_final_consistent.py
    run_cremad_controlled_full_vs_adaptive.py
    fusiongap_cremad/
    CREMA-D/   (or the dataset directory recognized by Config)

Expected controlled checkpoints:
    fusion_gap_cremad_modular_outputs/large_files/checkpoints/
        audio_cremad_v2_seed{seed}.pt
        video_cremad_v2_seed{seed}.pt
        fusion_cremad_v2_seed{seed}_{arch}.pt

Run:
    python compute_cremad_clean_accuracies.py

Optional GPU selection:
    PowerShell:
        $env:FUSIONGAP_DEVICE="cuda:0"
        python compute_cremad_clean_accuracies.py

    Linux/macOS:
        FUSIONGAP_DEVICE=cuda:0 python compute_cremad_clean_accuracies.py

Output:
    fusion_gap_cremad_modular_outputs/logs/
        usenix_clean_accuracies_v1_cremad.json
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch

from fusiongap_cremad.config import Config
from fusiongap_cremad.data import (
    build_or_load_packed_cache,
    scan_records,
    split_records,
)
from fusiongap_cremad.train import clean_evaluation
from fusiongap_cremad.utils import set_seed

# Reuse the exact checkpoint/model loader and pinned configuration used by
# the completed final experiment.
from run_cremad_controlled_full_vs_adaptive import (
    _configure_controlled_attack,
    _load_controlled_models,
)
from run_cremad_usenix_final_consistent import _validate_pinned_config


PAPER_SEEDS = (42, 43, 44)
PAPER_ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")

# These are the all-three-correct TEST counts from the completed final
# controlled CREMA-D results used in the paper. They are independent of
# attack method and therefore provide a strong guard that this clean
# evaluation is using the same split + checkpoints as the reported attacks.
EXPECTED_ELIGIBILITY_COUNTS = {
    (42, "concat"): 330,
    (42, "gated"): 304,
    (42, "group_gated"): 317,
    (42, "interaction"): 302,
    (43, "concat"): 266,
    (43, "gated"): 225,
    (43, "group_gated"): 258,
    (43, "interaction"): 241,
    (44, "concat"): 352,
    (44, "gated"): 334,
    (44, "group_gated"): 322,
    (44, "interaction"): 347,
}
EXPECTED_TEST_SAMPLES = 1487


def select_device() -> torch.device:
    requested = os.environ.get("FUSIONGAP_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested {requested}, but CUDA is unavailable.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {requested}, but only {torch.cuda.device_count()} "
                    "CUDA device(s) are visible."
                )
            torch.cuda.set_device(device)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def sample_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list.")
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def percent_summary(values: list[float]) -> dict[str, float]:
    s = sample_summary(values)
    return {k: 100.0 * v for k, v in s.items()}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = Config(project_root=project_root)

    # Populate the exact additional settings used by the controlled final run,
    # then verify that they match the locked final protocol.
    _configure_controlled_attack(cfg)
    cfg.ensure_dirs()
    _validate_pinned_config(cfg)

    device = select_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    records = scan_records(cfg)
    per_config: list[dict[str, Any]] = []
    mismatches: list[str] = []

    for seed in PAPER_SEEDS:
        print("\n" + "=" * 80)
        print(f"Seed {seed}")
        print("=" * 80)

        set_seed(seed)
        train_records, val_records, test_records = split_records(records, seed, cfg)

        if len(test_records) != EXPECTED_TEST_SAMPLES:
            raise RuntimeError(
                f"Seed {seed}: expected {EXPECTED_TEST_SAMPLES} test records, "
                f"found {len(test_records)}."
            )

        # Only the test cache is needed. If the final experiment's packed cache
        # already exists, this should load it rather than recomputing features.
        test_cache = build_or_load_packed_cache(
            test_records, "test", seed, cfg, device
        )

        for arch in PAPER_ARCHITECTURES:
            print(f"\n[clean] seed={seed}, arch={arch}")
            set_seed(seed)

            # This loader refuses to train missing checkpoints.
            audio, video, fusion = _load_controlled_models(
                seed, arch, cfg, device
            )

            metrics = clean_evaluation(
                audio, video, fusion, test_cache, cfg, device
            )

            expected_eligible = EXPECTED_ELIGIBILITY_COUNTS[(seed, arch)]
            observed_eligible = int(metrics["all_three_correct_count"])
            observed_n = int(metrics["test_samples"])

            if observed_n != EXPECTED_TEST_SAMPLES:
                mismatches.append(
                    f"seed={seed}, arch={arch}: test_samples "
                    f"{observed_n} != {EXPECTED_TEST_SAMPLES}"
                )
            if observed_eligible != expected_eligible:
                mismatches.append(
                    f"seed={seed}, arch={arch}: all_three_correct_count "
                    f"{observed_eligible} != expected {expected_eligible}"
                )

            row = {
                "seed": seed,
                "fusion_architecture": arch,
                "test_samples": observed_n,
                "audio_acc": float(metrics["audio_acc"]),
                "video_acc": float(metrics["video_acc"]),
                "fusion_acc": float(metrics["fusion_acc"]),
                "all_three_correct_count": observed_eligible,
                "all_three_correct_rate": float(
                    metrics["all_three_correct_rate"]
                ),
                "expected_all_three_correct_count": expected_eligible,
                "eligibility_guard_passed": (
                    observed_n == EXPECTED_TEST_SAMPLES
                    and observed_eligible == expected_eligible
                ),
            }
            per_config.append(row)

            print(
                "  audio={:.2f}% | video={:.2f}% | fusion={:.2f}% | "
                "eligible={}/{} ({:.2f}%) | guard={}".format(
                    100 * row["audio_acc"],
                    100 * row["video_acc"],
                    100 * row["fusion_acc"],
                    row["all_three_correct_count"],
                    row["test_samples"],
                    100 * row["all_three_correct_rate"],
                    "PASS" if row["eligibility_guard_passed"] else "FAIL",
                )
            )

            del audio, video, fusion
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Never silently report numbers if they do not reproduce the exact
    # all-three-correct pools used by the paper.
    if mismatches:
        print("\nCONSISTENCY GUARD FAILED:")
        for msg in mismatches:
            print("  -", msg)
        raise RuntimeError(
            "Clean evaluation does not reproduce the eligibility counts from "
            "the final paper experiments. Do not use these clean accuracies."
        )

    audio_vals = [r["audio_acc"] for r in per_config]
    video_vals = [r["video_acc"] for r in per_config]
    fusion_vals = [r["fusion_acc"] for r in per_config]
    eligible_counts = [float(r["all_three_correct_count"]) for r in per_config]
    eligible_rates = [r["all_three_correct_rate"] for r in per_config]

    # Paper convention: unweighted mean +/- SAMPLE SD across the same
    # 12 seed x architecture configurations used for the attack tables.
    aggregate = {
        "aggregation_note": (
            "Mean and sample SD are unweighted across the 12 "
            "seed x architecture configurations."
        ),
        "configs": len(per_config),
        "clean_test_accuracy_percent": {
            "audio": percent_summary(audio_vals),
            "video": percent_summary(video_vals),
            "fusion": percent_summary(fusion_vals),
        },
        "eligibility_count": sample_summary(eligible_counts),
        "eligibility_rate_percent": percent_summary(eligible_rates),
    }

    # Audio/video checkpoints are shared across the four architectures for
    # a given seed. Also report a deduplicated 3-seed summary for reference,
    # while keeping the 12-config summary as the paper-compatible one.
    per_seed_unimodal = []
    for seed in PAPER_SEEDS:
        seed_rows = [r for r in per_config if r["seed"] == seed]
        a = {round(r["audio_acc"], 15) for r in seed_rows}
        v = {round(r["video_acc"], 15) for r in seed_rows}
        if len(a) != 1 or len(v) != 1:
            raise RuntimeError(
                f"Seed {seed}: unimodal clean accuracies differ across "
                "fusion architectures, which is unexpected."
            )
        per_seed_unimodal.append(
            {
                "seed": seed,
                "audio_acc": seed_rows[0]["audio_acc"],
                "video_acc": seed_rows[0]["video_acc"],
            }
        )

    aggregate["unimodal_unique_seed_accuracy_percent"] = {
        "audio": percent_summary([r["audio_acc"] for r in per_seed_unimodal]),
        "video": percent_summary([r["video_acc"] for r in per_seed_unimodal]),
        "note": (
            "This deduplicates the repeated unimodal checkpoints across the "
            "four fusion architectures. Use the 12-config summary above if "
            "you want aggregation to match the paper's main-table convention."
        ),
    }

    result = {
        "dataset": "CREMA-D",
        "purpose": "clean-test accuracy audit for final USENIX configurations",
        "seeds": list(PAPER_SEEDS),
        "fusion_architectures": list(PAPER_ARCHITECTURES),
        "consistency_guard_passed": True,
        "per_config": per_config,
        "per_seed_unimodal": per_seed_unimodal,
        "aggregate": aggregate,
    }

    out_path = cfg.log_dir / "usenix_clean_accuracies_v1_cremad.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    a = aggregate["clean_test_accuracy_percent"]["audio"]
    v = aggregate["clean_test_accuracy_percent"]["video"]
    f = aggregate["clean_test_accuracy_percent"]["fusion"]
    ec = aggregate["eligibility_count"]
    er = aggregate["eligibility_rate_percent"]

    print("\n" + "=" * 80)
    print("CREMA-D CLEAN TEST ACCURACIES — PAPER-COMPATIBLE 12-CONFIG AGGREGATE")
    print("=" * 80)
    print(f"Audio : {a['mean']:.2f} +/- {a['sd']:.2f}%")
    print(f"Video : {v['mean']:.2f} +/- {v['sd']:.2f}%")
    print(f"Fusion: {f['mean']:.2f} +/- {f['sd']:.2f}%")
    print(
        f"Eligible count: {ec['mean']:.2f} +/- {ec['sd']:.2f} "
        f"out of {EXPECTED_TEST_SAMPLES}"
    )
    print(
        f"Eligible rate : {er['mean']:.2f} +/- {er['sd']:.2f}% "
        f"(range {er['min']:.2f}%-{er['max']:.2f}%)"
    )
    print("Eligibility consistency guard: PASS")
    print(f"Saved: {out_path}")

    print("\nPaper-ready sentence:")
    print(
        "On CREMA-D, the audio, visual, and fused classifiers achieve "
        f"mean clean test accuracies of {a['mean']:.2f} +/- {a['sd']:.2f}%, "
        f"{v['mean']:.2f} +/- {v['sd']:.2f}%, and "
        f"{f['mean']:.2f} +/- {f['sd']:.2f}%, respectively."
    )


if __name__ == "__main__":
    main()
