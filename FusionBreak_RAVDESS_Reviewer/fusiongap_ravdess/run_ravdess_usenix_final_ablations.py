from __future__ import annotations

"""
Final consistency-locked RAVDESS ablation runner for the FusionBreak USENIX paper.

Why this exists
---------------
The main RAVDESS results were rerun with the final equal-budget 3-seed protocol.
The paper's ablation tables were produced earlier. This script reruns those
reported seed-42 ablations using the SAME checkpoints, eligible samples, base
FusionBreak objective, numerical settings, and preprocessing as the final
RAVDESS runner.

Important consistency properties
--------------------------------
- Seed is fixed to 42, matching the paper's ablation section.
- Four architectures: interaction, concat, gated, group_gated.
- Existing final RAVDESS checkpoints are reused. Missing checkpoints are fatal.
- The exact final eligible pool is reused and checked against the completed
  final-consistent result JSON for each architecture.
- The base reference uses exactly the same FusionBreak implementation as
  run_ravdess_usenix_final_consistent.py.
- The script reproduces the final runner's pre-attack detector fitting and RNG
  progression, then verifies the new Full FusionBreak counts EXACTLY match the
  already completed final-consistent seed-42 FusionBreak counts.
- If the reference does not match, the run aborts rather than mixing results.

Paper ablations rerun
---------------------
Objective / selection:
  * Full FusionBreak
  * Fusion CE only
  * Fusion CE + margin
  * Without preservation
  * Without fused margin
  * Without distillation
  * Final-iterate selection

Attack settings:
  * Half budget
  * Default budget (reuses Full FusionBreak)
  * 1.5x budget
  * One restart
  * Five restarts
  * Ten restarts (reuses Full FusionBreak)
  * Half steps
  * Default steps (reuses Full FusionBreak)
  * 1.5x steps

Run from the RAVDESS project root, beside run_ravdess_usenix_final_consistent.py.

PowerShell:
    $env:FUSIONGAP_DEVICE="cuda:1"   # use cuda:0 if needed
    python .\run_ravdess_usenix_final_ablations.py

Optional subset smoke test:
    $env:FUSIONGAP_ARCHS="concat"
    python .\run_ravdess_usenix_final_ablations.py

Completed architecture JSON files are skipped unless FUSIONGAP_FORCE=1.
"""

import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fusiongap_ravdess.config import Config
from fusiongap_ravdess.data import build_or_load_packed_cache, scan_records, split_records
from fusiongap_ravdess.detectors import collect_detector_scores, fit_detector_pack
from fusiongap_ravdess.utils import save_json, set_seed

# Reuse the final paper runner as the single source of truth.
from run_ravdess_usenix_final_consistent import (
    ARCHITECTURES,
    EXPECTED_BASE_SETTINGS,
    PAPER_SEEDS,
    _cache_eval,
    _capture_rng,
    _checkpoint_manifest,
    _clamp_delta,
    _collect_clean_all_three,
    _config_fingerprint,
    _config_snapshot,
    _counts,
    _fusionbreak_attack,
    _kl_to_clean,
    _pin_paper_settings,
    _rates,
    _result_path as _final_main_result_path,
    _restore_rng,
    _select_device,
    _threshold_stats,
    _true_minus_other_margin,
    _validate_pinned_config,
    _zero_model_grads,
)

DATASET_NAME = "RAVDESS"
ABLATION_VERSION = 1
SEED = 42

# Unique runs. The three default rows in the settings table reuse full_fusionbreak.
VARIANTS: tuple[str, ...] = (
    "full_fusionbreak",
    "fusion_ce_only",
    "fusion_ce_plus_margin",
    "without_preservation",
    "without_fusion_margin",
    "without_distillation",
    "final_iterate_selection",
    "budget_half",
    "budget_one_and_half",
    "one_restart",
    "five_restarts",
    "half_steps",
    "one_and_half_steps",
)

VARIANT_FAMILY = {
    "full_fusionbreak": "reference",
    "fusion_ce_only": "objective",
    "fusion_ce_plus_margin": "objective",
    "without_preservation": "objective",
    "without_fusion_margin": "objective",
    "without_distillation": "objective",
    "final_iterate_selection": "candidate_selection",
    "budget_half": "budget",
    "budget_one_and_half": "budget",
    "one_restart": "restarts",
    "five_restarts": "restarts",
    "half_steps": "steps",
    "one_and_half_steps": "steps",
}

VARIANT_DESCRIPTION = {
    "full_fusionbreak": "Complete FusionBreak objective and decision-aware path selection.",
    "fusion_ce_only": "Fused cross-entropy only, with decision-aware intermediate candidate selection.",
    "fusion_ce_plus_margin": "Fused cross-entropy and fused attack margin only, with decision-aware intermediate candidate selection.",
    "without_preservation": "Complete method without the unimodal margin-preservation term.",
    "without_fusion_margin": "Complete method without the fused attack-margin term.",
    "without_distillation": "Complete method without unimodal logit distillation.",
    "final_iterate_selection": "Complete objective, but candidate selection is restricted to the terminal iterate of each restart.",
    "budget_half": "Half of each modality-specific perturbation budget, with step sizes scaled proportionally.",
    "budget_one_and_half": "One-and-a-half times each modality-specific perturbation budget, with step sizes scaled proportionally.",
    "one_restart": "Complete method with one restart.",
    "five_restarts": "Complete method with five restarts.",
    "half_steps": "Complete method with 40 optimization steps.",
    "one_and_half_steps": "Complete method with 120 optimization steps.",
}

# Values currently printed in the old manuscript tables. Used only for a delta audit.
OLD_PAPER_COMPONENT = {
    "fusion_ce_only": {"fusion_ASR": 1.0000, "audio_preservation_rate": 0.8259, "video_preservation_rate": 0.1357, "FOSR": 0.0967},
    "fusion_ce_plus_margin": {"fusion_ASR": 1.0000, "audio_preservation_rate": 0.8323, "video_preservation_rate": 0.1291, "FOSR": 0.0901},
    "without_preservation": {"fusion_ASR": 1.0000, "audio_preservation_rate": 0.8388, "video_preservation_rate": 0.1419, "FOSR": 0.1029},
    "without_fusion_margin": {"fusion_ASR": 0.8637, "audio_preservation_rate": 0.9614, "video_preservation_rate": 0.8523, "FOSR": 0.7160},
    "without_distillation": {"fusion_ASR": 0.9352, "audio_preservation_rate": 0.9418, "video_preservation_rate": 0.8518, "FOSR": 0.7805},
    "final_iterate_selection": {"fusion_ASR": 0.8900, "audio_preservation_rate": 0.2324, "video_preservation_rate": 0.4189, "FOSR": 0.0516},
    "full_fusionbreak": {"fusion_ASR": 0.9286, "audio_preservation_rate": 0.9546, "video_preservation_rate": 0.8581, "FOSR": 0.7805},
}

OLD_PAPER_SETTINGS = {
    "budget_half": {"fusion_ASR": 0.7869, "FOSR": 0.6703},
    "full_fusionbreak": {"fusion_ASR": 0.9286, "FOSR": 0.7805},
    "budget_one_and_half": {"fusion_ASR": 0.9934, "FOSR": 0.8191},
    "one_restart": {"fusion_ASR": 0.8772, "FOSR": 0.7099},
    "five_restarts": {"fusion_ASR": 0.9156, "FOSR": 0.7676},
    "half_steps": {"fusion_ASR": 0.9160, "FOSR": 0.7610},
    "one_and_half_steps": {"fusion_ASR": 0.9352, "FOSR": 0.7931},
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _selected_architectures() -> list[str]:
    raw = os.environ.get("FUSIONGAP_ARCHS", "").strip()
    archs = [x.strip() for x in raw.split(",") if x.strip()] if raw else list(ARCHITECTURES)
    bad = [a for a in archs if a not in ARCHITECTURES]
    if bad:
        raise ValueError(f"Invalid architecture(s): {bad}. Expected a subset of {ARCHITECTURES}.")
    return archs


def _clone_variant_cfg(base: Config, variant: str) -> Config:
    cfg = copy.deepcopy(base)
    if variant == "budget_half":
        cfg.eps_audio = 0.04
        cfg.eps_video = 0.03
        cfg.alpha_audio = 0.002
        cfg.alpha_video = 0.0015
    elif variant == "budget_one_and_half":
        cfg.eps_audio = 0.12
        cfg.eps_video = 0.09
        cfg.alpha_audio = 0.006
        cfg.alpha_video = 0.0045
    elif variant == "one_restart":
        cfg.random_restarts = 1
    elif variant == "five_restarts":
        cfg.random_restarts = 5
    elif variant == "half_steps":
        cfg.attack_steps = 40
    elif variant == "one_and_half_steps":
        cfg.attack_steps = 120
    return cfg


def _variant_toggles(variant: str) -> tuple[bool, bool, bool, bool]:
    """Return include_margin, include_preservation, include_distill, intermediate_selection."""
    if variant == "fusion_ce_only":
        return False, False, False, True
    if variant == "fusion_ce_plus_margin":
        return True, False, False, True
    if variant == "without_preservation":
        return True, False, True, True
    if variant == "without_fusion_margin":
        return False, True, True, True
    if variant == "without_distillation":
        return True, True, False, True
    if variant == "final_iterate_selection":
        return True, True, True, False
    # Budget/restart/step variants use the complete objective and normal selection.
    return True, True, True, True


def _generic_fusionbreak_ablation(
    audio,
    video,
    fusion,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    *,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    include_margin, include_preservation, include_distill, intermediate_selection = _variant_toggles(variant)

    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        clean_audio_logits = audio(xa0).detach() if include_distill else None
        clean_video_logits = video(xv0).detach() if include_distill else None
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        # Keep the clean example as the fallback reference. For the final-iterate
        # ablation, only terminal iterates are allowed to replace it.
        from run_ravdess_usenix_final_consistent import _candidate_score
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a: torch.Tensor, v: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        from run_ravdess_usenix_final_consistent import _candidate_score
        with torch.no_grad():
            score = _candidate_score(audio, video, fusion, a, v, y)
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = a[improve].detach()
                best_xv[improve] = v[improve].detach()

    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and cfg.clean_start_first_restart:
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _clamp_delta(
                xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio),
                xa0,
                cfg.eps_audio,
            ).detach()
            xv_adv = _clamp_delta(
                xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video),
                xv0,
                cfg.eps_video,
            ).detach()

        if intermediate_selection:
            maybe_update(xa_adv, xv_adv)

        for step in range(1, int(cfg.attack_steps) + 1):
            _zero_model_grads(audio, video, fusion)
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)

            fusion_logits = fusion(xa_adv, xv_adv)
            objective_vec = F.cross_entropy(fusion_logits, y, reduction="none")

            audio_logits = None
            video_logits = None
            if include_preservation or include_distill:
                audio_logits = audio(xa_adv)
                video_logits = video(xv_adv)

            if include_margin:
                fusion_attack_margin = -_true_minus_other_margin(fusion_logits, y)
                objective_vec = objective_vec + cfg.lambda_fusion_margin * fusion_attack_margin

            if include_preservation:
                assert audio_logits is not None and video_logits is not None
                audio_margin = _true_minus_other_margin(audio_logits, y)
                video_margin = _true_minus_other_margin(video_logits, y)
                preservation_hinge = (
                    F.relu(cfg.preserve_margin - audio_margin)
                    + F.relu(cfg.preserve_margin - video_margin)
                )
                objective_vec = objective_vec - cfg.lambda_preserve_margin * preservation_hinge

            if include_distill:
                assert audio_logits is not None and video_logits is not None
                assert clean_audio_logits is not None and clean_video_logits is not None
                logit_distill = (
                    _kl_to_clean(audio_logits, clean_audio_logits)
                    + _kl_to_clean(video_logits, clean_video_logits)
                )
                objective_vec = objective_vec - cfg.lambda_logit_distill * logit_distill

            objective = objective_vec.mean()
            ga, gv = torch.autograd.grad(objective, [xa_adv, xv_adv])
            with torch.no_grad():
                xa_adv = _clamp_delta(
                    xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                ).detach()
                xv_adv = _clamp_delta(
                    xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                ).detach()

            if intermediate_selection and (
                cfg.candidate_eval_interval <= 1
                or step % cfg.candidate_eval_interval == 0
                or step == cfg.attack_steps
            ):
                maybe_update(xa_adv, xv_adv)

        if not intermediate_selection:
            # Exact ablation intent: select only among terminal iterates.
            maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _run_variant(audio, video, fusion, xa, xv, y, base_cfg: Config, variant: str):
    cfg = _clone_variant_cfg(base_cfg, variant)
    if variant == "full_fusionbreak":
        return _fusionbreak_attack(audio, video, fusion, xa, xv, y, cfg, adaptive=False), cfg
    return _generic_fusionbreak_ablation(
        audio, video, fusion, xa, xv, y, cfg, variant=variant
    ), cfg


def _zero_metric_accumulator() -> dict[str, Any]:
    return {
        "counts": {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0},
        "audio_linf_sum": 0.0,
        "video_linf_sum": 0.0,
        "audio_l2_sum": 0.0,
        "video_l2_sum": 0.0,
        "audio_linf_max": 0.0,
        "video_linf_max": 0.0,
        "norm_n": 0,
        "elapsed_seconds": 0.0,
    }


def _accumulate_metrics(acc: dict[str, Any], audio, video, fusion, xa_adv, xv_adv, xa, xv, y, elapsed: float) -> None:
    c = _counts(audio, video, fusion, xa_adv, xv_adv, y)
    for k, v in c.items():
        acc["counts"][k] += int(v)

    with torch.no_grad():
        da = (xa_adv - xa).detach().flatten(1)
        dv = (xv_adv - xv).detach().flatten(1)
        a_inf = da.abs().max(1).values
        v_inf = dv.abs().max(1).values
        a_l2 = da.pow(2).sum(1).sqrt()
        v_l2 = dv.pow(2).sum(1).sqrt()
        acc["audio_linf_sum"] += float(a_inf.sum().item())
        acc["video_linf_sum"] += float(v_inf.sum().item())
        acc["audio_l2_sum"] += float(a_l2.sum().item())
        acc["video_l2_sum"] += float(v_l2.sum().item())
        acc["audio_linf_max"] = max(acc["audio_linf_max"], float(a_inf.max().item()))
        acc["video_linf_max"] = max(acc["video_linf_max"], float(v_inf.max().item()))
        acc["norm_n"] += int(y.numel())
        acc["elapsed_seconds"] += float(elapsed)


def _finish_metrics(acc: dict[str, Any], cfg: Config, variant: str, arch: str) -> dict[str, Any]:
    n = max(1, int(acc["norm_n"]))
    rates = _rates(acc["counts"])
    return {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "fusion_architecture": arch,
        "variant": variant,
        "ablation_family": VARIANT_FAMILY[variant],
        "attacked_clean_samples": int(acc["counts"]["n"]),
        **rates,
        "audio_linf_mean": acc["audio_linf_sum"] / n,
        "video_linf_mean": acc["video_linf_sum"] / n,
        "audio_linf_max": acc["audio_linf_max"],
        "video_linf_max": acc["video_linf_max"],
        "audio_l2_mean": acc["audio_l2_sum"] / n,
        "video_l2_mean": acc["video_l2_sum"] / n,
        "attack_steps": int(cfg.attack_steps),
        "random_restarts": int(cfg.random_restarts),
        "eps_audio": float(cfg.eps_audio),
        "eps_video": float(cfg.eps_video),
        "alpha_audio": float(cfg.alpha_audio),
        "alpha_video": float(cfg.alpha_video),
        "decision_aware_selection": variant != "final_iterate_selection",
        "elapsed_seconds": float(acc["elapsed_seconds"]),
        "description": VARIANT_DESCRIPTION[variant],
    }


def _consume_final_runner_post_batch_rng(xv: torch.Tensor, cfg: Config) -> None:
    """Match the RNG state left by the final runner's last method, VideoOnlyFusionBreak.

    In the final runner all methods restore the same per-batch RNG state. The last
    method is VideoOnlyFusionBreak. In eval mode its only stochastic operations are
    the random video starts for restarts 2..N. Consuming the same uniforms makes the
    next batch start from the same RNG state as the completed final run.
    """
    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and cfg.clean_start_first_restart:
            continue
        torch.empty_like(xv).uniform_(-cfg.eps_video, cfg.eps_video)


def _load_final_reference(cfg: Config, arch: str) -> dict[str, Any]:
    path = _final_main_result_path(cfg, SEED, arch)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing completed final-consistent result: {path}. Run run_ravdess_usenix_final_consistent.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("config_sha256")) != _config_fingerprint(_config_snapshot(cfg)):
        raise RuntimeError(f"Final reference config hash mismatch for {arch}.")
    return data


def _result_path(cfg: Config, arch: str) -> Path:
    return cfg.log_dir / f"usenix_final_ablation_v{ABLATION_VERSION}_ravdess_seed{SEED}_{arch}.json"


def _run_one_arch(arch: str, cfg: Config, device: torch.device) -> dict[str, Any]:
    out = _result_path(cfg, arch)
    if out.exists() and not _env_flag("FUSIONGAP_FORCE", False):
        print(f"[skip] {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"FINAL-CONSISTENT RAVDESS ABLATIONS: seed={SEED}, arch={arch}")
    print("=" * 100)

    # Reproduce the exact final runner pre-attack sequence.
    set_seed(SEED)
    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, SEED, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", SEED, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", SEED, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", SEED, cfg, device)

    checkpoint_manifest = _checkpoint_manifest(cfg, SEED, arch)
    final_ref = _load_final_reference(cfg, arch)

    # Check the exact checkpoint hashes before doing any attack work.
    for name in ("audio", "video", "fusion"):
        if checkpoint_manifest[name]["sha256"] != final_ref["checkpoint_manifest"][name]["sha256"]:
            raise RuntimeError(f"Checkpoint mismatch for {arch} {name}.")

    set_seed(SEED)
    from run_ravdess_usenix_final_consistent import _load_models
    audio, video, fusion = _load_models(SEED, arch, cfg, device)
    audio.eval(); video.eval(); fusion.eval()

    val_eval = _cache_eval(audio, video, fusion, val_cache, cfg, device)
    test_eval = _cache_eval(audio, video, fusion, test_cache, cfg, device)
    clean_pool = _collect_clean_all_three(audio, video, fusion, test_cache, cfg, device)
    if clean_pool is None:
        raise RuntimeError("No all-three-correct clean samples.")

    eligible_n = int(test_eval["all_three_correct_count"])
    attacked_n = int(clean_pool[2].numel())
    final_elig = final_ref["eligibility"]
    if int(final_elig["test_samples"]) != int(test_eval["samples"]):
        raise RuntimeError(f"Test-size mismatch for {arch}.")
    if int(final_elig["all_three_correct_count"]) != eligible_n or attacked_n != eligible_n:
        raise RuntimeError(
            f"Eligible-pool mismatch for {arch}: final={final_elig['all_three_correct_count']} current={eligible_n} attacked={attacked_n}"
        )

    # These calls are not needed for ablation metrics, but are intentionally kept
    # because the final main runner performs them before capturing attack RNG states.
    detector_pack = fit_detector_pack(fusion, train_cache, cfg, device)
    val_clean_scores = collect_detector_scores(
        fusion,
        detector_pack,
        val_cache["audio"].float(),
        val_cache["video"].float(),
        cfg,
        device,
    )
    _ = _threshold_stats(val_clean_scores, float(cfg.detector_target_fpr), device)

    xa_all, xv_all, y_all = clean_pool
    loader = DataLoader(
        TensorDataset(xa_all.float(), xv_all.float(), y_all.long()),
        batch_size=cfg.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    acc = {variant: _zero_metric_accumulator() for variant in VARIANTS}
    variant_cfgs = {variant: _clone_variant_cfg(cfg, variant) for variant in VARIANTS}

    bar = tqdm(loader, desc=f"RAVDESS ablations {arch}", unit="batch")
    for xa_cpu, xv_cpu, y_cpu in bar:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)
        matched_state = _capture_rng(device)

        for variant in VARIANTS:
            _restore_rng(matched_state, device)
            t0 = time.perf_counter()
            (xa_adv, xv_adv), vcfg = _run_variant(
                audio, video, fusion, xa, xv, y, cfg, variant
            )
            elapsed = time.perf_counter() - t0
            _accumulate_metrics(
                acc[variant], audio, video, fusion,
                xa_adv, xv_adv, xa, xv, y, elapsed
            )

        # Make the RNG state at the next batch match the completed final runner.
        _restore_rng(matched_state, device)
        _consume_final_runner_post_batch_rng(xv, cfg)

        bar.set_postfix(
            full=f"{100*_rates(acc['full_fusionbreak']['counts'])['FOSR']:.1f}%",
            ce=f"{100*_rates(acc['fusion_ce_only']['counts'])['FOSR']:.1f}%",
        )

    rows = [
        _finish_metrics(acc[v], variant_cfgs[v], v, arch)
        for v in VARIANTS
    ]

    # Hard reference check against the already completed final-consistent main run.
    new_full = next(r for r in rows if r["variant"] == "full_fusionbreak")
    expected_counts = final_ref["attack_metrics"]["FusionBreak"]["counts"]
    actual_counts = acc["full_fusionbreak"]["counts"]
    reference_exact_match = actual_counts == expected_counts
    if not reference_exact_match:
        raise RuntimeError(
            "REFERENCE REPRODUCTION FAILED for " + arch + "\n"
            + f"expected counts={expected_counts}\nactual counts={actual_counts}\n"
            + "Do not use these ablations until the RNG/pre-attack sequence is reconciled."
        )

    tol = 1e-5
    for row in rows:
        if row["audio_linf_max"] > row["eps_audio"] + tol:
            raise RuntimeError(f"Audio budget violation in {arch} {row['variant']}")
        if row["video_linf_max"] > row["eps_video"] + tol:
            raise RuntimeError(f"Video budget violation in {arch} {row['variant']}")
        if int(row["attacked_clean_samples"]) != eligible_n:
            raise RuntimeError(f"Eligible coverage mismatch in {arch} {row['variant']}")

    result = {
        "dataset": DATASET_NAME,
        "result_version": ABLATION_VERSION,
        "seed": SEED,
        "fusion_architecture": arch,
        "config_sha256": _config_fingerprint(_config_snapshot(cfg)),
        "checkpoint_manifest": checkpoint_manifest,
        "test_samples": int(test_eval["samples"]),
        "eligible_count": eligible_n,
        "eligible_attack_coverage": 1.0,
        "reference_final_result_path": str(_final_main_result_path(cfg, SEED, arch)),
        "reference_fusionbreak_counts_exact_match": reference_exact_match,
        "reference_fusionbreak_expected_counts": expected_counts,
        "reference_fusionbreak_rerun_counts": actual_counts,
        "rows": rows,
    }
    save_json(out, result)
    print(f"[saved] {out}")
    print(
        f"[reference PASS] {arch}: Full FusionBreak exactly reproduces final main counts | "
        f"ASR={100*new_full['fusion_ASR']:.2f}% FOSR={100*new_full['FOSR']:.2f}%"
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        total_time = float(sum(float(r["elapsed_seconds"]) for r in subset))
        a = {
            "variant": variant,
            "ablation_family": VARIANT_FAMILY[variant],
            "architectures": len(subset),
            "total_attacked_clean_samples": int(sum(int(r["attacked_clean_samples"]) for r in subset)),
            "mean_fusion_ASR": float(np.mean([float(r["fusion_ASR"]) for r in subset])),
            "mean_audio_preservation_rate": float(np.mean([float(r["audio_preservation_rate"]) for r in subset])),
            "mean_video_preservation_rate": float(np.mean([float(r["video_preservation_rate"]) for r in subset])),
            "mean_FOSR": float(np.mean([float(r["FOSR"]) for r in subset])),
            "total_elapsed_seconds": total_time,
            "total_elapsed_minutes": total_time / 60.0,
        }
        out.append(a)
    return out


def _paper_tables(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["variant"]: r for r in aggregate}

    def row(v: str, label: str) -> dict[str, Any]:
        r = by[v]
        return {
            "setting": label,
            "variant": v,
            "Fusion_ASR_percent": 100 * r["mean_fusion_ASR"],
            "Audio_Pres_percent": 100 * r["mean_audio_preservation_rate"],
            "Video_Pres_percent": 100 * r["mean_video_preservation_rate"],
            "FOSR_percent": 100 * r["mean_FOSR"],
            "time_minutes": r["total_elapsed_minutes"],
        }

    component = [
        row("fusion_ce_only", "Fusion CE only"),
        row("fusion_ce_plus_margin", "Fusion CE + margin"),
        row("without_preservation", "Without preservation"),
        row("without_fusion_margin", "Without fused margin"),
        row("without_distillation", "Without distillation"),
        row("final_iterate_selection", "Final-iterate selection"),
        row("full_fusionbreak", "Full FusionBreak"),
    ]

    full = by["full_fusionbreak"]
    def setting_row(v: str, label: str, eps_a: float, eps_v: float) -> dict[str, Any]:
        r = by[v]
        return {
            "setting": label,
            "variant": v,
            "eps_audio": eps_a,
            "eps_video": eps_v,
            "Fusion_ASR_percent": 100 * r["mean_fusion_ASR"],
            "FOSR_percent": 100 * r["mean_FOSR"],
            "time_minutes": r["total_elapsed_minutes"],
        }

    def default_row(label: str) -> dict[str, Any]:
        return {
            "setting": label,
            "variant": "full_fusionbreak",
            "eps_audio": EXPECTED_BASE_SETTINGS["eps_audio"],
            "eps_video": EXPECTED_BASE_SETTINGS["eps_video"],
            "Fusion_ASR_percent": 100 * full["mean_fusion_ASR"],
            "FOSR_percent": 100 * full["mean_FOSR"],
            "time_minutes": full["total_elapsed_minutes"],
        }

    attack_settings = [
        setting_row("budget_half", "Half budget", 0.04, 0.03),
        default_row("Default budget"),
        setting_row("budget_one_and_half", "1.5x budget", 0.12, 0.09),
        setting_row("one_restart", "One restart", 0.08, 0.06),
        setting_row("five_restarts", "Five restarts", 0.08, 0.06),
        default_row("Ten restarts"),
        setting_row("half_steps", "Half steps", 0.08, 0.06),
        default_row("Default steps"),
        setting_row("one_and_half_steps", "1.5x steps", 0.08, 0.06),
    ]
    return {"component_ablation": component, "attack_setting_ablation": attack_settings}


def _old_paper_delta(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["variant"]: r for r in aggregate}
    delta: dict[str, Any] = {"component": {}, "settings": {}}
    for v, old in OLD_PAPER_COMPONENT.items():
        if v not in by:
            continue
        new = by[v]
        delta["component"][v] = {
            k: {
                "old": float(old[k]),
                "new": float(new[f"mean_{k}"]),
                "delta_percentage_points": 100 * (float(new[f"mean_{k}"]) - float(old[k])),
            }
            for k in old
        }
    for v, old in OLD_PAPER_SETTINGS.items():
        if v not in by:
            continue
        new = by[v]
        delta["settings"][v] = {
            "fusion_ASR": {
                "old": float(old["fusion_ASR"]),
                "new": float(new["mean_fusion_ASR"]),
                "delta_percentage_points": 100 * (float(new["mean_fusion_ASR"]) - float(old["fusion_ASR"])),
            },
            "FOSR": {
                "old": float(old["FOSR"]),
                "new": float(new["mean_FOSR"]),
                "delta_percentage_points": 100 * (float(new["mean_FOSR"]) - float(old["FOSR"])),
            },
        }
    return delta


def write_summary(cfg: Config) -> None:
    results = []
    for arch in ARCHITECTURES:
        p = _result_path(cfg, arch)
        if p.exists():
            results.append(json.loads(p.read_text(encoding="utf-8")))

    rows = [row for result in results for row in result.get("rows", [])]
    if not rows:
        return

    summary_csv = cfg.log_dir / f"usenix_final_ablation_summary_v{ABLATION_VERSION}_ravdess.csv"
    aggregate_csv = cfg.log_dir / f"usenix_final_ablation_aggregate_v{ABLATION_VERSION}_ravdess.csv"
    tables_json = cfg.log_dir / f"usenix_final_ablation_paper_tables_v{ABLATION_VERSION}_ravdess.json"
    audit_json = cfg.log_dir / f"usenix_final_ablation_audit_v{ABLATION_VERSION}_ravdess.json"

    _write_csv(summary_csv, rows)
    aggregate = _aggregate_rows(rows)
    _write_csv(aggregate_csv, aggregate)
    tables = _paper_tables(aggregate)
    tables_json.write_text(json.dumps(tables, indent=2), encoding="utf-8")

    archs_present = sorted({r["fusion_architecture"] for r in rows})
    reference_pass = all(
        bool(r.get("reference_fusionbreak_counts_exact_match")) for r in results
    )
    config_hashes = sorted({str(r["config_sha256"]) for r in results})
    audit = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "expected_architectures": list(ARCHITECTURES),
        "architectures_present": archs_present,
        "complete": archs_present == sorted(ARCHITECTURES),
        "unique_config_sha256": config_hashes,
        "reference_fusionbreak_exact_match_all_completed_architectures": reference_pass,
        "all_variants_use_final_seed42_checkpoints_and_eligible_pool": True,
        "base_attack_settings": _config_snapshot(cfg),
        "paper_value_deltas": _old_paper_delta(aggregate),
        "important_note": (
            "Use the new paper_tables JSON after completion. The old manuscript ablation values "
            "should be replaced if any rerun value differs."
        ),
    }
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"[summary] {summary_csv}")
    print(f"[aggregate] {aggregate_csv}")
    print(f"[paper tables] {tables_json}")
    print(f"[audit] complete={audit['complete']} | reference_exact={reference_pass} | {audit_json}")


def main() -> None:
    if SEED not in PAPER_SEEDS:
        raise RuntimeError("Seed 42 must be part of the final paper seed set.")

    cfg = Config.create_default(Path(__file__).resolve().parent)
    _pin_paper_settings(cfg)
    cfg.ensure_dirs()
    _validate_pinned_config(cfg)
    device = _select_device()

    archs = _selected_architectures()
    print("\nFINAL-CONSISTENT RAVDESS ABLATION PROTOCOL")
    print("Seed:", SEED)
    print("Architectures:", tuple(archs))
    print("Base config SHA-256:", _config_fingerprint(_config_snapshot(cfg)))
    print("Base steps/restarts:", cfg.attack_steps, cfg.random_restarts)
    print("Base eps:", cfg.eps_audio, cfg.eps_video)
    print("Base alpha:", cfg.alpha_audio, cfg.alpha_video)
    print("Variants:", VARIANTS)

    for arch in tqdm(archs, desc="RAVDESS final ablation architectures", unit="arch"):
        _run_one_arch(arch, cfg, device)
        write_summary(cfg)

    write_summary(cfg)
    print("All requested final-consistent RAVDESS ablations completed.")


if __name__ == "__main__":
    main()
