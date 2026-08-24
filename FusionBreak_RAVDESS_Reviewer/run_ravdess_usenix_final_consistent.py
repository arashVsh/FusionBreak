from __future__ import annotations

"""
Final consistency-locked RAVDESS runner for the USENIX FusionBreak paper.

Paper protocol locked here
--------------------------
Seeds: 42, 43, 44
Fusion architectures: interaction, concat, gated, group_gated
Methods evaluated on the exact same eligible pool and checkpoints:
  1) FusionBreak
  2) AdaptiveFusionBreak
  3) PGD
  4) CEOnlySelection
  5) AudioOnlyFusionBreak
  6) VideoOnlyFusionBreak

The numerical attack budget is identical for all applicable methods:
  - 80 optimization steps
  - 10 random restarts
  - eps_audio = 0.08
  - eps_video = 0.06
  - alpha_audio = 0.004
  - alpha_video = 0.003

FusionBreak and AdaptiveFusionBreak use the same base objective, same candidate
selection, same restart initialization, and same number of optimization steps.
The ONLY objective difference is the detector-evasion term in AdaptiveFusionBreak.
There are no adaptive-only stealth/refinement steps.

The runner also evaluates:
  - eligibility/coverage,
  - fusion-consistency defense on clean validation/test and attacks,
  - generic detectors (feature squeezing, LID, Mahalanobis, MSP),
  - perturbation norm checks,
  - checkpoint SHA-256 hashes,
  - a configuration SHA-256 fingerprint.

It NEVER trains missing checkpoints. Missing checkpoints are an error.
Completed seed x architecture JSON files are skipped unless FUSIONGAP_FORCE=1.

Place this file in the RAVDESS project root containing:
    fusiongap_ravdess/
    RAVDESS/

Smoke test:
    $env:FUSIONGAP_SEEDS="42"
    $env:FUSIONGAP_ARCHS="concat"
    python .\run_ravdess_usenix_final_consistent.py

Final paper run:
    $env:FUSIONGAP_SEEDS="42,43,44"
    $env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
    python .\run_ravdess_usenix_final_consistent.py

Optional GPU selection:
    $env:FUSIONGAP_DEVICE="cuda:1"
"""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fusiongap_ravdess.attacks import (
    _candidate_score,
    _clamp_delta,
    _collect_clean_all_three,
    _counts,
    _rates,
)
from fusiongap_ravdess.config import Config
from fusiongap_ravdess.data import build_or_load_packed_cache, scan_records, split_records
from fusiongap_ravdess.detectors import collect_detector_scores, detection_metrics, fit_detector_pack
from fusiongap_ravdess.utils import save_json, set_seed
from fusiongap_ravdess.adaptive_detector_experiment import (
    _adaptive_detector_penalty,
    _checkpoint_paths,
    _ensure_adaptive_defaults,
    _kl_to_clean,
    _load_models,
    _threshold_stats,
    _true_minus_other_margin,
    _zero_model_grads,
)

DATASET_NAME = "RAVDESS"
RESULT_VERSION = 1
PAPER_SEEDS = (42, 43, 44)
ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")
ATTACK_NAMES = (
    "FusionBreak",
    "AdaptiveFusionBreak",
    "PGD",
    "CEOnlySelection",
    "AudioOnlyFusionBreak",
    "VideoOnlyFusionBreak",
)
GENERIC_DETECTORS = (
    "feature_squeezing_xu",
    "lid_ma",
    "mahalanobis_lee",
    "msp_hendrycks_gimpel",
)

EXPECTED_BASE_SETTINGS = {
    "attack_samples_cap": 500,
    "attack_steps": 80,
    "random_restarts": 10,
    "eps_audio": 0.08,
    "eps_video": 0.06,
    "alpha_audio": 0.004,
    "alpha_video": 0.003,
    "lambda_fusion_margin": 0.5,
    "lambda_preserve_margin": 5.0,
    "lambda_logit_distill": 0.25,
    "preserve_margin": 0.05,
    "candidate_eval_interval": 1,
    "clean_start_first_restart": True,
}

EXPECTED_ADAPTIVE_SETTINGS = {
    "lambda_detector_evasion": 2.0,
    "lambda_detector_msp": 0.5,
    "lambda_detector_mahalanobis": 3.0,
    "lambda_detector_feature_squeezing": 5.0,
    "lambda_detector_lid": 0.0,
    "adaptive_stealth_steps": 0,
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _select_device() -> torch.device:
    requested = os.environ.get("FUSIONGAP_DEVICE", "").strip()
    if not requested:
        if torch.cuda.is_available():
            requested = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        else:
            requested = "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {requested}") from exc
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {requested}, but only {torch.cuda.device_count()} CUDA GPU(s) are visible."
                )
        torch.cuda.set_device(torch.device(requested))

    device = torch.device(requested)
    print(f"Requested device: {requested}")
    if device.type == "cuda":
        print(f"Using GPU {device.index}: {torch.cuda.get_device_name(device)}")
    return device


def _pin_paper_settings(cfg: Config) -> None:
    # Base attack settings.
    cfg.attack_samples = 500
    cfg.attack_steps = 80
    cfg.random_restarts = 10
    cfg.eps_audio = 0.08
    cfg.eps_video = 0.06
    cfg.alpha_audio = 0.004
    cfg.alpha_video = 0.003
    cfg.lambda_fusion_margin = 0.5
    cfg.lambda_preserve_margin = 5.0
    cfg.lambda_logit_distill = 0.25
    cfg.preserve_margin = 0.05
    cfg.candidate_eval_interval = 1
    cfg.clean_start_first_restart = True

    # Detector defaults first, then explicitly override the adaptive settings
    # that previously caused inconsistent comparisons.
    _ensure_adaptive_defaults(cfg)
    cfg.detector_target_fpr = 0.05
    cfg.adaptive_detector_names = [
        "msp_hendrycks_gimpel",
        "mahalanobis_lee",
        "feature_squeezing_xu",
    ]
    cfg.lambda_detector_evasion = 2.0
    cfg.lambda_detector_msp = 0.5
    cfg.lambda_detector_mahalanobis = 3.0
    cfg.lambda_detector_feature_squeezing = 5.0
    cfg.lambda_detector_lid = 0.0

    # CRITICAL: no adaptive-only extra refinement.
    cfg.adaptive_stealth_steps = 0


def _config_snapshot(cfg: Config) -> dict[str, Any]:
    return {
        "paper_seeds": list(PAPER_SEEDS),
        "paper_architectures": list(ARCHITECTURES),
        "attack_samples_cap": int(cfg.attack_samples),
        "attack_steps": int(cfg.attack_steps),
        "random_restarts": int(cfg.random_restarts),
        "eps_audio": float(cfg.eps_audio),
        "eps_video": float(cfg.eps_video),
        "alpha_audio": float(cfg.alpha_audio),
        "alpha_video": float(cfg.alpha_video),
        "lambda_fusion_margin": float(cfg.lambda_fusion_margin),
        "lambda_preserve_margin": float(cfg.lambda_preserve_margin),
        "lambda_logit_distill": float(cfg.lambda_logit_distill),
        "preserve_margin": float(cfg.preserve_margin),
        "candidate_eval_interval": int(cfg.candidate_eval_interval),
        "clean_start_first_restart": bool(cfg.clean_start_first_restart),
        "detector_target_fpr": float(cfg.detector_target_fpr),
        "adaptive_detector_names": list(cfg.adaptive_detector_names),
        "lambda_detector_evasion": float(cfg.lambda_detector_evasion),
        "lambda_detector_msp": float(cfg.lambda_detector_msp),
        "lambda_detector_mahalanobis": float(cfg.lambda_detector_mahalanobis),
        "lambda_detector_feature_squeezing": float(cfg.lambda_detector_feature_squeezing),
        "lambda_detector_lid": float(cfg.lambda_detector_lid),
        "adaptive_stealth_steps": int(cfg.adaptive_stealth_steps),
        "projection": "L_inf ball around clean tensor for both RAVDESS modalities",
    }


def _config_fingerprint(snapshot: dict[str, Any]) -> str:
    raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(
            f"CONSISTENCY GUARD FAILED for {name}: actual={actual!r}, expected={expected!r}"
        )


def _validate_pinned_config(cfg: Config) -> None:
    snap = _config_snapshot(cfg)
    for key, expected in EXPECTED_BASE_SETTINGS.items():
        actual = snap[key]
        if isinstance(expected, bool):
            if bool(actual) != expected:
                raise RuntimeError(
                    f"CONSISTENCY GUARD FAILED for config.{key}: actual={actual!r}, expected={expected!r}"
                )
        else:
            _assert_close(f"config.{key}", actual, expected)
    for key, expected in EXPECTED_ADAPTIVE_SETTINGS.items():
        _assert_close(f"config.{key}", snap[key], expected)
    if snap["adaptive_detector_names"] != [
        "msp_hendrycks_gimpel",
        "mahalanobis_lee",
        "feature_squeezing_xu",
    ]:
        raise RuntimeError(
            "CONSISTENCY GUARD FAILED: unexpected adaptive_detector_names="
            f"{snap['adaptive_detector_names']}"
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_manifest(cfg: Config, seed: int, arch: str) -> dict[str, Any]:
    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    paths = {"audio": audio_path, "video": video_path, "fusion": fusion_path}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Final USENIX RAVDESS runner will NOT train missing checkpoints. Missing: "
            + "; ".join(missing)
        )
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _zero_counts() -> dict[str, int]:
    return {
        "n": 0,
        "fusion_fail": 0,
        "audio_preserved": 0,
        "video_preserved": 0,
        "fusion_only_success": 0,
    }


def _add_counts(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] += int(v)


def _zero_norms() -> dict[str, float]:
    return {
        "audio_linf_max": 0.0,
        "video_linf_max": 0.0,
        "audio_linf_sum": 0.0,
        "video_linf_sum": 0.0,
        "n": 0,
    }


@torch.no_grad()
def _add_norms(total: dict[str, float], xa_adv, xv_adv, xa0, xv0) -> None:
    an = (xa_adv - xa0).abs().flatten(1).max(1).values
    vn = (xv_adv - xv0).abs().flatten(1).max(1).values
    total["audio_linf_max"] = max(float(total["audio_linf_max"]), float(an.max().item()))
    total["video_linf_max"] = max(float(total["video_linf_max"]), float(vn.max().item()))
    total["audio_linf_sum"] += float(an.sum().item())
    total["video_linf_sum"] += float(vn.sum().item())
    total["n"] += int(an.numel())


def _finish_norms(total: dict[str, float]) -> dict[str, float]:
    n = max(1, int(total["n"]))
    return {
        "audio_linf_max": float(total["audio_linf_max"]),
        "video_linf_max": float(total["video_linf_max"]),
        "audio_linf_mean": float(total["audio_linf_sum"]) / n,
        "video_linf_mean": float(total["video_linf_sum"]) / n,
    }


def _zero_consistency() -> dict[str, int]:
    return {
        "n": 0,
        "flagged": 0,
        "unimodal_agree": 0,
        "fusion_only_n": 0,
        "fusion_only_flagged": 0,
    }


@torch.no_grad()
def _consistency_counts(audio, video, fusion, xa, xv, y) -> dict[str, int]:
    pa = audio(xa).argmax(1)
    pv = video(xv).argmax(1)
    pf = fusion(xa, xv).argmax(1)
    agree = pa == pv
    flagged = agree & (pf != pa)
    fusion_only = (pa == y) & (pv == y) & (pf != y)
    return {
        "n": int(y.numel()),
        "flagged": int(flagged.sum().item()),
        "unimodal_agree": int(agree.sum().item()),
        "fusion_only_n": int(fusion_only.sum().item()),
        "fusion_only_flagged": int((flagged & fusion_only).sum().item()),
    }


def _add_consistency(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] += int(v)


def _consistency_rates(c: dict[str, int]) -> dict[str, float]:
    return {
        "flag_rate": c["flagged"] / max(1, c["n"]),
        "unimodal_agreement_rate": c["unimodal_agree"] / max(1, c["n"]),
        "flag_rate_given_unimodal_agreement": c["flagged"] / max(1, c["unimodal_agree"]),
        "fusion_only_detection_rate": c["fusion_only_flagged"] / max(1, c["fusion_only_n"]),
    }


@torch.no_grad()
def _cache_eval(audio, video, fusion, cache, cfg: Config, device: torch.device) -> dict[str, Any]:
    ds = TensorDataset(cache["audio"].float(), cache["video"].float(), cache["labels"].long())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    consistency = _zero_consistency()
    n = 0
    audio_correct = 0
    video_correct = 0
    fusion_correct = 0
    all_three = 0

    for xa_cpu, xv_cpu, y_cpu in loader:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)
        pa = audio(xa).argmax(1)
        pv = video(xv).argmax(1)
        pf = fusion(xa, xv).argmax(1)
        n += int(y.numel())
        audio_correct += int((pa == y).sum().item())
        video_correct += int((pv == y).sum().item())
        fusion_correct += int((pf == y).sum().item())
        all_three += int(((pa == y) & (pv == y) & (pf == y)).sum().item())
        _add_consistency(consistency, _consistency_counts(audio, video, fusion, xa, xv, y))

    return {
        "samples": n,
        "audio_acc": audio_correct / max(1, n),
        "video_acc": video_correct / max(1, n),
        "fusion_acc": fusion_correct / max(1, n),
        "all_three_correct_count": all_three,
        "all_three_correct_rate": all_three / max(1, n),
        "consistency": {
            "counts": consistency,
            "rates": _consistency_rates(consistency),
        },
    }


def _fusionbreak_attack(
    audio,
    video,
    fusion,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    *,
    detector_pack=None,
    detector_stats=None,
    adaptive: bool = False,
):
    """Common FusionBreak implementation for the controlled comparison.

    FusionBreak and AdaptiveFusionBreak differ ONLY by the detector-evasion
    penalty in the objective below. Candidate selection and all numerical
    optimization settings are identical.
    """
    xa0, xv0 = xa.detach(), xv.detach()

    with torch.no_grad():
        clean_audio_logits = audio(xa0).detach()
        clean_video_logits = video(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a: torch.Tensor, v: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
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

        maybe_update(xa_adv, xv_adv)

        for step in range(1, int(cfg.attack_steps) + 1):
            _zero_model_grads(audio, video, fusion)
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)

            fusion_logits = fusion(xa_adv, xv_adv)
            audio_logits = audio(xa_adv)
            video_logits = video(xv_adv)

            fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
            fusion_attack_margin = -_true_minus_other_margin(fusion_logits, y)
            audio_margin = _true_minus_other_margin(audio_logits, y)
            video_margin = _true_minus_other_margin(video_logits, y)
            preservation_hinge = (
                F.relu(cfg.preserve_margin - audio_margin)
                + F.relu(cfg.preserve_margin - video_margin)
            )
            logit_distill = (
                _kl_to_clean(audio_logits, clean_audio_logits)
                + _kl_to_clean(video_logits, clean_video_logits)
            )

            objective_vec = (
                fusion_ce
                + cfg.lambda_fusion_margin * fusion_attack_margin
                - cfg.lambda_preserve_margin * preservation_hinge
                - cfg.lambda_logit_distill * logit_distill
            )

            if adaptive:
                if detector_pack is None or detector_stats is None:
                    raise ValueError("AdaptiveFusionBreak requires detector_pack and detector_stats.")
                det_penalty = _adaptive_detector_penalty(
                    fusion, detector_pack, xa_adv, xv_adv, cfg, detector_stats
                )
                objective_vec = objective_vec - cfg.lambda_detector_evasion * det_penalty

            objective = objective_vec.mean()
            ga, gv = torch.autograd.grad(objective, [xa_adv, xv_adv])

            with torch.no_grad():
                xa_adv = _clamp_delta(
                    xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                ).detach()
                xv_adv = _clamp_delta(
                    xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                ).detach()

            if (
                cfg.candidate_eval_interval <= 1
                or step % cfg.candidate_eval_interval == 0
                or step == cfg.attack_steps
            ):
                maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _standard_pgd(audio, video, fusion, xa, xv, y, cfg: Config):
    """Conventional fused-model PGD under the same numerical attack budget.

    Objective: fused cross-entropy only.
    Selection: highest FINAL fused CE over restarts.
    No FOSR-aware intermediate candidate selection.
    """
    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_loss = torch.full((xa0.shape[0],), -1e9, device=xa0.device)

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

        for _ in range(int(cfg.attack_steps)):
            _zero_model_grads(audio, video, fusion)
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)
            loss = F.cross_entropy(fusion(xa_adv, xv_adv), y)
            ga, gv = torch.autograd.grad(loss, [xa_adv, xv_adv])
            with torch.no_grad():
                xa_adv = _clamp_delta(
                    xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                ).detach()
                xv_adv = _clamp_delta(
                    xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                ).detach()

        with torch.no_grad():
            final_loss = F.cross_entropy(fusion(xa_adv, xv_adv), y, reduction="none")
            improve = final_loss > best_loss
            if improve.any():
                best_loss[improve] = final_loss[improve]
                best_xa[improve] = xa_adv[improve].detach()
                best_xv[improve] = xv_adv[improve].detach()

    return best_xa.detach(), best_xv.detach()


def _ce_only_selection_attack(audio, video, fusion, xa, xv, y, cfg: Config):
    """Fused CE only, but with the SAME decision-aware intermediate selection.

    This isolates whether FusionBreak's preservation objective adds value beyond
    the stronger search/selection procedure.
    """
    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a, v):
        nonlocal best_xa, best_xv, best_score
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
        maybe_update(xa_adv, xv_adv)

        for step in range(1, int(cfg.attack_steps) + 1):
            _zero_model_grads(audio, video, fusion)
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)
            loss = F.cross_entropy(fusion(xa_adv, xv_adv), y)
            ga, gv = torch.autograd.grad(loss, [xa_adv, xv_adv])
            with torch.no_grad():
                xa_adv = _clamp_delta(
                    xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                ).detach()
                xv_adv = _clamp_delta(
                    xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                ).detach()
            if (
                cfg.candidate_eval_interval <= 1
                or step % cfg.candidate_eval_interval == 0
                or step == cfg.attack_steps
            ):
                maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _single_modality_fusionbreak(audio, video, fusion, xa, xv, y, cfg: Config, mode: str):
    if mode not in {"audio", "video"}:
        raise ValueError(mode)

    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        clean_audio_logits = audio(xa0).detach()
        clean_video_logits = video(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a, v):
        nonlocal best_xa, best_xv, best_score
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
        elif mode == "audio":
            xa_adv = _clamp_delta(
                xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio),
                xa0,
                cfg.eps_audio,
            ).detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = xa0.clone().detach()
            xv_adv = _clamp_delta(
                xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video),
                xv0,
                cfg.eps_video,
            ).detach()

        maybe_update(xa_adv, xv_adv)

        for step in range(1, int(cfg.attack_steps) + 1):
            _zero_model_grads(audio, video, fusion)
            if mode == "audio":
                xa_adv.requires_grad_(True)
            else:
                xv_adv.requires_grad_(True)

            fusion_logits = fusion(xa_adv, xv_adv)
            audio_logits = audio(xa_adv)
            video_logits = video(xv_adv)
            fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
            fusion_attack_margin = -_true_minus_other_margin(fusion_logits, y)
            audio_margin = _true_minus_other_margin(audio_logits, y)
            video_margin = _true_minus_other_margin(video_logits, y)
            preservation_hinge = (
                F.relu(cfg.preserve_margin - audio_margin)
                + F.relu(cfg.preserve_margin - video_margin)
            )
            logit_distill = (
                _kl_to_clean(audio_logits, clean_audio_logits)
                + _kl_to_clean(video_logits, clean_video_logits)
            )
            objective = (
                fusion_ce
                + cfg.lambda_fusion_margin * fusion_attack_margin
                - cfg.lambda_preserve_margin * preservation_hinge
                - cfg.lambda_logit_distill * logit_distill
            ).mean()

            if mode == "audio":
                (g,) = torch.autograd.grad(objective, [xa_adv])
                with torch.no_grad():
                    xa_adv = _clamp_delta(
                        xa_adv + cfg.alpha_audio * g.sign(), xa0, cfg.eps_audio
                    ).detach()
                    xv_adv = xv0.clone().detach()
            else:
                (g,) = torch.autograd.grad(objective, [xv_adv])
                with torch.no_grad():
                    xv_adv = _clamp_delta(
                        xv_adv + cfg.alpha_video * g.sign(), xv0, cfg.eps_video
                    ).detach()
                    xa_adv = xa0.clone().detach()

            if (
                cfg.candidate_eval_interval <= 1
                or step % cfg.candidate_eval_interval == 0
                or step == cfg.attack_steps
            ):
                maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _concat_score_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _run_all_matched_attacks(
    audio,
    video,
    fusion,
    clean_pool,
    detector_pack,
    detector_stats,
    cfg: Config,
    device: torch.device,
):
    xa_all, xv_all, y_all = clean_pool
    loader = DataLoader(
        TensorDataset(xa_all.float(), xv_all.float(), y_all.long()),
        batch_size=cfg.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    counts = {name: _zero_counts() for name in ATTACK_NAMES}
    norms = {name: _zero_norms() for name in ATTACK_NAMES}
    consistency = {name: _zero_consistency() for name in ATTACK_NAMES}
    detector_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in ATTACK_NAMES
    }

    bar = tqdm(loader, desc="USENIX final matched RAVDESS", unit="batch")
    for xa_cpu, xv_cpu, y_cpu in bar:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)

        # All six methods begin from the exact same CPU/CUDA RNG state for this batch.
        matched_state = _capture_rng(device)

        xa_full, xv_full = _fusionbreak_attack(
            audio, video, fusion, xa, xv, y, cfg, adaptive=False
        )

        _restore_rng(matched_state, device)
        xa_adapt, xv_adapt = _fusionbreak_attack(
            audio,
            video,
            fusion,
            xa,
            xv,
            y,
            cfg,
            detector_pack=detector_pack,
            detector_stats=detector_stats,
            adaptive=True,
        )

        _restore_rng(matched_state, device)
        xa_pgd, xv_pgd = _standard_pgd(audio, video, fusion, xa, xv, y, cfg)

        _restore_rng(matched_state, device)
        xa_ce, xv_ce = _ce_only_selection_attack(audio, video, fusion, xa, xv, y, cfg)

        _restore_rng(matched_state, device)
        xa_audio, xv_audio = _single_modality_fusionbreak(
            audio, video, fusion, xa, xv, y, cfg, "audio"
        )

        _restore_rng(matched_state, device)
        xa_video, xv_video = _single_modality_fusionbreak(
            audio, video, fusion, xa, xv, y, cfg, "video"
        )

        outputs = {
            "FusionBreak": (xa_full, xv_full),
            "AdaptiveFusionBreak": (xa_adapt, xv_adapt),
            "PGD": (xa_pgd, xv_pgd),
            "CEOnlySelection": (xa_ce, xv_ce),
            "AudioOnlyFusionBreak": (xa_audio, xv_audio),
            "VideoOnlyFusionBreak": (xa_video, xv_video),
        }

        for name, (xa_adv, xv_adv) in outputs.items():
            _add_counts(counts[name], _counts(audio, video, fusion, xa_adv, xv_adv, y))
            _add_norms(norms[name], xa_adv, xv_adv, xa, xv)
            _add_consistency(
                consistency[name],
                _consistency_counts(audio, video, fusion, xa_adv, xv_adv, y),
            )

            # RAVDESS is small, so evaluate the same detector suite on all six
            # methods. This makes the final dataset self-contained.
            scores = collect_detector_scores(
                fusion,
                detector_pack,
                xa_adv.detach().cpu(),
                xv_adv.detach().cpu(),
                cfg,
                device,
            )
            for det_name, arr in scores.items():
                detector_parts[name].setdefault(det_name, []).append(arr)

        bar.set_postfix(
            FB=f"{100*_rates(counts['FusionBreak'])['FOSR']:.1f}%",
            Adapt=f"{100*_rates(counts['AdaptiveFusionBreak'])['FOSR']:.1f}%",
            CE=f"{100*_rates(counts['CEOnlySelection'])['FOSR']:.1f}%",
            PGD=f"{100*_rates(counts['PGD'])['FOSR']:.1f}%",
        )

    return {
        name: {
            "counts": counts[name],
            "rates": _rates(counts[name]),
            "perturbation_norms": _finish_norms(norms[name]),
            "consistency_detector": {
                "counts": consistency[name],
                "rates": _consistency_rates(consistency[name]),
            },
            "detector_scores": _concat_score_parts(detector_parts[name]),
        }
        for name in ATTACK_NAMES
    }


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"usenix_final_allmethods_v{RESULT_VERSION}_ravdess_seed{seed}_{arch}.json"


def _mean_generic_detection(result: dict[str, Any], attack_name: str) -> float | None:
    dets = result.get("detector_metrics", {}).get(attack_name, {})
    vals = []
    for det in dets.values():
        if "attack_detection_rate_at_threshold" in det:
            vals.append(float(det["attack_detection_rate_at_threshold"]))
    return float(np.mean(vals)) if vals else None


def run_one(seed: int, arch: str, cfg: Config, device: torch.device) -> dict[str, Any]:
    if seed not in PAPER_SEEDS:
        raise ValueError(f"Seed {seed} is not in locked paper seeds {PAPER_SEEDS}.")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Architecture {arch!r} is not in {ARCHITECTURES}.")

    out = _result_path(cfg, seed, arch)
    if out.exists() and not _env_flag("FUSIONGAP_FORCE", False):
        print(f"[skip] {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"USENIX FINAL CONSISTENT RAVDESS: seed={seed}, arch={arch}")
    print("=" * 100)

    set_seed(seed)
    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", seed, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", seed, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", seed, cfg, device)

    # Never train here. Lock to the already trained model files.
    checkpoint_manifest = _checkpoint_manifest(cfg, seed, arch)

    set_seed(seed)
    audio, video, fusion = _load_models(seed, arch, cfg, device)
    audio.eval()
    video.eval()
    fusion.eval()

    val_eval = _cache_eval(audio, video, fusion, val_cache, cfg, device)
    test_eval = _cache_eval(audio, video, fusion, test_cache, cfg, device)

    clean_pool = _collect_clean_all_three(audio, video, fusion, test_cache, cfg, device)
    if clean_pool is None:
        raise RuntimeError("No all-three-correct clean samples.")

    attacked_n = int(clean_pool[2].numel())
    eligible_n = int(test_eval["all_three_correct_count"])
    test_n = int(test_eval["samples"])
    coverage = attacked_n / max(1, eligible_n)

    # RAVDESS has a small test set, so the 500-sample cap should not truncate
    # the eligible pool. Fail rather than silently report a subset.
    if attacked_n != eligible_n:
        raise RuntimeError(
            "CONSISTENCY GUARD FAILED: the RAVDESS eligible pool was truncated. "
            f"eligible={eligible_n}, attacked={attacked_n}. Increase attack_samples or investigate."
        )

    print(
        "[consistency guard] PASS | "
        f"test={test_n} | eligible={eligible_n} | attacked={attacked_n} | coverage={coverage:.3f}"
    )

    detector_pack = fit_detector_pack(fusion, train_cache, cfg, device)
    val_clean_scores = collect_detector_scores(
        fusion,
        detector_pack,
        val_cache["audio"].float(),
        val_cache["video"].float(),
        cfg,
        device,
    )
    detector_stats = _threshold_stats(
        val_clean_scores, float(cfg.detector_target_fpr), device
    )

    attack_out = _run_all_matched_attacks(
        audio,
        video,
        fusion,
        clean_pool,
        detector_pack,
        detector_stats,
        cfg,
        device,
    )

    xa_clean, xv_clean, _ = clean_pool
    clean_pool_scores = collect_detector_scores(
        fusion, detector_pack, xa_clean, xv_clean, cfg, device
    )

    detector_metrics: dict[str, dict[str, Any]] = {name: {} for name in ATTACK_NAMES}
    for attack_name in ATTACK_NAMES:
        for detector_name in sorted(val_clean_scores.keys()):
            detector_metrics[attack_name][detector_name] = detection_metrics(
                clean_pool_scores[detector_name],
                attack_out[attack_name]["detector_scores"][detector_name],
                val_clean_scores[detector_name],
                float(cfg.detector_target_fpr),
            )
        attack_out[attack_name].pop("detector_scores", None)
        attack_out[attack_name]["test_set_fusion_only_yield"] = (
            attack_out[attack_name]["counts"]["fusion_only_success"] / max(1, test_n)
        )

    snapshot = _config_snapshot(cfg)
    config_sha = _config_fingerprint(snapshot)

    result = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "seed": int(seed),
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_manifest": checkpoint_manifest,
        "clean_validation": val_eval,
        "clean_test": test_eval,
        "eligibility": {
            "test_samples": test_n,
            "all_three_correct_count": eligible_n,
            "all_three_correct_rate": eligible_n / max(1, test_n),
            "attacked_eligible_samples": attacked_n,
            "eligible_attack_coverage": coverage,
        },
        "fusion_consistency_defense": {
            "definition": "flag if audio and video predictions agree but fusion predicts another class",
            "clean_validation": val_eval["consistency"],
            "clean_test": test_eval["consistency"],
        },
        "comparison_design": {
            "paper_seeds": list(PAPER_SEEDS),
            "paper_architectures": list(ARCHITECTURES),
            "same_eligible_samples_for_all_six_methods": True,
            "same_checkpoints_for_all_six_methods": True,
            "same_numerical_attack_budget_where_applicable": True,
            "matched_random_restart_rng_state_within_each_batch": True,
            "fusionbreak_vs_adaptive_same_candidate_selection": True,
            "fusionbreak_vs_adaptive_only_objective_difference": "AdaptiveFusionBreak adds detector-evasion penalty",
            "adaptive_extra_stealth_steps": 0,
            "CEOnlySelection": "fused CE objective with the same decision-aware intermediate candidate selection",
            "PGD": "fused CE objective with final-iterate per-restart selection, no FOSR-aware selection",
            "single_modality_controls": "one modality perturbed and the other kept exactly clean",
        },
        "config_snapshot": snapshot,
        "config_sha256": config_sha,
        "attack_metrics": attack_out,
        "detector_metrics": detector_metrics,
    }

    # Final hard budget assertions before saving.
    tol = 1e-5
    for name, obj in attack_out.items():
        norms = obj["perturbation_norms"]
        if norms["audio_linf_max"] > cfg.eps_audio + tol:
            raise RuntimeError(
                f"BUDGET VIOLATION {name}: audio Linf={norms['audio_linf_max']} > {cfg.eps_audio}"
            )
        if norms["video_linf_max"] > cfg.eps_video + tol:
            raise RuntimeError(
                f"BUDGET VIOLATION {name}: video Linf={norms['video_linf_max']} > {cfg.eps_video}"
            )
    if attack_out["AudioOnlyFusionBreak"]["perturbation_norms"]["video_linf_max"] > tol:
        raise RuntimeError("AudioOnlyFusionBreak changed the video modality.")
    if attack_out["VideoOnlyFusionBreak"]["perturbation_norms"]["audio_linf_max"] > tol:
        raise RuntimeError("VideoOnlyFusionBreak changed the audio modality.")

    save_json(out, result)
    print(f"[saved] {out}")
    for name in ATTACK_NAMES:
        rates = attack_out[name]["rates"]
        mean_det = _mean_generic_detection(result, name)
        det_text = "n/a" if mean_det is None else f"{100*mean_det:.2f}%"
        print(
            f"{name}: ASR={100*rates['fusion_ASR']:.2f}% | "
            f"Audio={100*rates['audio_preservation_rate']:.2f}% | "
            f"Video={100*rates['video_preservation_rate']:.2f}% | "
            f"FOSR={100*rates['FOSR']:.2f}% | generic detection={det_text}"
        )
    return result


def _flatten_row(result: dict[str, Any], attack_name: str) -> dict[str, Any]:
    obj = result["attack_metrics"][attack_name]
    rates = obj["rates"]
    cons = obj["consistency_detector"]["rates"]
    norms = obj["perturbation_norms"]
    elig = result["eligibility"]
    row = {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "fusion_architecture": result["fusion_architecture"],
        "attack": attack_name,
        "test_samples": elig["test_samples"],
        "eligible_count": elig["all_three_correct_count"],
        "eligible_rate": elig["all_three_correct_rate"],
        "attacked_eligible_samples": elig["attacked_eligible_samples"],
        "eligible_attack_coverage": elig["eligible_attack_coverage"],
        "fusion_ASR": rates["fusion_ASR"],
        "audio_preservation_rate": rates["audio_preservation_rate"],
        "video_preservation_rate": rates["video_preservation_rate"],
        "FOSR": rates["FOSR"],
        "clean_val_consistency_FPR": result["fusion_consistency_defense"]["clean_validation"]["rates"]["flag_rate"],
        "clean_test_consistency_FPR": result["fusion_consistency_defense"]["clean_test"]["rates"]["flag_rate"],
        "attack_consistency_detection_rate": cons["flag_rate"],
        "fusion_only_consistency_detection_rate": cons["fusion_only_detection_rate"],
        "test_set_fusion_only_yield": obj["test_set_fusion_only_yield"],
        "audio_linf_max": norms["audio_linf_max"],
        "video_linf_max": norms["video_linf_max"],
        "mean_generic_detection_rate": _mean_generic_detection(result, attack_name),
        "config_sha256": result["config_sha256"],
    }
    for det_name, det in result.get("detector_metrics", {}).get(attack_name, {}).items():
        if "attack_detection_rate_at_threshold" in det:
            row[f"det_{det_name}"] = float(det["attack_detection_rate_at_threshold"])
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _aggregate_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = [
        "fusion_ASR",
        "audio_preservation_rate",
        "video_preservation_rate",
        "FOSR",
        "mean_generic_detection_rate",
        "clean_test_consistency_FPR",
        "attack_consistency_detection_rate",
        "fusion_only_consistency_detection_rate",
        "eligible_rate",
        "test_set_fusion_only_yield",
    ] + [f"det_{d}" for d in GENERIC_DETECTORS]

    # Per architecture mean +/- SD over the three paper seeds.
    per_arch: list[dict[str, Any]] = []
    for attack in ATTACK_NAMES:
        for arch in ARCHITECTURES:
            subset = [r for r in rows if r["attack"] == attack and r["fusion_architecture"] == arch]
            if not subset:
                continue
            out: dict[str, Any] = {
                "attack": attack,
                "fusion_architecture": arch,
                "n_configs": len(subset),
                "seeds": ",".join(str(r["seed"]) for r in sorted(subset, key=lambda x: x["seed"])),
            }
            for k in metric_keys:
                vals = [float(r[k]) for r in subset if r.get(k) is not None and r.get(k) != ""]
                if vals:
                    m, s = _mean_sd(vals)
                    out[f"{k}_mean"] = m
                    out[f"{k}_sd"] = s
            per_arch.append(out)

    # Overall mean +/- SD over 12 seed x architecture configurations.
    overall: list[dict[str, Any]] = []
    for attack in ATTACK_NAMES:
        subset = [r for r in rows if r["attack"] == attack]
        if not subset:
            continue
        out = {"attack": attack, "n_configs": len(subset)}
        for k in metric_keys:
            vals = [float(r[k]) for r in subset if r.get(k) is not None and r.get(k) != ""]
            if vals:
                m, s = _mean_sd(vals)
                out[f"{k}_mean"] = m
                out[f"{k}_sd"] = s
        overall.append(out)

    # Sample-weighted aggregate over all eligible attacked examples.
    weighted: list[dict[str, Any]] = []
    for attack in ATTACK_NAMES:
        subset = [r for r in rows if r["attack"] == attack]
        if not subset:
            continue
        weights = np.asarray([float(r["attacked_eligible_samples"]) for r in subset], dtype=np.float64)
        denom = max(1.0, float(weights.sum()))
        out = {
            "attack": attack,
            "eligible_samples_total": int(weights.sum()),
            "n_configs": len(subset),
        }
        for k in [
            "fusion_ASR",
            "audio_preservation_rate",
            "video_preservation_rate",
            "FOSR",
            "mean_generic_detection_rate",
            "attack_consistency_detection_rate",
        ] + [f"det_{d}" for d in GENERIC_DETECTORS]:
            vals = []
            ws = []
            for r, w in zip(subset, weights):
                if r.get(k) is not None and r.get(k) != "":
                    vals.append(float(r[k]))
                    ws.append(float(w))
            if vals:
                out[k] = float(np.sum(np.asarray(vals) * np.asarray(ws)) / max(1.0, np.sum(ws)))
        weighted.append(out)

    return per_arch, overall, weighted


def write_summaries_and_audit(cfg: Config) -> None:
    results: list[dict[str, Any]] = []
    for p in sorted(cfg.log_dir.glob(f"usenix_final_allmethods_v{RESULT_VERSION}_ravdess_seed*_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[summary warning] {p}: {exc}")
            continue
        if r.get("seed") not in PAPER_SEEDS or r.get("fusion_architecture") not in ARCHITECTURES:
            continue
        if "attack_metrics" not in r:
            continue
        results.append(r)

    if not results:
        return

    rows: list[dict[str, Any]] = []
    for r in results:
        for name in ATTACK_NAMES:
            if name not in r["attack_metrics"]:
                raise RuntimeError(
                    f"AUDIT FAILED: {r['seed']} {r['fusion_architecture']} missing {name}."
                )
            rows.append(_flatten_row(r, name))

    # Cross-run consistency audit.
    config_hashes = sorted({str(r["config_sha256"]) for r in results})
    if len(config_hashes) != 1:
        raise RuntimeError(f"AUDIT FAILED: multiple config hashes found: {config_hashes}")

    config_keys = {(int(r["seed"]), str(r["fusion_architecture"])) for r in results}
    if len(config_keys) != len(results):
        raise RuntimeError("AUDIT FAILED: duplicate seed x architecture result files detected.")

    # Audio/video checkpoints must be identical across architectures within a seed.
    checkpoint_audit: dict[str, Any] = {}
    for seed in PAPER_SEEDS:
        seed_results = [r for r in results if int(r["seed"]) == seed]
        if not seed_results:
            continue
        audio_hashes = {r["checkpoint_manifest"]["audio"]["sha256"] for r in seed_results}
        video_hashes = {r["checkpoint_manifest"]["video"]["sha256"] for r in seed_results}
        if len(audio_hashes) != 1 or len(video_hashes) != 1:
            raise RuntimeError(
                f"AUDIT FAILED: seed {seed} uses inconsistent unimodal checkpoints across architectures."
            )
        checkpoint_audit[str(seed)] = {
            "audio_sha256": next(iter(audio_hashes)),
            "video_sha256": next(iter(video_hashes)),
            "architectures_present": sorted(r["fusion_architecture"] for r in seed_results),
        }

    # Validate per-row budgets and sample counts.
    for row in rows:
        if int(row["attacked_eligible_samples"]) != int(row["eligible_count"]):
            raise RuntimeError(
                f"AUDIT FAILED: incomplete eligible coverage in {row['seed']} {row['fusion_architecture']} {row['attack']}"
            )
        if float(row["audio_linf_max"]) > EXPECTED_BASE_SETTINGS["eps_audio"] + 1e-5:
            raise RuntimeError("AUDIT FAILED: audio perturbation exceeds paper budget.")
        if float(row["video_linf_max"]) > EXPECTED_BASE_SETTINGS["eps_video"] + 1e-5:
            raise RuntimeError("AUDIT FAILED: video perturbation exceeds paper budget.")

    summary_csv = cfg.log_dir / f"usenix_final_summary_v{RESULT_VERSION}_ravdess.csv"
    summary_json = cfg.log_dir / f"usenix_final_summary_v{RESULT_VERSION}_ravdess.json"
    _write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    per_arch, overall, weighted = _aggregate_tables(rows)
    _write_csv(cfg.log_dir / f"usenix_final_per_arch_mean_sd_v{RESULT_VERSION}_ravdess.csv", per_arch)
    _write_csv(cfg.log_dir / f"usenix_final_overall_12config_mean_sd_v{RESULT_VERSION}_ravdess.csv", overall)
    _write_csv(cfg.log_dir / f"usenix_final_weighted_aggregate_v{RESULT_VERSION}_ravdess.csv", weighted)

    expected_configs = {(s, a) for s in PAPER_SEEDS for a in ARCHITECTURES}
    missing_configs = sorted(expected_configs - config_keys)
    audit = {
        "dataset": DATASET_NAME,
        "paper_seeds": list(PAPER_SEEDS),
        "paper_architectures": list(ARCHITECTURES),
        "expected_seed_arch_configs": len(expected_configs),
        "completed_seed_arch_configs": len(config_keys),
        "complete": len(missing_configs) == 0,
        "missing_configs": [list(x) for x in missing_configs],
        "unique_config_sha256": config_hashes,
        "all_rows_have_full_eligible_coverage": True,
        "all_rows_within_linf_budgets": True,
        "unimodal_checkpoint_consistency_by_seed": checkpoint_audit,
        "methods_per_config": list(ATTACK_NAMES),
        "fusionbreak_adaptive_only_objective_difference": "detector-evasion penalty",
        "adaptive_extra_stealth_steps": 0,
    }
    audit_path = cfg.log_dir / f"usenix_final_consistency_audit_v{RESULT_VERSION}_ravdess.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"[summary] {summary_csv}")
    print(f"[summary] {summary_json}")
    print(f"[audit] completed {len(config_keys)}/12 seed x architecture configs")
    print(f"[audit] complete={audit['complete']} | {audit_path}")


def _parse_plan(cfg: Config) -> tuple[list[int], list[str]]:
    seed_env = os.environ.get("FUSIONGAP_SEEDS", "").strip()
    arch_env = os.environ.get("FUSIONGAP_ARCHS", "").strip()

    seeds = (
        [int(x.strip()) for x in seed_env.split(",") if x.strip()]
        if seed_env
        else list(PAPER_SEEDS)
    )
    archs = (
        [x.strip() for x in arch_env.split(",") if x.strip()]
        if arch_env
        else list(ARCHITECTURES)
    )

    invalid_seeds = [s for s in seeds if s not in PAPER_SEEDS]
    invalid_archs = [a for a in archs if a not in ARCHITECTURES]
    if invalid_seeds:
        raise ValueError(
            f"Locked USENIX RAVDESS runner accepts only paper seeds {PAPER_SEEDS}. Invalid: {invalid_seeds}"
        )
    if invalid_archs:
        raise ValueError(f"Invalid architecture(s): {invalid_archs}")
    return seeds, archs


def main() -> None:
    cfg = Config.create_default(Path(__file__).resolve().parent)
    _pin_paper_settings(cfg)
    cfg.ensure_dirs()
    _validate_pinned_config(cfg)
    device = _select_device()

    seeds, archs = _parse_plan(cfg)
    plan = [(s, a) for s in seeds for a in archs]

    print("\nLOCKED RAVDESS USENIX PROTOCOL")
    print("Paper seeds:", PAPER_SEEDS)
    print("Architectures:", ARCHITECTURES)
    print("Requested plan:", plan)
    print("Config SHA-256:", _config_fingerprint(_config_snapshot(cfg)))
    print("Adaptive extra stealth steps:", cfg.adaptive_stealth_steps)
    print("Adaptive LID objective weight:", cfg.lambda_detector_lid)

    for seed, arch in tqdm(plan, desc="RAVDESS final USENIX configs", unit="config"):
        run_one(seed, arch, cfg, device)
        write_summaries_and_audit(cfg)

    write_summaries_and_audit(cfg)
    print("All requested RAVDESS final-consistent experiments completed.")


if __name__ == "__main__":
    main()
