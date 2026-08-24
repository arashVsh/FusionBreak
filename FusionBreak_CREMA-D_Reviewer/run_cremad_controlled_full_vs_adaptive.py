from __future__ import annotations

"""
Controlled CREMA-D comparison:
    Full FusionBreak vs AdaptiveFusionBreak

Why this runner exists
----------------------
The older CREMA-D "FusionGap" baseline used a simpler CE-preservation
objective, while the adaptive detector-aware attack used the full
margin + KL objective. This runner removes that confound.

Both attacks below use exactly the same:
  - eligible clean samples
  - checkpoints
  - 80 attack steps
  - 10 random restarts
  - eps_audio = 0.08
  - eps_video = 0.06
  - alpha_audio = 0.004
  - alpha_video = 0.003
  - fused CE + fused margin
  - unimodal margin preservation
  - KL/logit consistency
  - decision-aware intermediate candidate selection
  - random-restart initializations

The ONLY optimization-objective difference is:
  AdaptiveFusionBreak adds the detector-evasion penalty.

Important projection fix
------------------------
CREMA-D audio tensors are log-magnitude spectrograms and are not
naturally restricted to [0, 1]. Therefore:
  - audio: project only into the L_inf ball around the clean tensor
  - video: project into the L_inf ball and clamp to [0, 1]

This runner also logs observed max L_inf perturbations so the threat
model can be verified from the output.

Run from the project root:
    python run_cremad_controlled_full_vs_adaptive.py

Optional subsets:
    PowerShell:
      $env:FUSIONGAP_SEEDS="42"
      $env:FUSIONGAP_ARCHS="concat"
      python run_cremad_controlled_full_vs_adaptive.py

    All final runs:
      $env:FUSIONGAP_SEEDS="42,43,44,123,2026"
      $env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
      python run_cremad_controlled_full_vs_adaptive.py

    Prepare corrected caches and train missing checkpoints without attacking:
      $env:FUSIONGAP_PREPARE_ONLY="1"
      python run_cremad_controlled_full_vs_adaptive.py

    Select a GPU explicitly on a multi-GPU machine:
      $env:FUSIONGAP_DEVICE="cuda:0"
"""

import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fusiongap_cremad.attacks import (
    _collect_clean_pool,
    _counts,
    _rates,
    _rnn_modules_train_mode,
    _zero_model_grads,
)
from fusiongap_cremad.config import Config
from fusiongap_cremad.data import (
    CACHE_FORMAT_VERSION,
    build_or_load_packed_cache,
    scan_records,
    split_records,
)
from fusiongap_cremad.detectors import (
    collect_detector_scores,
    detection_metrics,
    fit_detector_pack,
)
from fusiongap_cremad.models import AudioNet, FusionNet, VideoNet
from fusiongap_cremad.train import clean_evaluation, train_fusion, train_unimodal
from fusiongap_cremad.utils import get_device, save_json, set_seed

# Reuse model loading, full-objective helper functions, and detector penalty
# from the final adaptive CREMA-D implementation.
from fusiongap_cremad.adaptive_detector_experiment import (
    _adaptive_detector_penalty,
    _ensure_adaptive_defaults,
    _kl_to_clean,
    _load_checkpoint_state_dict,
    _threshold_stats,
    _true_minus_other_margin,
)

DATASET_NAME = "CREMA-D"
ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")
RESULT_VERSION = 5


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    return (
        cfg.checkpoint_dir / f"audio_cremad_v2_seed{seed}.pt",
        cfg.checkpoint_dir / f"video_cremad_v2_seed{seed}.pt",
        cfg.checkpoint_dir / f"fusion_cremad_v2_seed{seed}_{arch}.pt",
    )


def _load_controlled_models(
    seed: int,
    arch: str,
    cfg: Config,
    device: torch.device,
):
    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    missing = [
        str(path)
        for path in (audio_path, video_path, fusion_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing controlled checkpoint(s): " + "; ".join(missing)
        )

    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio.load_state_dict(_load_checkpoint_state_dict(audio_path, device))
    video.load_state_dict(_load_checkpoint_state_dict(video_path, device))

    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)
    fusion = FusionNet(
        fusion_audio,
        fusion_video,
        arch,
        num_classes=cfg.num_classes,
    ).to(device)
    fusion.load_state_dict(_load_checkpoint_state_dict(fusion_path, device))
    audio.eval()
    video.eval()
    fusion.eval()
    return audio, video, fusion


def _ensure_checkpoints(
    seed: int,
    arch: str,
    cfg: Config,
    device: torch.device,
    train_cache: dict[str, torch.Tensor],
    val_cache: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Train only the checkpoints missing from this seed/architecture."""
    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    missing_before = [
        str(path)
        for path in (audio_path, video_path, fusion_path)
        if not path.exists()
    ]
    if not missing_before:
        return {"trained_missing_checkpoints": False, "missing_before": []}

    if not _env_flag("FUSIONGAP_AUTO_TRAIN_MISSING", True):
        raise FileNotFoundError(
            "Missing checkpoint(s), and FUSIONGAP_AUTO_TRAIN_MISSING is disabled: "
            + "; ".join(missing_before)
        )

    print("[prepare] Training missing checkpoints:")
    for path in missing_before:
        print(f"  - {path}")

    # Component-specific seeds make interrupted/resumed preparation reproducible:
    # training a missing video/fusion model does not depend on whether audio was
    # trained earlier in the same process.
    set_seed(seed + 10_000)
    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    audio_metrics = train_unimodal(
        audio, train_cache, val_cache, "audio", cfg, device, audio_path
    )
    set_seed(seed + 20_000)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    video_metrics = train_unimodal(
        video, train_cache, val_cache, "video", cfg, device, video_path
    )

    set_seed(seed + 30_000 + ARCHITECTURES.index(arch))
    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)
    fusion = FusionNet(
        fusion_audio,
        fusion_video,
        arch,
        num_classes=cfg.num_classes,
    ).to(device)
    fusion_metrics = train_fusion(
        fusion, train_cache, val_cache, cfg, device, fusion_path
    )

    missing_after = [
        str(path)
        for path in (audio_path, video_path, fusion_path)
        if not path.exists()
    ]
    if missing_after:
        raise RuntimeError(
            "Checkpoint preparation finished without creating: "
            + "; ".join(missing_after)
        )

    return {
        "trained_missing_checkpoints": True,
        "missing_before": missing_before,
        "training_metrics": {
            "audio": audio_metrics,
            "video": video_metrics,
            "fusion": fusion_metrics,
        },
    }


def _project_audio(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Project standardized/log-spectrogram audio only into the L_inf ball."""
    return torch.maximum(torch.minimum(x_adv, x0 + eps), x0 - eps)


def _project_video(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Project video into the L_inf ball and its valid [0,1] input domain."""
    projected = torch.maximum(torch.minimum(x_adv, x0 + eps), x0 - eps)
    return projected.clamp(0.0, 1.0)


@torch.no_grad()
def _candidate_score(
    audio_model,
    video_model,
    fusion_model,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Decision-aware candidate ranking shared by BOTH attacks.

    Detector scores are intentionally NOT used here. This keeps candidate
    selection identical between Full FusionBreak and AdaptiveFusionBreak,
    so the controlled difference is the detector penalty in the objective.
    """
    audio_logits = audio_model(xa)
    video_logits = video_model(xv)
    fusion_logits = fusion_model(xa, xv)

    pa = audio_logits.argmax(dim=1)
    pv = video_logits.argmax(dim=1)
    pf = fusion_logits.argmax(dim=1)

    audio_ok = pa == y
    video_ok = pv == y
    fusion_fail = pf != y
    fusion_only = fusion_fail & audio_ok & video_ok
    one_unimodal_ok = fusion_fail & (audio_ok | video_ok)

    fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
    audio_margin = _true_minus_other_margin(audio_logits, y)
    video_margin = _true_minus_other_margin(video_logits, y)

    # Lexicographic-style ranking:
    # fusion-only > fusion fail with one modality preserved > generic fusion fail
    # > unimodal correctness > continuous tie breakers.
    return (
        fusion_only.float() * 1_000_000_000.0
        + one_unimodal_ok.float() * 10_000_000.0
        + fusion_fail.float() * 1_000_000.0
        + audio_ok.float() * 10_000.0
        + video_ok.float() * 10_000.0
        + fusion_ce
        + 0.01 * (audio_margin + video_margin)
    )


def _full_objective_attack(
    audio_model,
    video_model,
    fusion_model,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    detector_pack=None,
    detector_stats=None,
    adaptive: bool = False,
):
    """
    Full FusionBreak objective.

    If adaptive=False:
        J = CE_F + lambda_F * margin_F
            - lambda_M * preserve_hinge
            - lambda_KL * KL_unimodal

    If adaptive=True:
        same J - lambda_D * detector_penalty

    Everything else, including candidate selection, is identical.
    """
    xa0 = xa.detach()
    xv0 = xv.detach()

    with torch.no_grad():
        clean_audio_logits = audio_model(xa0).detach()
        clean_video_logits = video_model(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(
            audio_model, video_model, fusion_model, best_xa, best_xv, y
        )

    def maybe_update_best(xa_cand: torch.Tensor, xv_cand: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score(
                audio_model, video_model, fusion_model, xa_cand, xv_cand, y
            )
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = xa_cand[improve].detach()
                best_xv[improve] = xv_cand[improve].detach()

    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and bool(getattr(cfg, "clean_start_first_restart", True)):
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _project_audio(
                xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio),
                xa0,
                cfg.eps_audio,
            ).detach()
            xv_adv = _project_video(
                xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video),
                xv0,
                cfg.eps_video,
            ).detach()

        maybe_update_best(xa_adv, xv_adv)

        with _rnn_modules_train_mode(audio_model, video_model, fusion_model):
            for step in range(1, int(cfg.attack_steps) + 1):
                _zero_model_grads(audio_model, video_model, fusion_model)
                xa_adv.requires_grad_(True)
                xv_adv.requires_grad_(True)

                fusion_logits = fusion_model(xa_adv, xv_adv)
                audio_logits = audio_model(xa_adv)
                video_logits = video_model(xv_adv)

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
                        raise ValueError(
                            "Adaptive attack requires detector_pack and detector_stats."
                        )
                    detector_penalty = _adaptive_detector_penalty(
                        fusion_model,
                        detector_pack,
                        xa_adv,
                        xv_adv,
                        cfg,
                        detector_stats,
                    )
                    objective_vec = (
                        objective_vec
                        - cfg.lambda_detector_evasion * detector_penalty
                    )

                objective = objective_vec.mean()
                ga, gv = torch.autograd.grad(
                    objective,
                    [xa_adv, xv_adv],
                    retain_graph=False,
                    create_graph=False,
                )

                with torch.no_grad():
                    xa_adv = _project_audio(
                        xa_adv + cfg.alpha_audio * ga.sign(),
                        xa0,
                        cfg.eps_audio,
                    ).detach()
                    xv_adv = _project_video(
                        xv_adv + cfg.alpha_video * gv.sign(),
                        xv0,
                        cfg.eps_video,
                    ).detach()

                if (
                    cfg.candidate_eval_interval <= 1
                    or step % cfg.candidate_eval_interval == 0
                    or step == cfg.attack_steps
                ):
                    maybe_update_best(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


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
def _add_norms(
    total: dict[str, float],
    xa_adv: torch.Tensor,
    xv_adv: torch.Tensor,
    xa0: torch.Tensor,
    xv0: torch.Tensor,
) -> None:
    a = (xa_adv - xa0).abs().flatten(1).max(dim=1).values
    v = (xv_adv - xv0).abs().flatten(1).max(dim=1).values

    total["audio_linf_max"] = max(
        float(total["audio_linf_max"]), float(a.max().item())
    )
    total["video_linf_max"] = max(
        float(total["video_linf_max"]), float(v.max().item())
    )
    total["audio_linf_sum"] += float(a.sum().item())
    total["video_linf_sum"] += float(v.sum().item())
    total["n"] += int(a.numel())


def _finalize_norms(total: dict[str, float]) -> dict[str, float]:
    n = max(int(total["n"]), 1)
    return {
        "audio_linf_max": float(total["audio_linf_max"]),
        "video_linf_max": float(total["video_linf_max"]),
        "audio_linf_mean": float(total["audio_linf_sum"]) / n,
        "video_linf_mean": float(total["video_linf_sum"]) / n,
    }


def _concat_score_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _capture_attack_rng(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_attack_rng(state: dict[str, torch.Tensor], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _run_attacks_on_pool(
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
    ds = TensorDataset(xa_all.float(), xv_all.float(), y_all.long())
    loader = DataLoader(
        ds,
        batch_size=cfg.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    full_counts = _zero_counts()
    adaptive_counts = _zero_counts()
    full_norms = _zero_norms()
    adaptive_norms = _zero_norms()

    full_scores_parts: dict[str, list[np.ndarray]] = {}
    adaptive_scores_parts: dict[str, list[np.ndarray]] = {}

    bar = tqdm(loader, desc="Controlled Full vs Adaptive FusionBreak", unit="batch")

    for xa_cpu, xv_cpu, y_cpu in bar:
        xa = xa_cpu.to(device)
        xv = xv_cpu.to(device)
        y = y_cpu.to(device)

        # Reset the generator before the second attack so both methods receive
        # exactly the same clean/random restart initializations. Without this,
        # sequential execution adds random-start noise to the claimed controlled
        # objective comparison.
        matched_rng_state = _capture_attack_rng(device)

        # Full objective without detector penalty.
        xa_full, xv_full = _full_objective_attack(
            audio,
            video,
            fusion,
            xa,
            xv,
            y,
            cfg,
            adaptive=False,
        )

        # Same objective + detector-evasion term.
        _restore_attack_rng(matched_rng_state, device)
        xa_adapt, xv_adapt = _full_objective_attack(
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

        _add_counts(
            full_counts,
            _counts(audio, video, fusion, xa_full, xv_full, y),
        )
        _add_counts(
            adaptive_counts,
            _counts(audio, video, fusion, xa_adapt, xv_adapt, y),
        )

        _add_norms(full_norms, xa_full, xv_full, xa, xv)
        _add_norms(adaptive_norms, xa_adapt, xv_adapt, xa, xv)

        # Detector scores for the controlled comparison.
        for name, arr in collect_detector_scores(
            fusion,
            detector_pack,
            xa_full.detach().cpu(),
            xv_full.detach().cpu(),
            cfg,
            device,
        ).items():
            full_scores_parts.setdefault(name, []).append(arr)

        for name, arr in collect_detector_scores(
            fusion,
            detector_pack,
            xa_adapt.detach().cpu(),
            xv_adapt.detach().cpu(),
            cfg,
            device,
        ).items():
            adaptive_scores_parts.setdefault(name, []).append(arr)

        fr = _rates(full_counts)
        ar = _rates(adaptive_counts)
        bar.set_postfix(
            full_FOSR=f"{100*fr['FOSR']:.1f}%",
            adapt_FOSR=f"{100*ar['FOSR']:.1f}%",
        )

    return {
        "full_counts": full_counts,
        "adaptive_counts": adaptive_counts,
        "full_rates": _rates(full_counts),
        "adaptive_rates": _rates(adaptive_counts),
        "full_norms": _finalize_norms(full_norms),
        "adaptive_norms": _finalize_norms(adaptive_norms),
        "full_scores": _concat_score_parts(full_scores_parts),
        "adaptive_scores": _concat_score_parts(adaptive_scores_parts),
    }


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / (
        f"controlled_full_vs_adaptive_v{RESULT_VERSION}_cremad_"
        f"seed{seed}_cremad_{arch}.json"
    )


def _configure_controlled_attack(cfg: Config) -> None:
    _ensure_adaptive_defaults(cfg)

    # Explicitly pin all attack settings used in the paper.
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

    # Detector-aware objective settings from the final adaptive implementation.
    cfg.lambda_detector_evasion = 2.0
    cfg.lambda_detector_msp = 0.5
    cfg.lambda_detector_mahalanobis = 3.0
    cfg.lambda_detector_feature_squeezing = 5.0
    cfg.lambda_detector_lid = 0.0
    cfg.adaptive_detector_names = [
        "msp_hendrycks_gimpel",
        "mahalanobis_lee",
        "feature_squeezing_xu",
    ]

    # IMPORTANT:
    # Disable the extra adaptive-only stealth refinement because it would add
    # 30 extra optimization steps to only one side of the controlled comparison.
    cfg.adaptive_stealth_steps = 0

    # Never overwrite an existing controlled run unless explicitly requested.
    cfg.skip_completed_controlled_results = True


def run_one(
    seed: int,
    arch: str,
    cfg: Config,
    *,
    prepare_only: bool = False,
) -> dict[str, Any]:
    cfg.ensure_dirs()
    out_json = _result_path(cfg, seed, arch)

    if (
        not prepare_only
        and bool(getattr(cfg, "skip_completed_controlled_results", True))
        and out_json.exists()
    ):
        print(f"[skip] Controlled result exists: {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(
        f"CREMA-D CONTROLLED Full FusionBreak vs AdaptiveFusionBreak: "
        f"seed={seed}, arch={arch}"
    )
    print("=" * 100)

    set_seed(seed)
    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)

    train_cache = build_or_load_packed_cache(
        train_records, "train", seed, cfg, device
    )
    val_cache = build_or_load_packed_cache(
        val_records, "val", seed, cfg, device
    )
    test_cache = build_or_load_packed_cache(
        test_records, "test", seed, cfg, device
    )

    checkpoint_preparation = _ensure_checkpoints(
        seed,
        arch,
        cfg,
        device,
        train_cache,
        val_cache,
    )

    # Attack randomness must not depend on whether training happened during this
    # invocation or checkpoints were already present.
    set_seed(seed)
    audio, video, fusion = _load_controlled_models(
        seed, arch, cfg, device
    )

    clean_eval = clean_evaluation(
        audio, video, fusion, test_cache, cfg, device
    )
    print("Clean evaluation:", clean_eval)
    quality_gate_passed = (
        clean_eval["fusion_acc"] >= cfg.min_fusion_acc_for_attack
        and clean_eval["all_three_correct_rate"]
        >= cfg.min_all_three_rate_for_attack
    )

    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    preparation_result = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "checkpoints": {
            "audio": str(audio_path),
            "video": str(video_path),
            "fusion": str(fusion_path),
        },
        "checkpoint_preparation": checkpoint_preparation,
        "clean_eval": clean_eval,
        "quality_gate": {
            "passed": quality_gate_passed,
            "min_fusion_acc": cfg.min_fusion_acc_for_attack,
            "min_all_three_correct_rate": cfg.min_all_three_rate_for_attack,
        },
    }
    preparation_path = cfg.log_dir / (
        f"controlled_preparation_v{RESULT_VERSION}_cremad_"
        f"seed{seed}_cremad_{arch}.json"
    )
    save_json(preparation_path, preparation_result)
    print(f"[prepare] Saved: {preparation_path}")

    if not quality_gate_passed:
        raise RuntimeError(
            "Publication quality gate failed: "
            f"fusion_acc={clean_eval['fusion_acc']:.4f} "
            f"(minimum {cfg.min_fusion_acc_for_attack:.4f}), "
            "all_three_correct_rate="
            f"{clean_eval['all_three_correct_rate']:.4f} "
            f"(minimum {cfg.min_all_three_rate_for_attack:.4f})."
        )

    if prepare_only:
        print(f"[prepare] Ready for controlled attack: seed={seed}, arch={arch}")
        return preparation_result

    clean_pool = _collect_clean_pool(
        audio, video, fusion, test_cache, cfg, device
    )
    if clean_pool is None:
        result = {
            "dataset": DATASET_NAME,
            "seed": seed,
            "fusion_architecture": arch,
            "message": "No all-three-correct clean samples.",
        }
        save_json(out_json, result)
        return result

    xa_clean, xv_clean, y_clean = clean_pool
    print(
        f"[controlled] Eligible attacked clean samples: "
        f"{int(y_clean.numel())}"
    )

    # Fit/calibrate detectors exactly as in the adaptive evaluation.
    detector_pack = fit_detector_pack(fusion, train_cache, cfg, device)
    val_xa = val_cache["audio"].float()
    val_xv = val_cache["video"].float()

    val_clean_scores = collect_detector_scores(
        fusion, detector_pack, val_xa, val_xv, cfg, device
    )

    target_fpr = float(cfg.detector_target_fpr)
    detector_stats = _threshold_stats(
        val_clean_scores, target_fpr, device
    )

    attack_out = _run_attacks_on_pool(
        audio,
        video,
        fusion,
        clean_pool,
        detector_pack,
        detector_stats,
        cfg,
        device,
    )

    detector_metrics = {
        "FusionBreakFull": {},
        "AdaptiveFusionBreak": {},
    }
    clean_pool_scores = collect_detector_scores(
        fusion,
        detector_pack,
        xa_clean,
        xv_clean,
        cfg,
        device,
    )

    for detector_name in sorted(val_clean_scores.keys()):
        detector_metrics["FusionBreakFull"][detector_name] = detection_metrics(
            clean_pool_scores[detector_name],
            attack_out["full_scores"][detector_name],
            val_clean_scores[detector_name],
            target_fpr,
        )

        detector_metrics["AdaptiveFusionBreak"][detector_name] = detection_metrics(
            clean_pool_scores[detector_name],
            attack_out["adaptive_scores"][detector_name],
            val_clean_scores[detector_name],
            target_fpr,
        )

    result = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "attacked_clean_samples": int(y_clean.numel()),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "checkpoints": preparation_result["checkpoints"],
        "checkpoint_preparation": checkpoint_preparation,
        "clean_eval": clean_eval,
        "quality_gate": preparation_result["quality_gate"],
        "comparison_design": {
            "purpose": (
                "Controlled comparison of full FusionBreak and "
                "AdaptiveFusionBreak with identical base objective, "
                "candidate selection, budgets, steps, and restarts."
            ),
            "only_objective_difference": (
                "AdaptiveFusionBreak adds the detector-evasion penalty."
            ),
            "audio_projection": (
                "L_inf projection around clean audio tensor; no [0,1] clamp."
            ),
            "video_projection": (
                "L_inf projection around clean video tensor plus [0,1] clamp."
            ),
            "adaptive_extra_stealth_steps": 0,
            "matched_random_restart_initializations": True,
        },
        "attack_settings": {
            "attack_steps": cfg.attack_steps,
            "random_restarts": cfg.random_restarts,
            "attack_samples": cfg.attack_samples,
            "eps_audio": cfg.eps_audio,
            "eps_video": cfg.eps_video,
            "alpha_audio": cfg.alpha_audio,
            "alpha_video": cfg.alpha_video,
            "lambda_fusion_margin": cfg.lambda_fusion_margin,
            "lambda_preserve_margin": cfg.lambda_preserve_margin,
            "lambda_logit_distill": cfg.lambda_logit_distill,
            "preserve_margin": cfg.preserve_margin,
            "lambda_detector_evasion": cfg.lambda_detector_evasion,
            "lambda_detector_msp": cfg.lambda_detector_msp,
            "lambda_detector_mahalanobis": cfg.lambda_detector_mahalanobis,
            "lambda_detector_feature_squeezing": (
                cfg.lambda_detector_feature_squeezing
            ),
            "lambda_detector_lid": cfg.lambda_detector_lid,
            "target_clean_FPR": target_fpr,
        },
        "attack_metrics": {
            "FusionBreakFull": {
                "counts": attack_out["full_counts"],
                "rates": attack_out["full_rates"],
                "perturbation_norms": attack_out["full_norms"],
            },
            "AdaptiveFusionBreak": {
                "counts": attack_out["adaptive_counts"],
                "rates": attack_out["adaptive_rates"],
                "perturbation_norms": attack_out["adaptive_norms"],
            },
        },
        "detector_metrics": detector_metrics,
    }

    save_json(out_json, result)
    print(f"[done] Saved: {out_json}")

    f = result["attack_metrics"]["FusionBreakFull"]["rates"]
    a = result["attack_metrics"]["AdaptiveFusionBreak"]["rates"]
    fn = result["attack_metrics"]["FusionBreakFull"]["perturbation_norms"]
    an = result["attack_metrics"]["AdaptiveFusionBreak"]["perturbation_norms"]

    print(
        f"Full FusionBreak: ASR={100*f['fusion_ASR']:.2f}% | "
        f"Audio={100*f['audio_preservation_rate']:.2f}% | "
        f"Video={100*f['video_preservation_rate']:.2f}% | "
        f"FOSR={100*f['FOSR']:.2f}%"
    )
    print(
        f"AdaptiveFusionBreak: ASR={100*a['fusion_ASR']:.2f}% | "
        f"Audio={100*a['audio_preservation_rate']:.2f}% | "
        f"Video={100*a['video_preservation_rate']:.2f}% | "
        f"FOSR={100*a['FOSR']:.2f}%"
    )
    print(
        f"Norm check full: audio max={fn['audio_linf_max']:.6f}, "
        f"video max={fn['video_linf_max']:.6f}"
    )
    print(
        f"Norm check adaptive: audio max={an['audio_linf_max']:.6f}, "
        f"video max={an['video_linf_max']:.6f}"
    )

    return result


def _flatten_summary_row(
    result: dict[str, Any],
    attack_name: str,
) -> dict[str, Any]:
    metrics = result["attack_metrics"][attack_name]
    rates = metrics["rates"]
    norms = metrics["perturbation_norms"]

    row = {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "fusion_architecture": result["fusion_architecture"],
        "attacked_clean_samples": result["attacked_clean_samples"],
        "attack": attack_name,
        "fusion_ASR": rates["fusion_ASR"],
        "audio_preservation_rate": rates["audio_preservation_rate"],
        "video_preservation_rate": rates["video_preservation_rate"],
        "FOSR": rates["FOSR"],
        "audio_linf_max": norms["audio_linf_max"],
        "video_linf_max": norms["video_linf_max"],
        "audio_linf_mean": norms["audio_linf_mean"],
        "video_linf_mean": norms["video_linf_mean"],
    }

    dets = result.get("detector_metrics", {}).get(attack_name, {})
    if dets:
        detect_rates = []
        for detector_name, det_metrics in dets.items():
            key = "attack_detection_rate_at_threshold"
            if key in det_metrics:
                detect_rates.append(float(det_metrics[key]))
        if detect_rates:
            row["mean_detection_rate"] = float(np.mean(detect_rates))

    return row


def write_summary(cfg: Config) -> None:
    rows: list[dict[str, Any]] = []

    for path in sorted(
        cfg.log_dir.glob(
            f"controlled_full_vs_adaptive_v{RESULT_VERSION}_cremad_"
            "seed*_cremad_*.json"
        )
    ):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "attack_metrics" not in result:
            continue

        rows.append(_flatten_summary_row(result, "FusionBreakFull"))
        rows.append(_flatten_summary_row(result, "AdaptiveFusionBreak"))

    if not rows:
        return

    json_path = cfg.log_dir / (
        f"cremad_controlled_full_vs_adaptive_summary_v{RESULT_VERSION}.json"
    )
    csv_path = cfg.log_dir / (
        f"cremad_controlled_full_vs_adaptive_summary_v{RESULT_VERSION}.csv"
    )

    json_path.write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )

    keys = list(rows[0].keys())
    # Include any optional keys that appear later.
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] {json_path}")
    print(f"[summary] {csv_path}")


def main() -> None:
    cfg = Config(project_root=Path(__file__).resolve().parent)
    _configure_controlled_attack(cfg)
    cfg.ensure_dirs()

    seed_env = os.environ.get("FUSIONGAP_SEEDS", "").strip()
    arch_env = os.environ.get("FUSIONGAP_ARCHS", "").strip()

    seeds = (
        [int(x.strip()) for x in seed_env.split(",") if x.strip()]
        if seed_env
        else list(cfg.seeds)
    )
    archs = (
        [x.strip() for x in arch_env.split(",") if x.strip()]
        if arch_env
        else list(ARCHITECTURES)
    )
    invalid_archs = [arch for arch in archs if arch not in ARCHITECTURES]
    if invalid_archs:
        raise ValueError(
            f"Unsupported FUSIONGAP_ARCHS value(s): {invalid_archs}. "
            f"Choose from {list(ARCHITECTURES)}."
        )
    if not seeds:
        raise ValueError("No seeds were selected.")

    prepare_only = _env_flag("FUSIONGAP_PREPARE_ONLY", False)

    print("Project root:", cfg.project_root)
    print("Data root:", cfg.data_root())
    print("Seeds:", seeds)
    print("Architectures:", archs)
    print("Mode:", "prepare checkpoints only" if prepare_only else "controlled attacks")
    print(
        "Controlled attack settings:",
        {
            "steps": cfg.attack_steps,
            "restarts": cfg.random_restarts,
            "eps_audio": cfg.eps_audio,
            "eps_video": cfg.eps_video,
            "alpha_audio": cfg.alpha_audio,
            "alpha_video": cfg.alpha_video,
        },
    )

    plan = [(seed, arch) for seed in seeds for arch in archs]
    print(f"Runs: {len(plan)}")

    for seed, arch in tqdm(plan, desc="Controlled CREMA-D runs", unit="run"):
        run_one(seed, arch, cfg, prepare_only=prepare_only)
        if not prepare_only:
            write_summary(cfg)

    if prepare_only:
        print("All requested CREMA-D checkpoints are prepared and quality-checked.")
    else:
        write_summary(cfg)
        print("All controlled CREMA-D runs completed.")


if __name__ == "__main__":
    main()
