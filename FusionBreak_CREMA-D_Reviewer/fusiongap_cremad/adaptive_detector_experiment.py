from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .attacks import _clip_delta, _collect_clean_pool, _counts, _rates, _standard_pgd, _rnn_modules_train_mode, _zero_model_grads
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .detectors import collect_detector_scores, detection_metrics, fit_detector_pack, fusion_features, score_all_detectors
from .models import AudioNet, FusionNet, VideoNet
from .utils import get_device, save_json, set_seed

DATASET_NAME = "CREMA-D"
DATASET_TAG = "cremad"
DETECTOR_ARCHITECTURES = ["interaction", "concat", "gated", "group_gated"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, str):
                    v = v.replace('"', '""')
                    if "," in v:
                        v = f'"{v}"'
                values.append(str(v))
            f.write(",".join(values) + "\n")



def _zero_model_grads(*models: torch.nn.Module) -> None:
    for model in models:
        for p in model.parameters():
            p.grad = None


def _ensure_adaptive_defaults(cfg: Config) -> None:
    # General detector defaults.
    if not hasattr(cfg, "detector_batch_size"):
        cfg.detector_batch_size = cfg.batch_size
    if not hasattr(cfg, "detector_fit_samples"):
        cfg.detector_fit_samples = 2500
    if not hasattr(cfg, "detector_lid_k"):
        cfg.detector_lid_k = 20
    if not hasattr(cfg, "detector_mahalanobis_reg"):
        cfg.detector_mahalanobis_reg = 1e-3
    if not hasattr(cfg, "detector_target_fpr"):
        cfg.detector_target_fpr = 0.05
    if not hasattr(cfg, "skip_completed_adaptive_detector_results"):
        cfg.skip_completed_adaptive_detector_results = True

    # Attack defaults. These match the v3 decision-preserving attack.
    if not hasattr(cfg, "alpha_audio"):
        cfg.alpha_audio = 0.004
    if not hasattr(cfg, "alpha_video"):
        cfg.alpha_video = 0.003
    if not hasattr(cfg, "lambda_fusion_margin"):
        cfg.lambda_fusion_margin = 0.5
    if not hasattr(cfg, "lambda_preserve_margin"):
        cfg.lambda_preserve_margin = 5.0
    if not hasattr(cfg, "lambda_logit_distill"):
        cfg.lambda_logit_distill = 0.25
    if not hasattr(cfg, "preserve_margin"):
        cfg.preserve_margin = 0.05
    if not hasattr(cfg, "candidate_eval_interval"):
        cfg.candidate_eval_interval = 1
    if not hasattr(cfg, "clean_start_first_restart"):
        cfg.clean_start_first_restart = True
    if not hasattr(cfg, "attack_version"):
        cfg.attack_version = "v3_decision_preserving_attack"

    # Adaptive detector-aware defaults. We optimize only differentiable detector
    # scores, then evaluate all detectors. Feature squeezing and LID remain
    # evaluation-only because they are non-smooth / nearest-neighbor based.
    if not hasattr(cfg, "adaptive_detector_names"):
        # These are the differentiable / surrogate detector losses used during optimization.
        # All four detectors are still used in final candidate selection and evaluation.
        cfg.adaptive_detector_names = ["msp_hendrycks_gimpel", "mahalanobis_lee", "feature_squeezing_xu"]
    if not hasattr(cfg, "lambda_detector_evasion"):
        cfg.lambda_detector_evasion = 2.0
    if not hasattr(cfg, "lambda_detector_msp"):
        cfg.lambda_detector_msp = 0.5
    if not hasattr(cfg, "lambda_detector_mahalanobis"):
        cfg.lambda_detector_mahalanobis = 3.0
    if not hasattr(cfg, "lambda_detector_feature_squeezing"):
        cfg.lambda_detector_feature_squeezing = 5.0
    if not hasattr(cfg, "lambda_detector_lid"):
        # LID is expensive and nearest-neighbor based. It is used in candidate selection
        # and final evaluation by default, but not optimized directly unless this is > 0.
        cfg.lambda_detector_lid = 0.0
    if not hasattr(cfg, "adaptive_candidate_detector_weight"):
        cfg.adaptive_candidate_detector_weight = 10000.0
    if not hasattr(cfg, "adaptive_candidate_flag_weight"):
        cfg.adaptive_candidate_flag_weight = 50000.0
    if not hasattr(cfg, "adaptive_stealth_steps"):
        cfg.adaptive_stealth_steps = 30
    if not hasattr(cfg, "adaptive_stealth_alpha_scale"):
        cfg.adaptive_stealth_alpha_scale = 0.5
    if not hasattr(cfg, "adaptive_stealth_detector_multiplier"):
        cfg.adaptive_stealth_detector_multiplier = 2.0

    # Batch-wise progress reporting. This lets you watch evasion rate during
    # long runs and stop the process manually if the setting is not promising.
    # Override with: $env:FUSIONGAP_PROGRESS_EVERY_BATCHES="1"
    if not hasattr(cfg, "adaptive_progress_every_batches"):
        cfg.adaptive_progress_every_batches = int(os.environ.get("FUSIONGAP_PROGRESS_EVERY_BATCHES", "1"))
    if not hasattr(cfg, "adaptive_progress_save_jsonl"):
        cfg.adaptive_progress_save_jsonl = os.environ.get("FUSIONGAP_PROGRESS_SAVE_JSONL", "1").strip().lower() not in {"0", "false", "no"}


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    return (
        cfg.checkpoint_dir / f"audio_seed{seed}.pt",
        cfg.checkpoint_dir / f"video_seed{seed}.pt",
        cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt",
    )


def _load_checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}: {type(ckpt)}")
    for key in ("model", "model_state_dict", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    if any(isinstance(k, str) and k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def _load_models(seed: int, arch: str, cfg: Config, device: torch.device):
    audio_ckpt, video_ckpt, fusion_ckpt = _checkpoint_paths(cfg, seed, arch)
    missing = [str(p) for p in [audio_ckpt, video_ckpt, fusion_ckpt] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s). Run the main CREMA-D experiments first. Missing: " + "; ".join(missing))
    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio.load_state_dict(_load_checkpoint_state_dict(audio_ckpt, device))
    video.load_state_dict(_load_checkpoint_state_dict(video_ckpt, device))

    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)
    fusion = FusionNet(fusion_audio, fusion_video, arch, num_classes=cfg.num_classes).to(device)
    fusion.load_state_dict(_load_checkpoint_state_dict(fusion_ckpt, device))
    audio.eval(); video.eval(); fusion.eval()
    return audio, video, fusion


def _true_minus_other_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -1e9)
    other_logit = masked.max(dim=1).values
    return true_logit - other_logit


def _kl_to_clean(current_logits: torch.Tensor, clean_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(current_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction="none",
    ).sum(dim=1)


def _msp_score_grad(fusion_model, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(fusion_model(xa, xv), dim=1)
    return 1.0 - probs.max(dim=1).values


def _mahalanobis_score_grad(fusion_model, pack, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    z = F.normalize(fusion_features(fusion_model, xa, xv).float(), dim=1)
    means = pack.class_means.to(z.device).float()
    precision = pack.precision.to(z.device).float()
    diff = z[:, None, :] - means[None, :, :]
    left = torch.matmul(diff, precision)
    dist = (left * diff).sum(dim=2)
    return dist.min(dim=1).values


def _ste_round_quantize(x: torch.Tensor, lo: float, hi: float, bits: int) -> torch.Tensor:
    """Straight-through quantization used as a BPDA-style surrogate.

    Forward pass matches bit-depth reduction; backward pass is approximately
    identity through the quantizer, which is the standard practical choice for
    optimizing through non-smooth preprocessing defenses.
    """
    levels = float(2 ** bits - 1)
    x_clip = x.clamp(lo, hi)
    q = ((x_clip - lo) / (hi - lo) * levels).round() / levels
    q = q * (hi - lo) + lo
    return x_clip + (q - x_clip).detach()


def _feature_squeezing_score_grad(fusion_model, xa: torch.Tensor, xv: torch.Tensor, audio_bits: int = 5, video_bits: int = 5) -> torch.Tensor:
    """Differentiable/BPDA surrogate for the feature-squeezing detector score."""
    def smooth_audio(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def smooth_video(x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        y = x.reshape(b * t, c, h, w)
        y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        return y.reshape(b, t, c, h, w)

    p = F.softmax(fusion_model(xa, xv), dim=1)
    sq_a = smooth_audio(_ste_round_quantize(xa, -5.0, 5.0, audio_bits))
    sq_v = smooth_video(_ste_round_quantize(xv, -1.0, 1.0, video_bits))
    sq_p = F.softmax(fusion_model(sq_a, sq_v), dim=1)
    return torch.sum(torch.abs(p - sq_p), dim=1)


def _lid_score_grad(fusion_model, pack, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    """Differentiable-ish LID surrogate through the selected nearest neighbors.

    This is off by default because it is slower and less stable than MSP,
    Mahalanobis, and feature-squeezing losses. Set lambda_detector_lid > 0 to use.
    """
    z = F.normalize(fusion_features(fusion_model, xa, xv).float(), dim=1)
    bank = pack.feature_bank.to(z.device).float().detach()
    k = min(int(pack.lid_k), max(2, int(bank.shape[0]) - 1))
    d = torch.cdist(z, bank, p=2)
    knn = torch.topk(d, k=k, largest=False).values.clamp_min(1e-12)
    rk = knn[:, -1:].clamp_min(1e-12)
    logs = torch.log(knn / rk).clamp_max(-1e-12)
    return -float(k) / logs.sum(dim=1).clamp_max(-1e-12)


def _normalized_detector_excess(scores: dict[str, torch.Tensor], stats: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized detector excess and number of flags per example.

    Higher detector scores are more suspicious. A score below its clean-val
    threshold contributes zero penalty.
    """
    first = next(iter(scores.values()))
    excess = torch.zeros(first.shape[0], device=first.device)
    flags = torch.zeros(first.shape[0], device=first.device)
    for name, s in scores.items():
        if name not in stats:
            continue
        st = stats[name]
        e = F.relu((s.float() - st["threshold"]) / st["scale"].clamp_min(1e-6))
        excess = excess + e.pow(2)
        flags = flags + (s.float() >= st["threshold"]).float()
    return excess, flags


def _threshold_stats(val_clean_scores: dict[str, np.ndarray], target_fpr: float, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    stats = {}
    for name, arr in val_clean_scores.items():
        arr = np.asarray(arr, dtype=np.float64)
        threshold = float(np.quantile(arr, 1.0 - target_fpr))
        std = float(np.std(arr))
        if not np.isfinite(std) or std < 1e-6:
            std = max(abs(threshold), 1.0)
        stats[name] = {
            "threshold": torch.tensor(threshold, device=device, dtype=torch.float32),
            "scale": torch.tensor(std, device=device, dtype=torch.float32),
        }
    return stats


def _adaptive_detector_penalty(fusion_model, pack, xa: torch.Tensor, xv: torch.Tensor, cfg: Config, stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    penalties = []
    names = list(getattr(cfg, "adaptive_detector_names", ["msp_hendrycks_gimpel", "mahalanobis_lee", "feature_squeezing_xu"]))
    if "msp_hendrycks_gimpel" in names and "msp_hendrycks_gimpel" in stats:
        s = _msp_score_grad(fusion_model, xa, xv)
        st = stats["msp_hendrycks_gimpel"]
        penalties.append(float(getattr(cfg, "lambda_detector_msp", 0.5)) * F.relu((s - st["threshold"]) / st["scale"].clamp_min(1e-6)).pow(2))
    if "mahalanobis_lee" in names and "mahalanobis_lee" in stats:
        s = _mahalanobis_score_grad(fusion_model, pack, xa, xv)
        st = stats["mahalanobis_lee"]
        penalties.append(float(getattr(cfg, "lambda_detector_mahalanobis", 3.0)) * F.relu((s - st["threshold"]) / st["scale"].clamp_min(1e-6)).pow(2))
    if "feature_squeezing_xu" in names and "feature_squeezing_xu" in stats:
        s = _feature_squeezing_score_grad(fusion_model, xa, xv)
        st = stats["feature_squeezing_xu"]
        penalties.append(float(getattr(cfg, "lambda_detector_feature_squeezing", 5.0)) * F.relu((s - st["threshold"]) / st["scale"].clamp_min(1e-6)).pow(2))
    if float(getattr(cfg, "lambda_detector_lid", 0.0)) > 0.0 and "lid_ma" in names and "lid_ma" in stats:
        s = _lid_score_grad(fusion_model, pack, xa, xv)
        st = stats["lid_ma"]
        penalties.append(float(getattr(cfg, "lambda_detector_lid", 0.0)) * F.relu((s - st["threshold"]) / st["scale"].clamp_min(1e-6)).pow(2))
    if not penalties:
        return torch.zeros(xa.shape[0], device=xa.device)
    return torch.stack(penalties, dim=0).sum(dim=0)


@torch.no_grad()
def _candidate_score_adaptive(audio_model, video_model, fusion_model, pack, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor, cfg: Config, stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
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
    a_margin = _true_minus_other_margin(audio_logits, y)
    v_margin = _true_minus_other_margin(video_logits, y)

    # Candidate selection uses the actual detector implementations for all four
    # detectors, not only the differentiable losses. This is the critical v2 fix.
    detector_scores = score_all_detectors(fusion_model, pack, xa, xv)
    detector_excess, detector_flags = _normalized_detector_excess(detector_scores, stats)
    all_undetected = detector_flags <= 0.0

    return (
        (fusion_only & all_undetected).float() * 1_000_000_000.0
        + fusion_only.float() * 100_000_000.0
        + one_unimodal_ok.float() * 10_000_000.0
        + fusion_fail.float() * 1_000_000.0
        + audio_ok.float() * 10_000.0
        + video_ok.float() * 10_000.0
        + fusion_ce
        + 0.01 * (a_margin + v_margin)
        - float(getattr(cfg, "adaptive_candidate_flag_weight", 50000.0)) * detector_flags
        - float(getattr(cfg, "adaptive_candidate_detector_weight", 10000.0)) * detector_excess
    )


def _adaptive_fusiongap(audio_model, video_model, fusion_model, pack, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device, stats: dict[str, dict[str, torch.Tensor]]):
    xa0 = xa.detach()
    xv0 = xv.detach()
    with torch.no_grad():
        clean_audio_logits = audio_model(xa0).detach()
        clean_video_logits = video_model(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score_adaptive(audio_model, video_model, fusion_model, pack, best_xa, best_xv, y, cfg, stats)

    def maybe_update_best(xa_cand: torch.Tensor, xv_cand: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score_adaptive(audio_model, video_model, fusion_model, pack, xa_cand, xv_cand, y, cfg, stats)
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
            xa_adv = _clip_delta(xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio), xa0, cfg.eps_audio).detach()
            xv_adv = _clip_delta(xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video), xv0, cfg.eps_video).detach()
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
                preservation_hinge = F.relu(cfg.preserve_margin - audio_margin) + F.relu(cfg.preserve_margin - video_margin)
                logit_distill = _kl_to_clean(audio_logits, clean_audio_logits) + _kl_to_clean(video_logits, clean_video_logits)
                det_penalty = _adaptive_detector_penalty(fusion_model, pack, xa_adv, xv_adv, cfg, stats)
    
                objective_vec = (
                    fusion_ce
                    + cfg.lambda_fusion_margin * fusion_attack_margin
                    - cfg.lambda_preserve_margin * preservation_hinge
                    - cfg.lambda_logit_distill * logit_distill
                    - cfg.lambda_detector_evasion * det_penalty
                )
                objective = objective_vec.mean()
                ga, gv = torch.autograd.grad(objective, [xa_adv, xv_adv], retain_graph=False, create_graph=False)
                with torch.no_grad():
                    xa_adv = _clip_delta(xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio).detach()
                    xv_adv = _clip_delta(xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video).detach()
                if cfg.candidate_eval_interval <= 1 or step % cfg.candidate_eval_interval == 0 or step == cfg.attack_steps:
                    maybe_update_best(xa_adv, xv_adv)

    # Final stealth refinement: start from the best fusion-only candidates found
    # and spend extra small steps reducing detector scores while preserving the
    # fusion-only failure. Candidate selection still decides whether refinements
    # are accepted, so this should not replace a high-FOSR point with a low-FOSR one.
    stealth_steps = int(getattr(cfg, "adaptive_stealth_steps", 0))
    if stealth_steps > 0:
        xa_adv = best_xa.clone().detach()
        xv_adv = best_xv.clone().detach()
        maybe_update_best(xa_adv, xv_adv)
        alpha_scale = float(getattr(cfg, "adaptive_stealth_alpha_scale", 0.5))
        detector_mult = float(getattr(cfg, "adaptive_stealth_detector_multiplier", 2.0))
        with _rnn_modules_train_mode(audio_model, video_model, fusion_model):
            for step in range(1, stealth_steps + 1):
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
                preservation_hinge = F.relu(cfg.preserve_margin - audio_margin) + F.relu(cfg.preserve_margin - video_margin)
                logit_distill = _kl_to_clean(audio_logits, clean_audio_logits) + _kl_to_clean(video_logits, clean_video_logits)
                det_penalty = _adaptive_detector_penalty(fusion_model, pack, xa_adv, xv_adv, cfg, stats)

                objective_vec = (
                    0.5 * fusion_ce
                    + cfg.lambda_fusion_margin * fusion_attack_margin
                    - 1.25 * cfg.lambda_preserve_margin * preservation_hinge
                    - cfg.lambda_logit_distill * logit_distill
                    - detector_mult * cfg.lambda_detector_evasion * det_penalty
                )
                objective = objective_vec.mean()
                ga, gv = torch.autograd.grad(objective, [xa_adv, xv_adv], retain_graph=False, create_graph=False)
                with torch.no_grad():
                    xa_adv = _clip_delta(xa_adv + alpha_scale * cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio).detach()
                    xv_adv = _clip_delta(xv_adv + alpha_scale * cfg.alpha_video * gv.sign(), xv0, cfg.eps_video).detach()
                if cfg.candidate_eval_interval <= 1 or step % cfg.candidate_eval_interval == 0 or step == stealth_steps:
                    maybe_update_best(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


@torch.no_grad()
def _cache_inputs(cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return cache["audio"].float(), cache["video"].float()


def _zero_counts() -> dict[str, int]:
    return {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0}


def _add_counts(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] += int(v)


def _join(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in d.items()}


def _threshold_float(stats: dict[str, dict[str, torch.Tensor]], name: str) -> float:
    return float(stats[name]["threshold"].detach().cpu().item())


def _zero_detection_running(detector_names: list[str]) -> dict[str, dict[str, int]]:
    return {name: {"detected": 0, "n": 0} for name in detector_names}


def _add_detection_running(running: dict[str, dict[str, int]], scores: dict[str, np.ndarray], stats: dict[str, dict[str, torch.Tensor]]) -> None:
    for name, arr in scores.items():
        if name not in stats:
            continue
        arr = np.asarray(arr)
        threshold = _threshold_float(stats, name)
        running.setdefault(name, {"detected": 0, "n": 0})
        running[name]["detected"] += int(np.sum(arr >= threshold))
        running[name]["n"] += int(arr.size)


def _running_detection_rates(running: dict[str, dict[str, int]]) -> dict[str, float]:
    out = {}
    for name, obj in running.items():
        n = int(obj.get("n", 0))
        out[name] = float(obj.get("detected", 0)) / max(1, n)
    return out


def _mean_rate(rates: dict[str, float]) -> float:
    return float(np.mean(list(rates.values()))) if rates else 0.0


def _compact_detector_status(rates: dict[str, float]) -> str:
    short = {
        "msp_hendrycks_gimpel": "MSP",
        "feature_squeezing_xu": "FS",
        "mahalanobis_lee": "Mah",
        "lid_ma": "LID",
    }
    parts = []
    for name in ["feature_squeezing_xu", "mahalanobis_lee", "lid_ma", "msp_hendrycks_gimpel"]:
        if name in rates:
            # Evasion is what the user wants to monitor: 1 - detection rate.
            parts.append(f"{short[name]} evasion={100.0 * (1.0 - rates[name]):.1f}%")
    return " | ".join(parts)


def _progress_payload(
    *,
    batch_idx: int,
    num_batches: int,
    processed: int,
    adaptive_counts: dict[str, int],
    pgd_counts: dict[str, int],
    adaptive_detection_running: dict[str, dict[str, int]],
    pgd_detection_running: dict[str, dict[str, int]],
) -> dict[str, Any]:
    adaptive_rates = _rates(adaptive_counts)
    pgd_rates = _rates(pgd_counts)
    adaptive_det = _running_detection_rates(adaptive_detection_running)
    pgd_det = _running_detection_rates(pgd_detection_running)
    return {
        "batch": int(batch_idx),
        "num_batches": int(num_batches),
        "processed_samples": int(processed),
        "AdaptiveFusionGap": {
            "fusion_ASR": adaptive_rates.get("fusion_ASR", 0.0),
            "FOSR": adaptive_rates.get("FOSR", 0.0),
            "mean_detection_rate": _mean_rate(adaptive_det),
            "mean_evasion_rate": 1.0 - _mean_rate(adaptive_det),
            "detector_detection_rates": adaptive_det,
            "detector_evasion_rates": {k: 1.0 - v for k, v in adaptive_det.items()},
        },
        "PGD": {
            "fusion_ASR": pgd_rates.get("fusion_ASR", 0.0),
            "FOSR": pgd_rates.get("FOSR", 0.0),
            "mean_detection_rate": _mean_rate(pgd_det),
            "mean_evasion_rate": 1.0 - _mean_rate(pgd_det),
            "detector_detection_rates": pgd_det,
            "detector_evasion_rates": {k: 1.0 - v for k, v in pgd_det.items()},
        },
    }


def _print_batch_progress(payload: dict[str, Any]) -> None:
    a = payload["AdaptiveFusionGap"]
    p = payload["PGD"]
    a_det = a["detector_detection_rates"]
    p_det = p["detector_detection_rates"]
    tqdm.write(
        f"[batch {payload['batch']}/{payload['num_batches']} | n={payload['processed_samples']}] "
        f"AdaptiveFusionGap: FOSR={100*a['FOSR']:.1f}%, mean evasion={100*a['mean_evasion_rate']:.1f}% "
        f"({_compact_detector_status(a_det)})"
    )
    tqdm.write(
        f"[batch {payload['batch']}/{payload['num_batches']} | n={payload['processed_samples']}] "
        f"PGD: fusion ASR={100*p['fusion_ASR']:.1f}%, FOSR={100*p['FOSR']:.1f}%, mean evasion={100*p['mean_evasion_rate']:.1f}% "
        f"({_compact_detector_status(p_det)})"
    )


def _generate_adaptive_scores_and_counts(
    audio,
    video,
    fusion,
    clean_pool,
    pack,
    cfg: Config,
    device: torch.device,
    stats: dict[str, dict[str, torch.Tensor]],
    progress_path: Path | None = None,
):
    xa_all, xv_all, y_all = clean_pool
    ds = TensorDataset(xa_all.float(), xv_all.float(), y_all.long())
    loader = DataLoader(ds, batch_size=cfg.attack_batch_size, shuffle=False, num_workers=0)
    clean_scores_parts: dict[str, list[np.ndarray]] = {}
    adaptive_scores_parts: dict[str, list[np.ndarray]] = {}
    pgd_scores_parts: dict[str, list[np.ndarray]] = {}
    adaptive_counts = _zero_counts()
    pgd_counts = _zero_counts()

    detector_names = ["feature_squeezing_xu", "mahalanobis_lee", "lid_ma", "msp_hendrycks_gimpel"]
    adaptive_detection_running = _zero_detection_running(detector_names)
    pgd_detection_running = _zero_detection_running(detector_names)
    processed = 0
    progress_every = max(1, int(getattr(cfg, "adaptive_progress_every_batches", 1)))

    if progress_path is not None and bool(getattr(cfg, "adaptive_progress_save_jsonl", True)):
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")
        print(f"[progress] Batch-wise progress JSONL: {progress_path}")

    num_batches = len(loader)
    bar = tqdm(loader, desc="Adaptive detector-aware attacks", unit="batch")
    for batch_idx, (xa, xv, y) in enumerate(bar, start=1):
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)
        xa_adapt, xv_adapt = _adaptive_fusiongap(audio, video, fusion, pack, xa, xv, y, cfg, device, stats)
        xa_pgd, xv_pgd = _standard_pgd(audio, video, fusion, xa, xv, y, cfg, device)

        _add_counts(adaptive_counts, _counts(audio, video, fusion, xa_adapt, xv_adapt, y))
        _add_counts(pgd_counts, _counts(audio, video, fusion, xa_pgd, xv_pgd, y))
        processed += int(y.numel())

        batch_scores_by_target: dict[str, dict[str, np.ndarray]] = {}
        for target_name, target, pair in [
            ("clean", clean_scores_parts, (xa, xv)),
            ("AdaptiveFusionGap", adaptive_scores_parts, (xa_adapt, xv_adapt)),
            ("PGD", pgd_scores_parts, (xa_pgd, xv_pgd)),
        ]:
            scores = collect_detector_scores(fusion, pack, pair[0].detach().cpu(), pair[1].detach().cpu(), cfg, device)
            batch_scores_by_target[target_name] = scores
            for name, arr in scores.items():
                target.setdefault(name, []).append(arr)

        _add_detection_running(adaptive_detection_running, batch_scores_by_target["AdaptiveFusionGap"], stats)
        _add_detection_running(pgd_detection_running, batch_scores_by_target["PGD"], stats)

        payload = _progress_payload(
            batch_idx=batch_idx,
            num_batches=num_batches,
            processed=processed,
            adaptive_counts=adaptive_counts,
            pgd_counts=pgd_counts,
            adaptive_detection_running=adaptive_detection_running,
            pgd_detection_running=pgd_detection_running,
        )

        # Compact progress bar postfix. Full detector-by-detector values are printed/saved.
        bar.set_postfix({
            "adapt_FOSR": f"{100*payload['AdaptiveFusionGap']['FOSR']:.1f}%",
            "adapt_ev": f"{100*payload['AdaptiveFusionGap']['mean_evasion_rate']:.1f}%",
            "pgd_ev": f"{100*payload['PGD']['mean_evasion_rate']:.1f}%",
        })

        if batch_idx % progress_every == 0 or batch_idx == num_batches:
            _print_batch_progress(payload)
            if progress_path is not None and bool(getattr(cfg, "adaptive_progress_save_jsonl", True)):
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")

    return _join(clean_scores_parts), _join(adaptive_scores_parts), _join(pgd_scores_parts), adaptive_counts, pgd_counts


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"adaptive_detectors_v3_cremad_seed{seed}_cremad_{arch}.json"


def run_one_adaptive_detector(seed: int, arch: str, cfg: Config) -> dict[str, Any]:
    _ensure_adaptive_defaults(cfg)
    cfg.ensure_dirs()
    out_json = _result_path(cfg, seed, arch)
    if cfg.skip_completed_adaptive_detector_results and out_json.exists():
        print(f"[skip] Adaptive detector result exists: {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"CREMA-D adaptive detector-aware evaluation: seed={seed}, fusion_architecture={arch}")
    print("=" * 100)
    set_seed(seed)
    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", seed, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", seed, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", seed, cfg, device)

    audio, video, fusion = _load_models(seed, arch, cfg, device)
    pack = fit_detector_pack(fusion, train_cache, cfg, device)
    clean_pool = _collect_clean_pool(audio, video, fusion, test_cache, cfg, device)
    if clean_pool is None:
        result = {"dataset": DATASET_NAME, "seed": seed, "fusion_architecture": arch, "message": "No all-three-correct clean samples."}
        save_json(out_json, result)
        return result

    xa_clean, xv_clean, y_clean = clean_pool
    print(f"[adaptive] Evaluating on {int(y_clean.numel())} all-three-correct clean samples.")
    target_fpr = float(cfg.detector_target_fpr)
    val_xa, val_xv = _cache_inputs(val_cache)
    val_clean_scores = collect_detector_scores(fusion, pack, val_xa, val_xv, cfg, device)
    stats = _threshold_stats(val_clean_scores, target_fpr, device)

    progress_path = cfg.log_dir / f"adaptive_detectors_v3_progress_cremad_seed{seed}_cremad_{arch}.jsonl"
    clean_scores, adaptive_scores, pgd_scores, adaptive_counts, pgd_counts = _generate_adaptive_scores_and_counts(
        audio, video, fusion, clean_pool, pack, cfg, device, stats, progress_path=progress_path
    )
    adaptive_rates = _rates(adaptive_counts)
    pgd_rates = _rates(pgd_counts)

    metrics: dict[str, Any] = {"AdaptiveFusionGap": {}, "PGD": {}}
    for detector_name in sorted(clean_scores.keys()):
        metrics["AdaptiveFusionGap"][detector_name] = detection_metrics(
            clean_scores[detector_name], adaptive_scores[detector_name], val_clean_scores[detector_name], target_fpr
        )
        metrics["PGD"][detector_name] = detection_metrics(
            clean_scores[detector_name], pgd_scores[detector_name], val_clean_scores[detector_name], target_fpr
        )

    result = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "attacked_clean_samples": int(y_clean.numel()),
        "adaptive_objective": {
            "optimized_detectors": list(getattr(cfg, "adaptive_detector_names", [])),
            "note": "Adaptive FusionGap v2 optimizes MSP, Mahalanobis, and a BPDA-style feature-squeezing surrogate by default; all four detectors are used for candidate selection and final evaluation.",
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
            "lambda_detector_evasion": cfg.lambda_detector_evasion,
            "lambda_detector_msp": cfg.lambda_detector_msp,
            "lambda_detector_mahalanobis": cfg.lambda_detector_mahalanobis,
            "lambda_detector_feature_squeezing": cfg.lambda_detector_feature_squeezing,
            "lambda_detector_lid": cfg.lambda_detector_lid,
            "adaptive_candidate_detector_weight": cfg.adaptive_candidate_detector_weight,
            "adaptive_candidate_flag_weight": cfg.adaptive_candidate_flag_weight,
            "adaptive_stealth_steps": cfg.adaptive_stealth_steps,
            "adaptive_stealth_alpha_scale": cfg.adaptive_stealth_alpha_scale,
            "adaptive_stealth_detector_multiplier": cfg.adaptive_stealth_detector_multiplier,
            "preserve_margin": cfg.preserve_margin,
            "target_clean_FPR": target_fpr,
            "attack_version": "adaptive_detector_aware_fusiongap_v3_batch_progress",
        },
        "attack_metrics": {
            "AdaptiveFusionGap": {"counts": adaptive_counts, "rates": adaptive_rates},
            "PGD": {"counts": pgd_counts, "rates": pgd_rates},
        },
        "detector_metrics": metrics,
    }
    save_json(out_json, result)
    print(f"[done] Saved adaptive detector result: {out_json}")
    return result


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    attack_metrics = result.get("attack_metrics", {})
    for attack_name, dets in result.get("detector_metrics", {}).items():
        rates = attack_metrics.get(attack_name, {}).get("rates", {})
        for detector_name, m in dets.items():
            rows.append({
                "dataset": result.get("dataset", DATASET_NAME),
                "seed": result.get("seed", ""),
                "fusion_architecture": result.get("fusion_architecture", ""),
                "attacked_clean_samples": result.get("attacked_clean_samples", ""),
                "attack": attack_name,
                "detector": detector_name,
                "fusion_ASR": rates.get("fusion_ASR", ""),
                "audio_preservation_rate": rates.get("audio_preservation_rate", ""),
                "video_preservation_rate": rates.get("video_preservation_rate", ""),
                "FOSR": rates.get("FOSR", ""),
                "AUROC": m.get("AUROC", ""),
                "AUPR_adv": m.get("AUPR_adv", ""),
                "detect_rate_at_5pct_clean_val_FPR": m.get("attack_detection_rate_at_threshold", ""),
                "clean_test_FPR_at_threshold": m.get("clean_test_FPR_at_threshold", ""),
                "clean_score_mean": m.get("clean_score_mean", ""),
                "adv_score_mean": m.get("adv_score_mean", ""),
            })
    return rows


def _attack_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attack_name, obj in result.get("attack_metrics", {}).items():
        rates = obj.get("rates", {})
        rows.append({
            "dataset": result.get("dataset", DATASET_NAME),
            "seed": result.get("seed", ""),
            "fusion_architecture": result.get("fusion_architecture", ""),
            "attacked_clean_samples": result.get("attacked_clean_samples", ""),
            "attack": attack_name,
            "fusion_ASR": rates.get("fusion_ASR", ""),
            "audio_preservation_rate": rates.get("audio_preservation_rate", ""),
            "video_preservation_rate": rates.get("video_preservation_rate", ""),
            "FOSR": rates.get("FOSR", ""),
        })
    return rows


def write_adaptive_detector_summary(cfg: Config) -> None:
    rows = []
    attack_rows = []
    for p in sorted(cfg.log_dir.glob("adaptive_detectors_v3_cremad_seed*_cremad_*.json")):
        try:
            result = json.loads(p.read_text(encoding="utf-8"))
            rows.extend(_summary_rows(result))
            attack_rows.extend(_attack_summary_rows(result))
        except Exception as e:
            print(f"[summary] Skipped {p}: {e}")
    if not rows:
        return
    write_csv(cfg.artifact_dir / "cremad_adaptive_detector_summary_v3.csv", rows)
    save_json(cfg.artifact_dir / "cremad_adaptive_detector_summary_v3.json", {"rows": rows})
    write_csv(cfg.artifact_dir / "cremad_adaptive_attack_summary_v3.csv", attack_rows)
    save_json(cfg.artifact_dir / "cremad_adaptive_attack_summary_v3.json", {"rows": attack_rows})
    print(f"[summary] Saved: {cfg.artifact_dir / 'cremad_adaptive_detector_summary_v3.csv'}")


def run_all_adaptive_detectors(cfg: Config) -> None:
    _ensure_adaptive_defaults(cfg)
    cfg.ensure_dirs()
    print("Project root:", cfg.project_root)
    print("Data root:", cfg.data_root())
    print("Adaptive detector logs:", cfg.log_dir)
    print("Adaptive detector artifacts:", cfg.artifact_dir)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Optimized detectors:", cfg.adaptive_detector_names)

    seed_env = os.environ.get("FUSIONGAP_SEEDS", "").strip()
    arch_env = os.environ.get("FUSIONGAP_ARCHS", "").strip()
    seeds = [int(s.strip()) for s in seed_env.split(",") if s.strip()] if seed_env else list(cfg.seeds)
    archs = [a.strip() for a in arch_env.split(",") if a.strip()] if arch_env else list(DETECTOR_ARCHITECTURES)
    plan = [(seed, arch) for seed in seeds for arch in archs]
    print("Evaluation plan:", plan)
    for seed, arch in tqdm(plan, desc="CREMA-D adaptive detector evaluations", unit="run"):
        run_one_adaptive_detector(seed, arch, cfg)
        write_adaptive_detector_summary(cfg)
    write_adaptive_detector_summary(cfg)
    print("All requested CREMA-D adaptive detector evaluations completed.")


def main() -> None:
    cfg = Config(project_root=Path(__file__).resolve().parents[1])
    run_all_adaptive_detectors(cfg)
