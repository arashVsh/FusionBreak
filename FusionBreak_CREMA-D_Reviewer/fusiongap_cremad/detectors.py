from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


@dataclass
class DetectorPack:
    feature_bank: torch.Tensor
    label_bank: torch.Tensor
    class_means: torch.Tensor
    precision: torch.Tensor
    lid_k: int


# ---------------------------------------------------------------------------
# Fusion features
# ---------------------------------------------------------------------------

def fusion_features(fusion_model, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    """Return the fused representation before the fusion classifier.

    This mirrors FusionNet.forward without applying the final classifier. It is
    intentionally local to the detector code so the detector add-on can be
    dropped into an already trained project without changing the model class.
    """
    af = fusion_model.audio_model.features(audio)
    vf = fusion_model.video_model.features(video)
    arch = fusion_model.architecture

    if hasattr(fusion_model, "features"):
        try:
            return fusion_model.features(audio, video)
        except Exception:
            pass

    if arch == "concat":
        return torch.cat([af, vf], dim=1)
    if arch == "interaction":
        return torch.cat([af, vf, torch.abs(af - vf), af * vf], dim=1)
    if arch == "gated":
        g = fusion_model.gate(torch.cat([af, vf], dim=1))
        return g * af + (1.0 - g) * vf
    if arch == "group_gated":
        z = torch.cat([af, vf], dim=1)
        g = fusion_model.gate(z)
        return g * z
    if arch == "late":
        alog = fusion_model.audio_model.classifier(af)
        vlog = fusion_model.video_model.classifier(vf)
        w = torch.softmax(fusion_model.logit_weight, dim=0)
        return w[0] * alog + w[1] * vlog
    raise ValueError(f"Unknown fusion architecture: {arch}")


@torch.no_grad()
def _collect_features(fusion_model, cache: dict[str, torch.Tensor], cfg: Config, device: torch.device, max_samples: int | None = None):
    ds = TensorDataset(cache["audio"].float(), cache["video"].float(), cache["labels"])
    loader = DataLoader(ds, batch_size=getattr(cfg, "detector_batch_size", cfg.batch_size), shuffle=False, num_workers=0)
    feats, labels = [], []
    seen = 0
    fusion_model.eval()
    for xa, xv, y in loader:
        if max_samples is not None and seen >= max_samples:
            break
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)
        f = fusion_features(fusion_model, xa, xv)
        if max_samples is not None and seen + f.shape[0] > max_samples:
            keep = max_samples - seen
            f = f[:keep]
            y = y[:keep]
        feats.append(f.detach())
        labels.append(y.detach())
        seen += int(f.shape[0])
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# ---------------------------------------------------------------------------
# Detector fitting
# ---------------------------------------------------------------------------

def fit_detector_pack(fusion_model, train_cache: dict[str, torch.Tensor], cfg: Config, device: torch.device) -> DetectorPack:
    max_train = int(getattr(cfg, "detector_fit_samples", 2500))
    lid_k = int(getattr(cfg, "detector_lid_k", 20))
    reg = float(getattr(cfg, "detector_mahalanobis_reg", 1e-3))

    feature_bank, label_bank = _collect_features(fusion_model, train_cache, cfg, device, max_train)
    feature_bank = F.normalize(feature_bank.float(), dim=1)
    label_bank = label_bank.long()

    num_classes = int(getattr(cfg, "num_classes", int(label_bank.max().item()) + 1))
    means = []
    centered_parts = []
    global_mean = feature_bank.mean(dim=0)
    for c in range(num_classes):
        mask = label_bank == c
        if mask.any():
            mu = feature_bank[mask].mean(dim=0)
            means.append(mu)
            centered_parts.append(feature_bank[mask] - mu)
        else:
            means.append(global_mean)
    means = torch.stack(means, dim=0)
    centered = torch.cat(centered_parts, dim=0) if centered_parts else feature_bank - global_mean
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    cov = cov + reg * torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
    precision = torch.linalg.pinv(cov)

    return DetectorPack(
        feature_bank=feature_bank.detach(),
        label_bank=label_bank.detach(),
        class_means=means.detach(),
        precision=precision.detach(),
        lid_k=lid_k,
    )


# ---------------------------------------------------------------------------
# Scores. Higher means more likely adversarial.
# ---------------------------------------------------------------------------

@torch.no_grad()
def msp_score(fusion_model, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(fusion_model(xa, xv), dim=1)
    return 1.0 - probs.max(dim=1).values


@torch.no_grad()
def feature_squeezing_score(fusion_model, xa: torch.Tensor, xv: torch.Tensor, audio_bits: int = 5, video_bits: int = 5) -> torch.Tensor:
    def quantize_audio(x: torch.Tensor, bits: int) -> torch.Tensor:
        # Audio spectrograms are standardized, so use a broad fixed clipping range.
        lo, hi = -5.0, 5.0
        levels = float(2**bits - 1)
        y = ((x.clamp(lo, hi) - lo) / (hi - lo) * levels).round() / levels
        return y * (hi - lo) + lo

    def quantize_video(x: torch.Tensor, bits: int) -> torch.Tensor:
        lo, hi = -1.0, 1.0
        levels = float(2**bits - 1)
        y = ((x.clamp(lo, hi) - lo) / (hi - lo) * levels).round() / levels
        return y * (hi - lo) + lo

    def smooth_audio(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def smooth_video(x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        y = x.reshape(b * t, c, h, w)
        y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        return y.reshape(b, t, c, h, w)

    clean_p = F.softmax(fusion_model(xa, xv), dim=1)
    sq_a = smooth_audio(quantize_audio(xa, audio_bits))
    sq_v = smooth_video(quantize_video(xv, video_bits))
    sq_p = F.softmax(fusion_model(sq_a, sq_v), dim=1)
    return torch.sum(torch.abs(clean_p - sq_p), dim=1)


@torch.no_grad()
def mahalanobis_score(fusion_model, pack: DetectorPack, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    z = F.normalize(fusion_features(fusion_model, xa, xv).float(), dim=1)
    diff = z[:, None, :] - pack.class_means[None, :, :]
    # Squared Mahalanobis distance for every class.
    left = torch.matmul(diff, pack.precision)
    dist = (left * diff).sum(dim=2)
    return dist.min(dim=1).values


@torch.no_grad()
def lid_score(fusion_model, pack: DetectorPack, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    z = F.normalize(fusion_features(fusion_model, xa, xv).float(), dim=1)
    k = min(int(pack.lid_k), max(2, int(pack.feature_bank.shape[0]) - 1))
    d = torch.cdist(z, pack.feature_bank.float(), p=2)
    knn = torch.topk(d, k=k, largest=False).values.clamp_min(1e-12)
    rk = knn[:, -1:].clamp_min(1e-12)
    logs = torch.log(knn / rk).clamp_max(-1e-12)
    lid = -float(k) / logs.sum(dim=1).clamp_max(-1e-12)
    return lid


@torch.no_grad()
def score_all_detectors(fusion_model, pack: DetectorPack, xa: torch.Tensor, xv: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "msp_hendrycks_gimpel": msp_score(fusion_model, xa, xv),
        "feature_squeezing_xu": feature_squeezing_score(fusion_model, xa, xv),
        "mahalanobis_lee": mahalanobis_score(fusion_model, pack, xa, xv),
        "lid_ma": lid_score(fusion_model, pack, xa, xv),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.x removed/deprecated np.trapz in some builds; keep this self-contained."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _auroc(clean_scores: np.ndarray, adv_scores: np.ndarray) -> float:
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    adv_scores = np.asarray(adv_scores, dtype=np.float64)
    if clean_scores.size == 0 or adv_scores.size == 0:
        return 0.0
    scores = np.concatenate([clean_scores, adv_scores])
    labels = np.concatenate([np.zeros_like(clean_scores), np.ones_like(adv_scores)])
    order = np.argsort(-scores)
    labels = labels[order]
    pos = max(1.0, float(labels.sum()))
    neg = max(1.0, float((1 - labels).sum()))
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tps / pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / neg, [1.0]])
    return _trapz_compat(tpr, fpr)


def _aupr(clean_scores: np.ndarray, adv_scores: np.ndarray) -> float:
    scores = np.concatenate([clean_scores, adv_scores])
    labels = np.concatenate([np.zeros_like(clean_scores), np.ones_like(adv_scores)])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(labels.sum()), 1.0)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def detection_metrics(clean_test: np.ndarray, adv: np.ndarray, clean_val: np.ndarray, target_fpr: float = 0.05) -> dict[str, float]:
    clean_test = np.asarray(clean_test, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    clean_val = np.asarray(clean_val, dtype=np.float64)
    threshold = float(np.quantile(clean_val, 1.0 - target_fpr))
    return {
        "AUROC": _auroc(clean_test, adv),
        "AUPR_adv": _aupr(clean_test, adv),
        "threshold_from_clean_val": threshold,
        "target_clean_FPR": float(target_fpr),
        "clean_test_FPR_at_threshold": float(np.mean(clean_test >= threshold)),
        "attack_detection_rate_at_threshold": float(np.mean(adv >= threshold)),
        "clean_score_mean": float(clean_test.mean()) if clean_test.size else 0.0,
        "adv_score_mean": float(adv.mean()) if adv.size else 0.0,
    }


@torch.no_grad()
def collect_detector_scores(fusion_model, pack: DetectorPack, xa_all: torch.Tensor, xv_all: torch.Tensor, cfg: Config, device: torch.device) -> dict[str, np.ndarray]:
    ds = TensorDataset(xa_all.float(), xv_all.float())
    loader = DataLoader(ds, batch_size=int(getattr(cfg, "detector_batch_size", cfg.batch_size)), shuffle=False, num_workers=0)
    out: dict[str, list[np.ndarray]] = {}
    fusion_model.eval()
    for xa, xv in loader:
        xa = xa.to(device)
        xv = xv.to(device)
        scores = score_all_detectors(fusion_model, pack, xa, xv)
        for name, value in scores.items():
            out.setdefault(name, []).append(value.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}
