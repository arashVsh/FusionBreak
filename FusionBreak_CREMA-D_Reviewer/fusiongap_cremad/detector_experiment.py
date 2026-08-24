from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .attacks import _collect_clean_pool, _fusiongap, _standard_pgd
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .detectors import collect_detector_scores, detection_metrics, fit_detector_pack
from .models import AudioNet, FusionNet, VideoNet
from .utils import get_device, save_json, set_seed


DETECTOR_ARCHITECTURES = ["interaction", "concat", "gated", "group_gated"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    return (
        cfg.checkpoint_dir / f"audio_seed{seed}.pt",
        cfg.checkpoint_dir / f"video_seed{seed}.pt",
        cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt",
    )


def _load_checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load either a raw state_dict or a wrapped checkpoint.

    Some project versions saved checkpoints directly as model.state_dict().
    Other versions saved {"state_dict": ..., "metrics": ...}.
    This helper supports both formats and also strips a DataParallel
    ``module.`` prefix if present.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}: {type(ckpt)}")

    # If the checkpoint is wrapped in another common key, unwrap it.
    for key in ("model", "model_state_dict", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break

    # Strip torch.nn.DataParallel prefix when needed.
    if any(isinstance(k, str) and k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def _load_models(seed: int, arch: str, cfg: Config, device: torch.device):
    audio_ckpt, video_ckpt, fusion_ckpt = _checkpoint_paths(cfg, seed, arch)
    missing = [str(p) for p in [audio_ckpt, video_ckpt, fusion_ckpt] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s). Run the main CREMA-D experiments first. Missing: " + "; ".join(missing)
        )
    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio.load_state_dict(_load_checkpoint_state_dict(audio_ckpt, device))
    video.load_state_dict(_load_checkpoint_state_dict(video_ckpt, device))

    fusion = FusionNet(AudioNet(num_classes=cfg.num_classes), VideoNet(num_classes=cfg.num_classes), arch, num_classes=cfg.num_classes).to(device)
    fusion.load_state_dict(_load_checkpoint_state_dict(fusion_ckpt, device))
    audio.eval(); video.eval(); fusion.eval()
    return audio, video, fusion


@torch.no_grad()
def _cache_inputs(cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return cache["audio"].float(), cache["video"].float()


def _generate_attack_scores(audio, video, fusion, clean_pool, pack, cfg: Config, device: torch.device):
    xa_all, xv_all, y_all = clean_pool
    ds = TensorDataset(xa_all.float(), xv_all.float(), y_all.long())
    loader = DataLoader(ds, batch_size=cfg.attack_batch_size, shuffle=False, num_workers=0)

    clean_scores_parts: dict[str, list] = {}
    fg_scores_parts: dict[str, list] = {}
    pgd_scores_parts: dict[str, list] = {}

    for xa, xv, y in tqdm(loader, desc="Detector attacks", unit="batch"):
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)
        xa_fg, xv_fg = _fusiongap(audio, video, fusion, xa, xv, y, cfg, device)
        xa_pgd, xv_pgd = _standard_pgd(audio, video, fusion, xa, xv, y, cfg, device)

        for target, pair in [(clean_scores_parts, (xa, xv)), (fg_scores_parts, (xa_fg, xv_fg)), (pgd_scores_parts, (xa_pgd, xv_pgd))]:
            scores = collect_detector_scores(fusion, pack, pair[0].detach().cpu(), pair[1].detach().cpu(), cfg, device)
            for name, arr in scores.items():
                target.setdefault(name, []).append(arr)

    import numpy as np
    join = lambda d: {k: np.concatenate(v, axis=0) for k, v in d.items()}
    return join(clean_scores_parts), join(fg_scores_parts), join(pgd_scores_parts)


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"detectors_cremad_seed{seed}_cremad_{arch}.json"


def run_one_detector(seed: int, arch: str, cfg: Config) -> dict[str, Any]:
    cfg.ensure_dirs()
    out_json = _result_path(cfg, seed, arch)
    if getattr(cfg, "skip_completed_detector_results", True) and out_json.exists():
        print(f"[skip] Detector result exists: {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"CREMA-D detector evaluation: seed={seed}, fusion_architecture={arch}")
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
        result = {"dataset": "CREMA-D", "seed": seed, "fusion_architecture": arch, "message": "No all-three-correct clean samples."}
        save_json(out_json, result)
        return result

    xa_clean, xv_clean, y_clean = clean_pool
    print(f"[detectors] Evaluating detectors on {int(y_clean.numel())} all-three-correct clean samples.")

    val_xa, val_xv = _cache_inputs(val_cache)
    val_clean_scores = collect_detector_scores(fusion, pack, val_xa, val_xv, cfg, device)
    clean_scores, fg_scores, pgd_scores = _generate_attack_scores(audio, video, fusion, clean_pool, pack, cfg, device)

    target_fpr = float(getattr(cfg, "detector_target_fpr", 0.05))
    metrics: dict[str, Any] = {"FusionGap": {}, "PGD": {}}
    for detector_name in sorted(clean_scores.keys()):
        metrics["FusionGap"][detector_name] = detection_metrics(
            clean_scores[detector_name], fg_scores[detector_name], val_clean_scores[detector_name], target_fpr
        )
        metrics["PGD"][detector_name] = detection_metrics(
            clean_scores[detector_name], pgd_scores[detector_name], val_clean_scores[detector_name], target_fpr
        )

    result = {
        "dataset": "CREMA-D",
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "attacked_clean_samples": int(y_clean.numel()),
        "detector_reference": {
            "msp_hendrycks_gimpel": "maximum softmax probability baseline",
            "feature_squeezing_xu": "feature squeezing discrepancy",
            "mahalanobis_lee": "class conditional Gaussian Mahalanobis distance on fusion features",
            "lid_ma": "local intrinsic dimensionality on fusion features",
        },
        "attack_settings": {
            "attack_steps": cfg.attack_steps,
            "random_restarts": cfg.random_restarts,
            "attack_samples": cfg.attack_samples,
            "eps_audio": cfg.eps_audio,
            "eps_video": cfg.eps_video,
        },
        "detector_settings": {
            "target_clean_FPR": target_fpr,
            "detector_fit_samples": int(getattr(cfg, "detector_fit_samples", 2500)),
            "detector_lid_k": int(getattr(cfg, "detector_lid_k", 20)),
            "detector_mahalanobis_reg": float(getattr(cfg, "detector_mahalanobis_reg", 1e-3)),
        },
        "detector_metrics": metrics,
    }
    save_json(out_json, result)
    print(f"[done] Saved detector result: {out_json}")
    return result


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attack_name, dets in result.get("detector_metrics", {}).items():
        for detector_name, m in dets.items():
            rows.append({
                "dataset": result.get("dataset", "CREMA-D"),
                "seed": result.get("seed", ""),
                "fusion_architecture": result.get("fusion_architecture", ""),
                "attacked_clean_samples": result.get("attacked_clean_samples", ""),
                "attack": attack_name,
                "detector": detector_name,
                "AUROC": m.get("AUROC", ""),
                "AUPR_adv": m.get("AUPR_adv", ""),
                "detect_rate_at_5pct_clean_val_FPR": m.get("attack_detection_rate_at_threshold", ""),
                "clean_test_FPR_at_threshold": m.get("clean_test_FPR_at_threshold", ""),
                "clean_score_mean": m.get("clean_score_mean", ""),
                "adv_score_mean": m.get("adv_score_mean", ""),
            })
    return rows


def write_detector_summary(cfg: Config) -> None:
    rows = []
    for p in sorted(cfg.log_dir.glob("detectors_cremad_seed*_cremad_*.json")):
        try:
            rows.extend(_summary_rows(json.loads(p.read_text(encoding="utf-8"))))
        except Exception as e:
            print(f"[summary] Skipped {p}: {e}")
    if not rows:
        return
    _write_csv(cfg.artifact_dir / "cremad_detector_summary.csv", rows)
    save_json(cfg.artifact_dir / "cremad_detector_summary.json", {"rows": rows})
    print(f"[summary] Saved: {cfg.artifact_dir / 'cremad_detector_summary.csv'}")


def run_all_detectors(cfg: Config) -> None:
    cfg.ensure_dirs()
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
    if not hasattr(cfg, "skip_completed_detector_results"):
        cfg.skip_completed_detector_results = True

    print("Project root:", cfg.project_root)
    print("Data root:", cfg.data_root())
    print("Detector logs:", cfg.log_dir)
    print("Detector artifacts:", cfg.artifact_dir)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    plan = [(seed, arch) for seed in cfg.seeds for arch in DETECTOR_ARCHITECTURES]
    for seed, arch in tqdm(plan, desc="CREMA-D detector evaluations", unit="run"):
        run_one_detector(seed, arch, cfg)
        write_detector_summary(cfg)
    write_detector_summary(cfg)
    print("All requested CREMA-D detector evaluations completed.")


def main() -> None:
    cfg = Config(project_root=Path(__file__).resolve().parents[1])
    run_all_detectors(cfg)
