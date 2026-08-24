from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .attacks import run_attacks
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .models import AudioNet, FusionNet, VideoNet
from .train import clean_evaluation, train_fusion, train_unimodal
from .utils import get_device, result_path, save_json, set_seed, write_csv


def cleanup_incompatible_previous_outputs(cfg: Config) -> None:
    """Remove result files produced by the previous RAVDESS attack code once.

    The v3 change is attack-only. We keep decoded tensor caches and trained
    checkpoints because preprocessing/training did not change. We delete JSON
    result files and summaries so the project reruns attacks with the fixed
    decision-preserving objective instead of skipping completed old results.
    """
    cfg.ensure_dirs()
    marker = cfg.output_root / f".{cfg.attack_version}_cleanup_done"
    if marker.exists():
        return

    patterns = [
        cfg.log_dir / "fusion_gap_ravdess_seed*_ravdess_*.json",
        cfg.artifact_dir / "ravdess_results_summary.csv",
        cfg.artifact_dir / "ravdess_results_summary.json",
    ]
    deleted = 0
    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
    marker.write_text(
        f"cleanup completed for {cfg.attack_version}; deleted={deleted}; kept checkpoints and tensor caches\n",
        encoding="utf-8",
    )
    print(f"[cleanup] Removed {deleted} old result/summary files. Checkpoints and tensor caches were kept.")


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    audio_ckpt = cfg.checkpoint_dir / f"audio_seed{seed}.pt"
    video_ckpt = cfg.checkpoint_dir / f"video_seed{seed}.pt"
    fusion_ckpt = cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt"
    return audio_ckpt, video_ckpt, fusion_ckpt


def run_one(seed: int, arch: str, cfg: Config) -> dict[str, Any]:
    cfg.ensure_dirs()
    out_json = result_path(cfg.log_dir, seed, arch)
    if cfg.skip_completed_results and out_json.exists():
        print(f"[skip] Completed result exists: {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"RAVDESS multimodal experiment: seed={seed}, fusion_architecture={arch}")
    print("=" * 100)

    set_seed(seed)
    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is not available. This will be slow.")

    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", seed, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", seed, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", seed, cfg, device)

    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio_ckpt, video_ckpt, fusion_ckpt = _checkpoint_paths(cfg, seed, arch)

    audio_metrics = train_unimodal(audio, train_cache, val_cache, "audio", cfg, device, audio_ckpt)
    video_metrics = train_unimodal(video, train_cache, val_cache, "video", cfg, device, video_ckpt)

    # Important: do not pass the standalone audio/video objects into FusionNet.
    # Fusion training updates its internal backbones, so the standalone unimodal
    # models must stay separate for clean evaluation and preservation metrics.
    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)
    fusion = FusionNet(fusion_audio, fusion_video, arch, num_classes=cfg.num_classes).to(device)

    fusion_metrics = train_fusion(fusion, train_cache, val_cache, cfg, device, fusion_ckpt)

    audio.eval()
    video.eval()
    fusion.eval()
    clean = clean_evaluation(audio, video, fusion, test_cache, cfg, device)
    print("Clean evaluation:", clean)

    result: dict[str, Any] = {
        "dataset": "RAVDESS",
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "checkpoints": {
            "audio": str(audio_ckpt),
            "video": str(video_ckpt),
            "fusion": str(fusion_ckpt),
        },
        "training": {
            "audio": audio_metrics,
            "video": video_metrics,
            "fusion": fusion_metrics,
        },
        "clean_eval": clean,
        "attack_settings": {
            "attack_steps": cfg.attack_steps,
            "random_restarts": cfg.random_restarts,
            "attack_samples": cfg.attack_samples,
            "eps_audio": cfg.eps_audio,
            "eps_video": cfg.eps_video,
            "alpha_audio": cfg.alpha_audio,
            "alpha_video": cfg.alpha_video,
            "lambda_unimodal_ce": cfg.lambda_unimodal_ce,
            "lambda_fusion_margin": cfg.lambda_fusion_margin,
            "lambda_preserve_margin": cfg.lambda_preserve_margin,
            "lambda_logit_distill": cfg.lambda_logit_distill,
            "preserve_margin": cfg.preserve_margin,
            "candidate_eval_interval": cfg.candidate_eval_interval,
            "clean_start_first_restart": cfg.clean_start_first_restart,
            "attack_version": cfg.attack_version,
        },
    }

    gate_ok = (
        clean["fusion_acc"] >= cfg.min_fusion_acc_for_attack
        and clean["all_three_correct_rate"] >= cfg.min_all_three_rate_for_attack
    )
    if not gate_ok:
        msg = (
            f"Quality gate failed: fusion_acc={clean['fusion_acc']:.4f}, "
            f"all_three_correct_rate={clean['all_three_correct_rate']:.4f}. Attacks were skipped."
        )
        print("WARNING:", msg)
        result["quality_gate"] = {"passed": False, "message": msg}
        save_json(out_json, result)
        if cfg.stop_if_quality_gate_fails:
            return result
    else:
        result["quality_gate"] = {"passed": True}
        result["attack_results"] = run_attacks(audio, video, fusion, test_cache, cfg, device)
        save_json(out_json, result)

    print(f"[done] Saved result: {out_json}")
    return result


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    clean = result.get("clean_eval", {}) or {}
    attack = result.get("attack_results", {}) or {}
    fg = attack.get("fusion_gap_rates", {}) if isinstance(attack, dict) else {}
    pgd = attack.get("standard_pgd_rates", {}) if isinstance(attack, dict) else {}
    return {
        "dataset": result.get("dataset", "RAVDESS"),
        "seed": result.get("seed", ""),
        "fusion_architecture": result.get("fusion_architecture", ""),
        "train_samples": result.get("train_samples", ""),
        "val_samples": result.get("val_samples", ""),
        "test_samples": result.get("test_samples", ""),
        "audio_acc": clean.get("audio_acc", ""),
        "video_acc": clean.get("video_acc", ""),
        "fusion_acc": clean.get("fusion_acc", ""),
        "all_three_correct_rate": clean.get("all_three_correct_rate", ""),
        "all_three_correct_count": clean.get("all_three_correct_count", ""),
        "attacked_clean_samples": attack.get("attacked_clean_samples", "") if isinstance(attack, dict) else "",
        "fusiongap_fusion_ASR": fg.get("fusion_ASR", ""),
        "fusiongap_audio_preservation": fg.get("audio_preservation_rate", ""),
        "fusiongap_video_preservation": fg.get("video_preservation_rate", ""),
        "fusiongap_FOSR": fg.get("FOSR", ""),
        "pgd_fusion_ASR": pgd.get("fusion_ASR", ""),
        "pgd_audio_preservation": pgd.get("audio_preservation_rate", ""),
        "pgd_video_preservation": pgd.get("video_preservation_rate", ""),
        "pgd_FOSR": pgd.get("FOSR", ""),
    }


def write_summary(cfg: Config) -> None:
    rows = []
    for p in sorted(cfg.log_dir.glob("fusion_gap_ravdess_seed*_ravdess_*.json")):
        try:
            result = json.loads(p.read_text(encoding="utf-8"))
            rows.append(_summary_row(result))
        except Exception as e:
            print(f"[summary] Skipped {p}: {e}")
    if not rows:
        return
    write_csv(cfg.artifact_dir / "ravdess_results_summary.csv", rows)
    save_json(cfg.artifact_dir / "ravdess_results_summary.json", {"rows": rows})
    print(f"[summary] Saved: {cfg.artifact_dir / 'ravdess_results_summary.csv'}")


def run_all(cfg: Config) -> None:
    cfg.ensure_dirs()
    print("Project root:", cfg.project_root)
    print("Data root:", cfg.data_root)
    print("Audio root:", cfg.audio_root)
    print("Video root:", cfg.video_root)
    print("Logs:", cfg.log_dir)
    print("Large files:", cfg.large_dir)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    plan = [(seed, arch) for seed in cfg.seeds for arch in cfg.fusion_architectures]
    for seed, arch in tqdm(plan, desc="RAVDESS experiments", unit="experiment"):
        run_one(seed, arch, cfg)
        write_summary(cfg)
    print("All requested RAVDESS experiments completed.")
    write_summary(cfg)


def main() -> None:
    cfg = Config.create_default(Path(__file__).resolve().parents[1])
    cleanup_incompatible_previous_outputs(cfg)
    run_all(cfg)
