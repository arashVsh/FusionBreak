from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .attacks import run_attacks
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .models import AudioNet, VideoNet, FusionNet
from .train import clean_evaluation, train_fusion, train_unimodal
from .utils import get_device, result_path, save_json, set_seed
import copy

def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    a = cfg.checkpoint_dir / f"audio_seed{seed}.pt"
    v = cfg.checkpoint_dir / f"video_seed{seed}.pt"
    f = cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt"
    return a, v, f


def run_one(seed: int, arch: str, cfg: Config) -> dict[str, object]:
    cfg.ensure_dirs()
    out_json = result_path(cfg.log_dir, "cremad", seed, arch)
    if cfg.skip_completed_results and out_json.exists():
        print(f"[skip] Completed result exists: {out_json}")
        return {"skipped": True, "path": str(out_json)}

    print("\n" + "=" * 100)
    print(f"CREMA-D experiment: seed={seed}, fusion_architecture={arch}")
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

    # IMPORTANT:
    # Do not pass the standalone audio/video objects into FusionNet.
    # Fusion training would update them and corrupt unimodal evaluation.
    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)

    fusion = FusionNet(fusion_audio, fusion_video, arch, num_classes=cfg.num_classes).to(device)
    fusion_metrics = train_fusion(fusion, train_cache, val_cache, cfg, device, fusion_ckpt)

    # Make sure all models are in eval mode before clean evaluation.
    audio.eval()
    video.eval()
    fusion.eval()

    clean = clean_evaluation(audio, video, fusion, test_cache, cfg, device)
    print("Clean evaluation:", clean)

    result: dict[str, object] = {
        "dataset": "CREMA-D",
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "checkpoints": {"audio": str(audio_ckpt), "video": str(video_ckpt), "fusion": str(fusion_ckpt)},
        "training": {"audio": audio_metrics, "video": video_metrics, "fusion": fusion_metrics},
        "clean_eval": clean,
        "attack_settings": {
            "attack_steps": cfg.attack_steps,
            "random_restarts": cfg.random_restarts,
            "attack_samples": cfg.attack_samples,
            "eps_audio": cfg.eps_audio,
            "eps_video": cfg.eps_video,
        },
    }

    gate_ok = clean["fusion_acc"] >= cfg.min_fusion_acc_for_attack and clean["all_three_correct_rate"] >= cfg.min_all_three_rate_for_attack
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


def run_all(cfg: Config) -> None:
    cfg.ensure_dirs()
    print("Project root:", cfg.project_root)
    print("Data root:", cfg.data_root())
    print("Logs:", cfg.log_dir)
    print("Large files:", cfg.large_dir)
    plan = [(seed, arch) for seed in cfg.seeds for arch in cfg.fusion_architectures]
    for seed, arch in tqdm(plan, desc="CREMA-D experiments", unit="experiment"):
        run_one(seed, arch, cfg)
    print("All requested experiments completed.")
