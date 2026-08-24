from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Local project paths. The entry script is expected to sit in the parent project folder.
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir_names: tuple[str, ...] = ("CREMA-D", "Camera-D", "CREMAD")

    output_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    large_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)

    # Reproducibility and experiment plan.
    # seeds: tuple[int, ...] = (42, 123, 2026)
    seeds: tuple[int, ...] = (42, 43, 44)
    fusion_architectures: tuple[str, ...] = ("interaction", "concat", "gated", "group_gated")
    skip_completed_results: bool = True

    # Data processing.
    sample_rate: int = 16000
    audio_seconds: float = 3.0
    audio_bins: int = 64
    audio_frames: int = 128
    video_frames: int = 12
    image_size: int = 64
    train_fraction: float = 0.70
    val_fraction: float = 0.10
    test_fraction: float = 0.20

    # Training.
    num_classes: int = 6
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs_unimodal: int = 40
    epochs_fusion: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    min_epochs_before_stop: int = 8
    amp: bool = True
    num_workers: int = 0

    # Attack settings: kept consistent with the previous main project settings.
    attack_samples: int = 500
    attack_steps: int = 80
    random_restarts: int = 10
    eps_audio: float = 0.08
    eps_video: float = 0.06
    attack_batch_size: int = 16
    preservation_weight: float = 1.0

    # Quality gate to avoid spending time on attacks when clean classifiers are too weak.
    min_fusion_acc_for_attack: float = 0.45
    min_all_three_rate_for_attack: float = 0.10
    stop_if_quality_gate_fails: bool = True

    def __post_init__(self) -> None:
        self.output_dir = self.project_root / "fusion_gap_cremad_modular_outputs"
        self.log_dir = self.output_dir / "logs"
        self.large_dir = self.output_dir / "large_files"
        self.checkpoint_dir = self.large_dir / "checkpoints"
        self.cache_dir = self.large_dir / "packed_tensor_cache"
        self.artifact_dir = self.large_dir / "artifacts"

    def ensure_dirs(self) -> None:
        for path in [self.output_dir, self.log_dir, self.large_dir, self.checkpoint_dir, self.cache_dir, self.artifact_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def data_root(self) -> Path:
        for name in self.data_dir_names:
            candidate = self.project_root / name
            if candidate.exists():
                return candidate
        names = ", ".join(self.data_dir_names)
        raise FileNotFoundError(f"Could not find the dataset folder. Expected one of: {names} inside {self.project_root}")
