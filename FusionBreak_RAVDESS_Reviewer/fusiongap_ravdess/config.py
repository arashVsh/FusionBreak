from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    project_root: Path

    # RAVDESS multimodal folder layout expected by this project:
    # RAVDESS/Audio/Actor_01/*.wav ... Actor_24/*.wav
    # RAVDESS/Video/Actor_01/*.mp4 ... Actor_24/*.mp4
    ravdess_folder_name: str = "RAVDESS"
    audio_subfolder_name: str = "Audio"
    video_subfolder_name: str = "Video"

    # Output/cache/checkpoint folders.
    output_dir_name: str = "fusion_gap_ravdess_multimodal_outputs"
    run_version: str = "v2_crop_aug_freeze"
    attack_version: str = "v3_decision_preserving_attack"

    # Full experiment plan.
    # seeds: list[int] = field(default_factory=lambda: [42, 123, 2026])
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 123, 2026])
    fusion_architectures: list[str] = field(default_factory=lambda: [
        "interaction", "concat", "gated", "group_gated"
    ])

    # Dataset/split settings. Same ratio as the CREMA-D project.
    num_classes: int = 8
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing settings.
    sample_rate: int = 16000
    audio_duration_sec: float = 3.0
    audio_spec_size: int = 64
    video_frames: int = 12
    video_size: int = 64
    video_center_crop: bool = True
    use_cuda_for_audio_preprocess: bool = True

    # Training settings.
    batch_size: int = 64
    epochs_unimodal: int = 40
    epochs_fusion: int = 35
    early_stopping_patience: int = 8
    min_epochs_before_early_stop: int = 8
    lr: float = 7e-4
    fusion_lr: float = 3e-4
    weight_decay: float = 1e-4
    freeze_fusion_backbones: bool = True
    use_train_augmentation: bool = True
    audio_noise_std: float = 0.015
    audio_mask_max_frac: float = 0.12
    video_noise_std: float = 0.015
    video_brightness_jitter: float = 0.10
    video_hflip_prob: float = 0.50
    num_workers: int = 0
    amp: bool = True

    # Consistent attack settings matching the CREMA-D project.
    attack_samples: int = 500
    attack_batch_size: int = 32
    attack_steps: int = 80
    random_restarts: int = 10
    eps_audio: float = 0.08
    eps_video: float = 0.06
    alpha_audio: float = 0.004
    alpha_video: float = 0.003
    lambda_unimodal_ce: float = 0.10
    # v3 attack objective: keep unimodal decisions correct while forcing fusion failure.
    # These do not change eps/steps/restarts; they change the objective and candidate selection.
    lambda_fusion_margin: float = 0.50
    lambda_preserve_margin: float = 5.00
    lambda_logit_distill: float = 0.25
    preserve_margin: float = 0.05
    candidate_eval_interval: int = 1
    clean_start_first_restart: bool = True

    # Quality gate. Attacks run only if enough clean all-three-correct samples exist.
    min_fusion_acc_for_attack: float = 0.50
    min_all_three_rate_for_attack: float = 0.05
    stop_if_quality_gate_fails: bool = False

    # Reuse previous work.
    skip_completed_results: bool = True
    load_checkpoints_if_available: bool = True
    save_checkpoints: bool = True
    use_packed_tensor_cache: bool = True

    @classmethod
    def create_default(cls, project_root: Path | None = None) -> "Config":
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        return cls(project_root=Path(project_root).resolve())

    @property
    def data_root(self) -> Path:
        return self.project_root / self.ravdess_folder_name

    @property
    def audio_root(self) -> Path:
        return self.data_root / self.audio_subfolder_name

    @property
    def video_root(self) -> Path:
        return self.data_root / self.video_subfolder_name

    @property
    def output_root(self) -> Path:
        return self.project_root / self.output_dir_name

    @property
    def log_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def large_dir(self) -> Path:
        return self.output_root / "large_files"

    @property
    def checkpoint_dir(self) -> Path:
        return self.large_dir / "checkpoints"

    @property
    def packed_cache_dir(self) -> Path:
        return self.large_dir / "packed_tensor_cache"

    @property
    def artifact_dir(self) -> Path:
        return self.large_dir / "artifacts"

    def ensure_dirs(self) -> None:
        for path in [self.output_root, self.log_dir, self.large_dir, self.checkpoint_dir, self.packed_cache_dir, self.artifact_dir]:
            path.mkdir(parents=True, exist_ok=True)
