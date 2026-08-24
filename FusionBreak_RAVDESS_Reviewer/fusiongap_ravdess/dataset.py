from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

try:
    import torchaudio
except ImportError as exc:
    raise ImportError("torchaudio is required. Install CUDA PyTorch, torchvision, and torchaudio first.") from exc

try:
    import torchvision
except ImportError as exc:
    raise ImportError("torchvision is required. Install CUDA PyTorch, torchvision, and torchaudio first.") from exc


EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


@dataclass(frozen=True)
class RavdessRecord:
    audio_path: Path
    video_path: Path
    label: int
    actor_id: int
    filename_stem: str


def parse_ravdess_filename(path: Path) -> Dict[str, str]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {path.name}")
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def scan_multimodal_records(data_root: Path) -> List[RavdessRecord]:
    audio_root = data_root / "audio"
    video_root = data_root / "video"

    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio folder: {audio_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"Missing video folder: {video_root}")

    audio_files: Dict[str, Path] = {}
    for wav_path in audio_root.rglob("*.wav"):
        # RAVDESS speech audio files start with 03-01-...
        if not wav_path.name.startswith("03-01-"):
            continue
        matched_video_stem = wav_path.stem.replace("03-", "01-", 1)
        audio_files[matched_video_stem] = wav_path

    records: List[RavdessRecord] = []
    missing_audio = 0

    for video_path in video_root.rglob("*.mp4"):
        # Use full audio video speech files only.
        # 01 means audio video, 02 means video only, 03 means audio only.
        if not video_path.name.startswith("01-01-"):
            continue

        meta = parse_ravdess_filename(video_path)
        audio_path = audio_files.get(video_path.stem)
        if audio_path is None:
            missing_audio += 1
            continue

        emotion_code = meta["emotion"]
        if emotion_code not in EMOTION_MAP:
            continue

        records.append(
            RavdessRecord(
                audio_path=audio_path,
                video_path=video_path,
                label=EMOTION_MAP[emotion_code],
                actor_id=int(meta["actor"]),
                filename_stem=video_path.stem,
            )
        )

    records = sorted(records, key=lambda r: r.filename_stem)

    if not records:
        raise RuntimeError(
            f"No matched multimodal records found under {data_root}.\n"
            "Expected files like:\n"
            "  RAVDESS/audio/Actor_01/03-01-03-01-01-01-01.wav\n"
            "  RAVDESS/video/Actor_01/01-01-03-01-01-01-01.mp4\n"
        )

    print(f"[scan] matched multimodal records: {len(records)}")
    if missing_audio:
        print(f"[scan] video files skipped because matching audio was missing: {missing_audio}")

    return records


def split_records(
    records: List[RavdessRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[RavdessRecord], List[RavdessRecord], List[RavdessRecord]]:
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    return train, val, test


class RavdessMultimodalDataset(Dataset):
    def __init__(
        self,
        records: List[RavdessRecord],
        sample_rate: int = 16000,
        audio_seconds: float = 3.0,
        n_mels: int = 64,
        num_frames: int = 16,
        image_size: int = 112,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.audio_num_samples = int(sample_rate * audio_seconds)
        self.num_frames = num_frames
        self.image_size = image_size

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.shape[1] < self.audio_num_samples:
            pad = self.audio_num_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, : self.audio_num_samples]

        spec = self.to_db(self.mel(wav))
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec

    def _load_video(self, path: Path) -> torch.Tensor:
        video, _, _ = torchvision.io.read_video(
            str(path),
            pts_unit="sec",
            output_format="TCHW",
        )

        if video.numel() == 0:
            raise RuntimeError(f"Could not read video: {path}")

        video = video.float() / 255.0

        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            pad_count = self.num_frames - total_frames
            pad = video[-1:].repeat(pad_count, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)

        video = torch.nn.functional.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Output shape: C, T, H, W for 3D CNN.
        video = video.permute(1, 0, 2, 3).contiguous()
        return video

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return {
            "audio": self._load_audio(record.audio_path),
            "video": self._load_video(record.video_path),
            "label": torch.tensor(record.label, dtype=torch.long),
            "actor_id": record.actor_id,
            "stem": record.filename_stem,
        }
