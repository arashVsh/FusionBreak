from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config


@dataclass(frozen=True)
class Record:
    key: str
    audio_path: str
    video_path: str
    label: int
    emotion_code: str
    actor: str


EMOTION_TO_LABEL = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

LABEL_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def _parse_ravdess_stem(path: Path) -> dict[str, str] | None:
    # Filename format:
    # modality-vocal_channel-emotion-intensity-statement-repetition-actor
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    if vocal_channel != "01":  # speech only
        return None
    if emotion not in EMOTION_TO_LABEL:
        return None
    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "pair_key": "-".join(parts[1:]),
    }


def _records_cache_path(cfg: Config) -> Path:
    return cfg.packed_cache_dir / "ravdess_multimodal_record_cache.json"


def scan_records(cfg: Config) -> list[Record]:
    cfg.ensure_dirs()
    cache_path = _records_cache_path(cfg)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        records = [Record(**item) for item in data]
        print(f"[scan] Loaded {len(records)} cached multimodal records from {cache_path}")
        return records

    if not cfg.audio_root.exists():
        raise FileNotFoundError(f"Audio folder not found: {cfg.audio_root}")
    if not cfg.video_root.exists():
        raise FileNotFoundError(f"Video folder not found: {cfg.video_root}")

    audio_by_key: dict[str, Path] = {}
    video_by_key: dict[str, Path] = {}

    audio_files = sorted(cfg.audio_root.rglob("*.wav"))
    video_files = sorted(list(cfg.video_root.rglob("*.mp4")) + list(cfg.video_root.rglob("*.avi")) + list(cfg.video_root.rglob("*.mov")))

    for p in audio_files:
        meta = _parse_ravdess_stem(p)
        if meta is None:
            continue
        # Audio-only speech files start with 03.
        if meta["modality"] == "03":
            audio_by_key[meta["pair_key"]] = p

    for p in video_files:
        meta = _parse_ravdess_stem(p)
        if meta is None:
            continue
        # Full audio-video speech files start with 01. Ignore video-only 02 files.
        if meta["modality"] == "01":
            video_by_key[meta["pair_key"]] = p

    records: list[Record] = []
    for key in sorted(set(audio_by_key) & set(video_by_key)):
        parts = key.split("-")
        emotion = parts[1]
        actor = parts[-1]
        records.append(Record(
            key=key,
            audio_path=str(audio_by_key[key]),
            video_path=str(video_by_key[key]),
            label=EMOTION_TO_LABEL[emotion],
            emotion_code=emotion,
            actor=actor,
        ))

    if len(records) == 0:
        raise RuntimeError(
            "No paired RAVDESS multimodal records found. Expected files like "
            "RAVDESS/Audio/Actor_01/03-01-03-01-01-01-01.wav and "
            "RAVDESS/Video/Actor_01/01-01-03-01-01-01-01.mp4."
        )

    cache_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    print(f"[scan] matched multimodal records: {len(records)}")
    print(f"[scan] Saved record cache: {cache_path}")
    return records


def split_records(records: list[Record], seed: int, cfg: Config) -> tuple[list[Record], list[Record], list[Record]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = int(round(cfg.train_ratio * n))
    n_val = int(round(cfg.val_ratio * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    train = [records[int(i)] for i in train_idx]
    val = [records[int(i)] for i in val_idx]
    test = [records[int(i)] for i in test_idx]
    print(f"[split] seed={seed} train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    # Avoid torchaudio/torchcodec/FFmpeg issues on Windows.
    try:
        from scipy.io import wavfile
        sr, wav = wavfile.read(str(path))
        wav = np.asarray(wav)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            wav = np.frombuffer(frames, dtype=np.int16)
        elif sampwidth == 4:
            wav = np.frombuffer(frames, dtype=np.int32)
        else:
            wav = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        if channels > 1:
            wav = wav.reshape(-1, channels).mean(axis=1)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    max_abs = float(np.max(np.abs(wav))) if wav.size else 1.0
    if max_abs > 1.5:
        wav = wav / max_abs
    return wav, int(sr)


def _resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    if wav.size == 0:
        return wav.astype(np.float32)
    duration = wav.shape[0] / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    old_x = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, wav).astype(np.float32)


def _fix_audio_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    if wav.shape[0] >= target_len:
        start = max(0, (wav.shape[0] - target_len) // 2)
        return wav[start:start + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def _audio_to_spec(wav: np.ndarray, cfg: Config, device: torch.device) -> torch.Tensor:
    target_len = int(cfg.sample_rate * cfg.audio_duration_sec)
    wav = _fix_audio_length(wav, target_len)
    x = torch.from_numpy(wav).float().to(device)
    window = torch.hann_window(512, device=device)
    spec = torch.stft(x, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True)
    mag = spec.abs().clamp_min(1e-6).log()
    mag = mag.unsqueeze(0).unsqueeze(0)  # 1,1,F,T
    mag = F.interpolate(mag, size=(cfg.audio_spec_size, cfg.audio_spec_size), mode="bilinear", align_corners=False)
    mag = mag.squeeze(0)  # 1,H,W
    mag = (mag - mag.mean()) / (mag.std().clamp_min(1e-5))
    return mag.detach().cpu()


def load_audio_tensor(path: Path, cfg: Config, preprocess_device: torch.device) -> torch.Tensor:
    wav, sr = _read_wav(path)
    wav = _resample_linear(wav, sr, cfg.sample_rate)
    return _audio_to_spec(wav, cfg, preprocess_device)


def _center_crop_frame(frame: np.ndarray) -> np.ndarray:
    """Crop the central square before resizing.

    RAVDESS videos are wide frames. Resizing the whole frame directly makes the
    face small and horizontally distorted. A central square crop keeps the actor
    larger and improves the visual branch without needing any face detector.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    if side <= 0:
        return frame
    x0 = max(0, (w - side) // 2)
    # Slightly upper-centered crop: keeps face and upper torso.
    y0 = max(0, min(h - side, int((h - side) * 0.42)))
    return frame[y0:y0 + side, x0:x0 + side]


def load_video_tensor(path: Path, cfg: Config) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return torch.zeros(cfg.video_frames, 1, cfg.video_size, cfg.video_size, dtype=torch.float32)

    # Avoid the very first/last frames when possible; they often include neutral
    # posture or transition frames rather than the expressed emotion.
    start = int(0.10 * max(0, frame_count - 1))
    end = int(0.90 * max(0, frame_count - 1))
    if end <= start:
        start, end = 0, max(0, frame_count - 1)
    sample_positions = np.linspace(start, end, cfg.video_frames).astype(int)

    frames: list[np.ndarray] = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame_gray = np.zeros((cfg.video_size, cfg.video_size), dtype=np.float32)
        else:
            if cfg.video_center_crop:
                frame = _center_crop_frame(frame)
            frame = cv2.resize(frame, (cfg.video_size, cfg.video_size), interpolation=cv2.INTER_AREA)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(frame_gray)
    cap.release()
    arr = np.stack(frames, axis=0)[:, None, :, :]  # T,1,H,W
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).float()


def _cache_path(split_name: str, seed: int, cfg: Config) -> Path:
    # Include run_version because preprocessing changes invalidate tensor caches.
    return cfg.packed_cache_dir / f"ravdess_{cfg.run_version}_seed{seed}_{split_name}_tensors.pt"


def build_or_load_packed_cache(records: list[Record], split_name: str, seed: int, cfg: Config, device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    cfg.ensure_dirs()
    path = _cache_path(split_name, seed, cfg)
    if cfg.use_packed_tensor_cache and path.exists():
        print(f"[cache] Loading {split_name}: {path}")
        return torch.load(path, map_location="cpu")

    print(f"[cache] Building {split_name} cache with {len(records)} samples. This runs once per seed/split.")
    preprocess_device = device if (cfg.use_cuda_for_audio_preprocess and device.type == "cuda") else torch.device("cpu")

    audio_tensors = []
    video_tensors = []
    labels = []
    keys = []

    for r in tqdm(records, desc=f"Caching {split_name}", unit="sample"):
        audio_tensors.append(load_audio_tensor(Path(r.audio_path), cfg, preprocess_device))
        video_tensors.append(load_video_tensor(Path(r.video_path), cfg))
        labels.append(r.label)
        keys.append(r.key)

    cache = {
        "audio": torch.stack(audio_tensors, dim=0).contiguous(),
        "video": torch.stack(video_tensors, dim=0).contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "keys": keys,
    }
    torch.save(cache, path)
    print(f"[cache] Saved {split_name}: {path}")
    return cache
