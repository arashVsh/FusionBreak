from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import Config
from .utils import EMOTION_TO_LABEL, VIDEO_EXTS, load_json, save_json, atomic_torch_save


CACHE_FORMAT_VERSION = 2


@dataclass
class Record:
    video_path: str
    audio_path: str
    label: int
    emotion_id: str
    actor: int
    stem: str


def scan_records(cfg: Config) -> list[Record]:
    cfg.ensure_dirs()
    # Version two guarantees one record per utterance stem. Older caches could
    # silently include the same utterance twice when multiple video files shared
    # a stem (CREMA-D itself contains 7,442 unique utterances).
    cache_path = cfg.cache_dir / f"record_cache_v{CACHE_FORMAT_VERSION}.json"
    if cache_path.exists():
        raw = load_json(cache_path)
        items = raw.get("records", []) if isinstance(raw, dict) else raw
        records = [Record(**item) for item in items]
        unique_stems = {record.stem for record in records}
        paths_exist = all(
            Path(record.video_path).is_file() and Path(record.audio_path).is_file()
            for record in records
        )
        if len(unique_stems) == len(records) and paths_exist:
            print(f"[scan] Loaded {len(records)} cached records from {cache_path}")
            return records
        print(f"[scan] Ignoring invalid record cache: {cache_path}")

    root = cfg.data_root()
    print(f"[scan] Building record cache from {root}")
    all_files = [p for p in root.rglob("*") if p.is_file()]

    videos_by_stem: dict[str, list[Path]] = {}
    wavs_by_stem: dict[str, list[Path]] = {}
    for p in tqdm(sorted(all_files), desc="Indexing files", unit="file"):
        suffix = p.suffix.lower()
        if suffix in VIDEO_EXTS:
            videos_by_stem.setdefault(p.stem, []).append(p)
        elif suffix == ".wav":
            wavs_by_stem.setdefault(p.stem, []).append(p)

    def preferred_path(paths: list[Path], preferred_parent: str) -> Path:
        return sorted(
            paths,
            key=lambda path: (
                path.parent.name.lower() != preferred_parent,
                path.suffix.lower() != (".wav" if preferred_parent == "audiowav" else ".flv"),
                str(path).lower(),
            ),
        )[0]

    video_by_stem = {
        stem: preferred_path(paths, "videoflash")
        for stem, paths in videos_by_stem.items()
    }
    wav_by_stem = {
        stem: preferred_path(paths, "audiowav")
        for stem, paths in wavs_by_stem.items()
    }
    video_file_count = sum(len(paths) for paths in videos_by_stem.values())
    wav_file_count = sum(len(paths) for paths in wavs_by_stem.values())
    duplicate_video_files = video_file_count - len(video_by_stem)
    duplicate_wav_files = wav_file_count - len(wav_by_stem)

    records: list[Record] = []
    skipped_audio = 0
    skipped_name = 0
    for stem, vp in tqdm(
        sorted(video_by_stem.items()),
        desc="Matching unique video/WAV pairs",
        unit="utterance",
    ):
        parts = vp.stem.split("_")
        if len(parts) < 4 or parts[2] not in EMOTION_TO_LABEL:
            skipped_name += 1
            continue
        ap = wav_by_stem.get(stem)
        if ap is None:
            skipped_audio += 1
            continue
        try:
            actor = int(parts[0])
        except ValueError:
            actor = 0
        records.append(Record(str(vp), str(ap), EMOTION_TO_LABEL[parts[2]], parts[2], actor, vp.stem))

    print(
        f"[scan] video_files={video_file_count} unique_video_stems={len(video_by_stem)} "
        f"wav_files={wav_file_count} unique_wav_stems={len(wav_by_stem)} "
        f"duplicate_video_files={duplicate_video_files} duplicate_wav_files={duplicate_wav_files} "
        f"paired_records={len(records)} skipped_no_audio={skipped_audio} "
        f"skipped_bad_name={skipped_name}"
    )
    if len(records) < 1000:
        raise RuntimeError(f"Only {len(records)} paired records were found. Check that AudioWAV and VideoFlash are fully extracted.")

    save_json(
        cache_path,
        {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "data_root": str(root.resolve()),
            "records": [r.__dict__ for r in records],
        },
    )
    print(f"[scan] Saved record cache to {cache_path}")
    return records


def split_records(records: Sequence[Record], seed: int, cfg: Config) -> tuple[list[Record], list[Record], list[Record]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Record]] = {i: [] for i in range(cfg.num_classes)}
    for r in records:
        by_label[int(r.label)].append(r)

    train: list[Record] = []
    val: list[Record] = []
    test: list[Record] = []
    for label, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * cfg.train_fraction))
        n_val = int(round(n * cfg.val_fraction))
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    print(f"[split] seed={seed} train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def _read_wav(path: str, target_sr: int, seconds: float) -> np.ndarray:
    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        max_abs = max(float(np.max(np.abs(data))), 1.0)
        data = data / max_abs
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            data = resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        if sr != target_sr:
            raise RuntimeError("scipy is required for audio resampling. Install scipy or provide 16 kHz WAVs.")

    target_len = int(target_sr * seconds)
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]
    return data.astype(np.float32)


def audio_to_tensor(path: str, cfg: Config, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    wav = torch.from_numpy(_read_wav(path, cfg.sample_rate, cfg.audio_seconds)).to(device)
    window = torch.hann_window(400, device=device)
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=window, return_complex=True)
    mag = torch.log1p(spec.abs()).unsqueeze(0).unsqueeze(0)
    mag = F.interpolate(mag, size=(cfg.audio_bins, cfg.audio_frames), mode="bilinear", align_corners=False)
    x = mag.squeeze(0).cpu().to(torch.float16)
    return x


def video_to_tensor(path: str, cfg: Config) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = cfg.video_frames
    indices = np.linspace(0, max(frame_count - 1, 0), cfg.video_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frames.append(frames[-1])
                continue
            frame = np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
            frame = gray.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    arr = np.stack(frames, axis=0)[:, None, :, :]
    return torch.from_numpy(arr).to(torch.float16)


def _cache_path(cfg: Config, split_name: str, seed: int) -> Path:
    return cfg.cache_dir / (
        f"cremad_v{CACHE_FORMAT_VERSION}_seed{seed}_{split_name}_tensors.pt"
    )


def _legacy_duplicate_stems(cfg: Config) -> set[str]:
    legacy_path = cfg.cache_dir / "record_cache.json"
    if not legacy_path.exists():
        return set()
    try:
        raw = load_json(legacy_path)
        items = raw.get("records", []) if isinstance(raw, dict) else raw
        counts = Counter(str(item["stem"]) for item in items)
        return {stem for stem, count in counts.items() if count > 1}
    except Exception:
        return set()


def _build_cache_with_reuse(
    records: Sequence[Record],
    split_name: str,
    seed: int,
    cfg: Config,
    device: torch.device,
    target_path: Path,
) -> dict[str, torch.Tensor]:
    """Migrate compatible cached tensors by stem and preprocess only misses."""
    n = len(records)
    audio_out = torch.empty(
        (n, 1, cfg.audio_bins, cfg.audio_frames), dtype=torch.float16
    )
    video_out = torch.empty(
        (n, cfg.video_frames, 1, cfg.image_size, cfg.image_size),
        dtype=torch.float16,
    )
    filled = torch.zeros(n, dtype=torch.bool)
    positions = {record.stem: index for index, record in enumerate(records)}
    ambiguous_legacy_stems = _legacy_duplicate_stems(cfg)

    candidates = sorted(cfg.cache_dir.glob("cremad*_tensors.pt"))
    for candidate in candidates:
        if candidate.resolve() == target_path.resolve():
            continue
        try:
            source = torch.load(candidate, map_location="cpu")
            stems = source.get("stems")
            source_audio = source.get("audio")
            source_video = source.get("video")
            if (
                not isinstance(stems, list)
                or not isinstance(source_audio, torch.Tensor)
                or not isinstance(source_video, torch.Tensor)
                or len(stems) != int(source_audio.shape[0])
                or len(stems) != int(source_video.shape[0])
                or tuple(source_audio.shape[1:]) != tuple(audio_out.shape[1:])
                or tuple(source_video.shape[1:]) != tuple(video_out.shape[1:])
            ):
                continue
            source_version = int(source.get("cache_format_version", 1))
            for source_index, stem in enumerate(stems):
                target_index = positions.get(str(stem))
                if target_index is None or bool(filled[target_index]):
                    continue
                if source_version < 2 and str(stem) in ambiguous_legacy_stems:
                    continue
                audio_out[target_index].copy_(source_audio[source_index])
                video_out[target_index].copy_(source_video[source_index])
                filled[target_index] = True
        except Exception as exc:
            print(f"[cache] Could not reuse {candidate.name}: {exc}")
        finally:
            if "source" in locals():
                del source
        if bool(filled.all()):
            break

    reused = int(filled.sum().item())
    missing_indices = (~filled).nonzero(as_tuple=False).flatten().tolist()
    if reused:
        print(
            f"[cache] Reused {reused}/{n} {split_name} tensors from "
            "compatible existing caches."
        )

    use_cuda_audio = device.type == "cuda"
    for index in tqdm(
        missing_indices,
        desc=f"Preprocessing missing {split_name}",
        unit="sample",
    ):
        record = records[index]
        audio_out[index].copy_(
            audio_to_tensor(
                record.audio_path,
                cfg,
                device if use_cuda_audio else torch.device("cpu"),
            )
        )
        video_out[index].copy_(video_to_tensor(record.video_path, cfg))

    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "split_name": split_name,
        "seed": int(seed),
        "audio": audio_out,
        "video": video_out,
        "labels": torch.tensor([int(record.label) for record in records], dtype=torch.long),
        "stems": [record.stem for record in records],
    }


def build_or_load_packed_cache(records: Sequence[Record], split_name: str, seed: int, cfg: Config, device: torch.device) -> dict[str, torch.Tensor]:
    path = _cache_path(cfg, split_name, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        expected_stems = [record.stem for record in records]
        if (
            int(payload.get("cache_format_version", -1)) == CACHE_FORMAT_VERSION
            and payload.get("stems") == expected_stems
            and int(payload["labels"].shape[0]) == len(records)
        ):
            print(f"[cache] Loading {split_name}: {path}")
            return payload
        print(f"[cache] Rebuilding incompatible {split_name} cache: {path}")

    print(
        f"[cache] Building {split_name} cache with {len(records)} samples. "
        "Compatible tensors will be reused by utterance stem."
    )
    payload = _build_cache_with_reuse(
        records, split_name, seed, cfg, device, path
    )
    atomic_torch_save(payload, path)
    print(f"[cache] Saved {split_name}: {path}")
    return payload


class TensorDatasetAV(Dataset):
    def __init__(self, cache: dict[str, torch.Tensor], mode: str):
        assert mode in {"audio", "video", "fusion"}
        self.cache = cache
        self.mode = mode

    def __len__(self) -> int:
        return int(self.cache["labels"].shape[0])

    def __getitem__(self, idx: int):
        y = self.cache["labels"][idx]
        if self.mode == "audio":
            return self.cache["audio"][idx].float(), y
        if self.mode == "video":
            return self.cache["video"][idx].float(), y
        return self.cache["audio"][idx].float(), self.cache["video"][idx].float(), y
