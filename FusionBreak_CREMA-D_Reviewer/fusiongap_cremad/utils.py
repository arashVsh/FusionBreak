from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


EMOTION_TO_LABEL = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_TO_LABEL.items()}
VIDEO_EXTS = {".flv", ".mp4", ".avi", ".mov", ".mkv"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    requested = os.environ.get("FUSIONGAP_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"FUSIONGAP_DEVICE={requested!r}, but CUDA is not available."
                )
            index = 0 if device.index is None else int(device.index)
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"FUSIONGAP_DEVICE={requested!r} is invalid; "
                    f"PyTorch sees {torch.cuda.device_count()} CUDA device(s)."
                )
        return device

    # CUDA device zero is the portable default. Multi-GPU machines can select a
    # different device explicitly, for example FUSIONGAP_DEVICE=cuda:1.
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def result_path(log_dir: Path, dataset_name: str, seed: int, arch: str) -> Path:
    return log_dir / f"fusion_gap_cremad_seed{seed}_{dataset_name}_{arch}.json"
