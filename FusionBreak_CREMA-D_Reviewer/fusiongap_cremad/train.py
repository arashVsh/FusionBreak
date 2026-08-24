from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .data import TensorDatasetAV
from .utils import atomic_torch_save


def _checkpoint_state_and_metrics(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Accept both this project's wrapped checkpoints and raw state_dict files."""
    payload = torch.load(path, map_location=device)
    metrics: dict[str, float] = {}
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    if any(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {
            key.replace("module.", "", 1): value
            for key, value in state.items()
        }
    return state, metrics


def _loader(cache: dict[str, torch.Tensor], mode: str, cfg: Config, train: bool) -> DataLoader:
    return DataLoader(
        TensorDatasetAV(cache, mode),
        batch_size=cfg.batch_size if train else cfg.eval_batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


@torch.no_grad()
def evaluate_unimodal(model: nn.Module, cache: dict[str, torch.Tensor], mode: str, cfg: Config, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in _loader(cache, mode, cfg, train=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_fusion(model: nn.Module, cache: dict[str, torch.Tensor], cfg: Config, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for xa, xv, y in _loader(cache, "fusion", cfg, train=False):
        xa = xa.to(device, non_blocking=True)
        xv = xv.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(xa, xv).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_unimodal(model: nn.Module, train_cache: dict[str, torch.Tensor], val_cache: dict[str, torch.Tensor], mode: str, cfg: Config, device: torch.device, ckpt_path: Path) -> dict[str, float]:
    if ckpt_path.exists():
        state, metrics = _checkpoint_state_and_metrics(ckpt_path, device)
        model.load_state_dict(state)
        print(f"[train] Loaded {mode} checkpoint: {ckpt_path}")
        return metrics or {
            "best_val_acc": evaluate_unimodal(
                model, val_cache, mode, cfg, device
            )
        }

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    best_val = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    loader = _loader(train_cache, mode, cfg, train=True)

    for epoch in tqdm(range(1, cfg.epochs_unimodal + 1), desc=f"Training {mode}", unit="epoch"):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        val_acc = evaluate_unimodal(model, val_cache, mode, cfg, device)
        train_acc = correct / max(total, 1)
        print(f"[{mode}] epoch={epoch} loss={loss_sum/max(total,1):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={max(best_val,val_acc):.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch >= cfg.min_epochs_before_stop and bad >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {"best_val_acc": float(best_val), "best_epoch": int(best_epoch)}
    atomic_torch_save({"state_dict": model.state_dict(), "metrics": metrics}, ckpt_path)
    print(f"[train] Saved {mode} checkpoint: {ckpt_path}")
    return metrics


def train_fusion(model: nn.Module, train_cache: dict[str, torch.Tensor], val_cache: dict[str, torch.Tensor], cfg: Config, device: torch.device, ckpt_path: Path) -> dict[str, float]:
    if ckpt_path.exists():
        state, metrics = _checkpoint_state_and_metrics(ckpt_path, device)
        model.load_state_dict(state)
        print(f"[train] Loaded fusion checkpoint: {ckpt_path}")
        return metrics or {
            "best_val_acc": evaluate_fusion(
                model, val_cache, cfg, device
            )
        }

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    best_val = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    loader = _loader(train_cache, "fusion", cfg, train=True)

    for epoch in tqdm(range(1, cfg.epochs_fusion + 1), desc="Training fusion", unit="epoch"):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for xa, xv, y in loader:
            xa = xa.to(device, non_blocking=True)
            xv = xv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(xa, xv)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        val_acc = evaluate_fusion(model, val_cache, cfg, device)
        train_acc = correct / max(total, 1)
        print(f"[fusion] epoch={epoch} loss={loss_sum/max(total,1):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={max(best_val,val_acc):.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch >= cfg.min_epochs_before_stop and bad >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {"best_val_acc": float(best_val), "best_epoch": int(best_epoch)}
    atomic_torch_save({"state_dict": model.state_dict(), "metrics": metrics}, ckpt_path)
    print(f"[train] Saved fusion checkpoint: {ckpt_path}")
    return metrics


@torch.no_grad()
def clean_evaluation(audio: nn.Module, video: nn.Module, fusion: nn.Module, test_cache: dict[str, torch.Tensor], cfg: Config, device: torch.device) -> dict[str, float]:
    audio.eval(); video.eval(); fusion.eval()
    total = 0
    ca = cv = cf = call = 0
    for xa, xv, y in _loader(test_cache, "fusion", cfg, train=False):
        xa = xa.to(device, non_blocking=True)
        xv = xv.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pa = audio(xa).argmax(dim=1)
        pv = video(xv).argmax(dim=1)
        pf = fusion(xa, xv).argmax(dim=1)
        ma = pa == y
        mv = pv == y
        mf = pf == y
        ca += int(ma.sum().item())
        cv += int(mv.sum().item())
        cf += int(mf.sum().item())
        call += int((ma & mv & mf).sum().item())
        total += int(y.numel())
    return {
        "audio_acc": ca / max(total, 1),
        "video_acc": cv / max(total, 1),
        "fusion_acc": cf / max(total, 1),
        "all_three_correct_rate": call / max(total, 1),
        "test_samples": total,
        "all_three_correct_count": call,
    }
