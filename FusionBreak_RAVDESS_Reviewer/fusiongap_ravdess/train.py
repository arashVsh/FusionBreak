from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def _scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=(enabled and device.type == "cuda"))




def _augment_audio_batch(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    if not cfg.use_train_augmentation:
        return x
    # x: B,1,F,T. Add light noise and SpecAugment-style masks.
    if cfg.audio_noise_std > 0:
        x = x + torch.randn_like(x) * cfg.audio_noise_std
    b, c, f, t = x.shape
    max_f = max(1, int(f * cfg.audio_mask_max_frac))
    max_t = max(1, int(t * cfg.audio_mask_max_frac))
    for i in range(b):
        if torch.rand((), device=x.device) < 0.50:
            width = int(torch.randint(1, max_f + 1, (), device=x.device).item())
            start = int(torch.randint(0, max(1, f - width + 1), (), device=x.device).item())
            x[i, :, start:start + width, :] = 0.0
        if torch.rand((), device=x.device) < 0.50:
            width = int(torch.randint(1, max_t + 1, (), device=x.device).item())
            start = int(torch.randint(0, max(1, t - width + 1), (), device=x.device).item())
            x[i, :, :, start:start + width] = 0.0
    return x.clamp(-5.0, 5.0)


def _augment_video_batch(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    if not cfg.use_train_augmentation:
        return x
    # x: B,T,1,H,W. Horizontal flip is label-preserving for RAVDESS.
    if cfg.video_hflip_prob > 0:
        mask = torch.rand(x.shape[0], device=x.device) < cfg.video_hflip_prob
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(x[mask], dims=[-1])
    if cfg.video_brightness_jitter > 0:
        scale = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 2.0 - 1.0) * cfg.video_brightness_jitter
        x = x * scale
    if cfg.video_noise_std > 0:
        x = x + torch.randn_like(x) * cfg.video_noise_std
    return x.clamp(-1.0, 1.0)


def _freeze_fusion_backbones_if_needed(model: nn.Module, cfg: Config) -> None:
    if not cfg.freeze_fusion_backbones:
        return
    for sub_name in ["audio_model", "video_model"]:
        sub = getattr(model, sub_name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
        sub.eval()


def _set_frozen_backbones_eval(model: nn.Module, cfg: Config) -> None:
    if not cfg.freeze_fusion_backbones:
        return
    for sub_name in ["audio_model", "video_model"]:
        sub = getattr(model, sub_name, None)
        if sub is not None:
            sub.eval()


def _loader_for_modality(cache: dict, modality: Literal["audio", "video"], cfg: Config, shuffle: bool) -> DataLoader:
    ds = TensorDataset(cache[modality], cache["labels"])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


def _loader_for_fusion(cache: dict, cfg: Config, shuffle: bool) -> DataLoader:
    ds = TensorDataset(cache["audio"], cache["video"], cache["labels"])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def _eval_unimodal(model: nn.Module, cache: dict, modality: Literal["audio", "video"], cfg: Config, device: torch.device) -> float:
    model.eval()
    loader = _loader_for_modality(cache, modality, cfg, shuffle=False)
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def _eval_fusion(model: nn.Module, cache: dict, cfg: Config, device: torch.device) -> float:
    model.eval()
    loader = _loader_for_fusion(cache, cfg, shuffle=False)
    correct = 0
    total = 0
    for xa, xv, y in loader:
        xa = xa.to(device, non_blocking=True)
        xv = xv.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(xa, xv)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def train_unimodal(model: nn.Module, train_cache: dict, val_cache: dict, modality: Literal["audio", "video"], cfg: Config, device: torch.device, ckpt_path: Path) -> dict[str, float | int]:
    if cfg.load_checkpoints_if_available and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[train] Loaded {modality} checkpoint: {ckpt_path}")
        val_acc = _eval_unimodal(model, val_cache, modality, cfg, device)
        return {"best_val_acc": float(val_acc), "best_epoch": -1, "loaded_checkpoint": True}

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = _scaler(device, cfg.amp)
    loader = _loader_for_modality(train_cache, modality, cfg, shuffle=True)

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in tqdm(range(1, cfg.epochs_unimodal + 1), desc=f"Training {modality}", unit="epoch"):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if modality == "audio":
                x = _augment_audio_batch(x, cfg)
            else:
                x = _augment_video_batch(x, cfg)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, cfg.amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * y.numel()
            correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total += int(y.numel())

        val_acc = _eval_unimodal(model, val_cache, modality, cfg, device)
        train_acc = correct / max(1, total)
        print(f"[{modality}] epoch={epoch} loss={total_loss/max(1,total):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={max(best_acc,val_acc):.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= cfg.min_epochs_before_early_stop and no_improve >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if cfg.save_checkpoints:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train] Saved {modality} checkpoint: {ckpt_path}")
    return {"best_val_acc": float(best_acc), "best_epoch": int(best_epoch), "loaded_checkpoint": False}


def train_fusion(model: nn.Module, train_cache: dict, val_cache: dict, cfg: Config, device: torch.device, ckpt_path: Path) -> dict[str, float | int]:
    if cfg.load_checkpoints_if_available and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[train] Loaded fusion checkpoint: {ckpt_path}")
        val_acc = _eval_fusion(model, val_cache, cfg, device)
        return {"best_val_acc": float(val_acc), "best_epoch": -1, "loaded_checkpoint": True}

    model.to(device)
    _freeze_fusion_backbones_if_needed(model, cfg)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain for fusion training.")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.fusion_lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = _scaler(device, cfg.amp)
    loader = _loader_for_fusion(train_cache, cfg, shuffle=True)

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in tqdm(range(1, cfg.epochs_fusion + 1), desc="Training fusion", unit="epoch"):
        model.train()
        _set_frozen_backbones_eval(model, cfg)
        total_loss = 0.0
        correct = 0
        total = 0
        for xa, xv, y in loader:
            xa = xa.to(device, non_blocking=True)
            xv = xv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            xa = _augment_audio_batch(xa, cfg)
            xv = _augment_video_batch(xv, cfg)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, cfg.amp):
                logits = model(xa, xv)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * y.numel()
            correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total += int(y.numel())

        val_acc = _eval_fusion(model, val_cache, cfg, device)
        train_acc = correct / max(1, total)
        print(f"[fusion] epoch={epoch} loss={total_loss/max(1,total):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={max(best_acc,val_acc):.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= cfg.min_epochs_before_early_stop and no_improve >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if cfg.save_checkpoints:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train] Saved fusion checkpoint: {ckpt_path}")
    return {"best_val_acc": float(best_acc), "best_epoch": int(best_epoch), "loaded_checkpoint": False}


@torch.no_grad()
def clean_evaluation(audio_model: nn.Module, video_model: nn.Module, fusion_model: nn.Module, test_cache: dict, cfg: Config, device: torch.device) -> dict[str, float | int]:
    audio_model.eval()
    video_model.eval()
    fusion_model.eval()
    loader = _loader_for_fusion(test_cache, cfg, shuffle=False)
    a_correct = 0
    v_correct = 0
    f_correct = 0
    all_three = 0
    total = 0
    for xa, xv, y in loader:
        xa = xa.to(device, non_blocking=True)
        xv = xv.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pa = audio_model(xa).argmax(dim=1)
        pv = video_model(xv).argmax(dim=1)
        pf = fusion_model(xa, xv).argmax(dim=1)
        ca = pa == y
        cv = pv == y
        cf = pf == y
        a_correct += int(ca.sum().item())
        v_correct += int(cv.sum().item())
        f_correct += int(cf.sum().item())
        all_three += int((ca & cv & cf).sum().item())
        total += int(y.numel())
    return {
        "audio_acc": a_correct / max(1, total),
        "video_acc": v_correct / max(1, total),
        "fusion_acc": f_correct / max(1, total),
        "all_three_correct_rate": all_three / max(1, total),
        "test_samples": total,
        "all_three_correct_count": all_three,
    }
