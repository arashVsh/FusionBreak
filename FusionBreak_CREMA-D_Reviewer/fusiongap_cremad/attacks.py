from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from contextlib import contextmanager
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config


@contextmanager
def _rnn_modules_train_mode(*models: nn.Module):
    """
    cuDNN RNN layers cannot run backward while the RNN module is in eval mode.
    During adversarial-example generation we still want BatchNorm/Dropout in eval
    mode for deterministic classifier behavior, but GRU/LSTM/RNN modules must be
    marked as training for cuDNN backward. For single-layer GRUs used here, this
    does not change outputs because recurrent dropout is inactive.
    """
    seen = {}
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.RNNBase):
                seen[id(module)] = module
    previous = {mid: module.training for mid, module in seen.items()}
    try:
        for module in seen.values():
            module.train(True)
        yield
    finally:
        for mid, module in seen.items():
            module.train(previous[mid])


def _zero_model_grads(*models: nn.Module) -> None:
    # The attack only updates the input tensors, not model weights. Clearing any
    # accumulated parameter gradients avoids unnecessary memory growth.
    for model in models:
        for p in model.parameters():
            p.grad = None


@torch.no_grad()
def _collect_clean_pool(audio: nn.Module, video: nn.Module, fusion: nn.Module, cache: dict[str, torch.Tensor], cfg: Config, device: torch.device):
    ds = TensorDataset(cache["audio"].float(), cache["video"].float(), cache["labels"])
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False)
    audios = []
    videos = []
    labels = []
    for xa, xv, y in loader:
        xa = xa.to(device); xv = xv.to(device); y = y.to(device)
        ok = (audio(xa).argmax(1) == y) & (video(xv).argmax(1) == y) & (fusion(xa, xv).argmax(1) == y)
        if ok.any():
            audios.append(xa[ok].detach().cpu())
            videos.append(xv[ok].detach().cpu())
            labels.append(y[ok].detach().cpu())
    if not audios:
        return None
    xa = torch.cat(audios)[: cfg.attack_samples]
    xv = torch.cat(videos)[: cfg.attack_samples]
    y = torch.cat(labels)[: cfg.attack_samples]
    return xa, xv, y


def _clip_delta(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x_adv, x - eps, x + eps).clamp(0.0, 1.0)


def _counts(audio: nn.Module, video: nn.Module, fusion: nn.Module, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        pa = audio(xa).argmax(1)
        pv = video(xv).argmax(1)
        pf = fusion(xa, xv).argmax(1)
        audio_ok = pa == y
        video_ok = pv == y
        fusion_fail = pf != y
        fos = audio_ok & video_ok & fusion_fail
        return {
            "n": int(y.numel()),
            "fusion_fail": int(fusion_fail.sum().item()),
            "audio_preserved": int(audio_ok.sum().item()),
            "video_preserved": int(video_ok.sum().item()),
            "fusion_only_success": int(fos.sum().item()),
        }


def _rates(c: dict[str, int]) -> dict[str, float]:
    n = max(c["n"], 1)
    return {
        "fusion_ASR": c["fusion_fail"] / n,
        "audio_preservation_rate": c["audio_preserved"] / n,
        "video_preservation_rate": c["video_preserved"] / n,
        "FOSR": c["fusion_only_success"] / n,
    }


def _standard_pgd(audio: nn.Module, video: nn.Module, fusion: nn.Module, xa0: torch.Tensor, xv0: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device):
    xa = xa0.clone().to(device)
    xv = xv0.clone().to(device)
    y = y.to(device)
    step_a = cfg.eps_audio * 2.5 / cfg.attack_steps
    step_v = cfg.eps_video * 2.5 / cfg.attack_steps
    xa = _clip_delta(xa + torch.empty_like(xa).uniform_(-cfg.eps_audio, cfg.eps_audio), xa0.to(device), cfg.eps_audio).detach()
    xv = _clip_delta(xv + torch.empty_like(xv).uniform_(-cfg.eps_video, cfg.eps_video), xv0.to(device), cfg.eps_video).detach()
    with _rnn_modules_train_mode(audio, video, fusion):
        for _ in range(cfg.attack_steps):
            _zero_model_grads(audio, video, fusion)
            xa.requires_grad_(True); xv.requires_grad_(True)
            loss = F.cross_entropy(fusion(xa, xv), y)
            ga, gv = torch.autograd.grad(loss, [xa, xv], retain_graph=False, create_graph=False)
            xa = _clip_delta(xa.detach() + step_a * ga.sign(), xa0.to(device), cfg.eps_audio)
            xv = _clip_delta(xv.detach() + step_v * gv.sign(), xv0.to(device), cfg.eps_video)
    return xa.detach(), xv.detach()


def _fusiongap(audio: nn.Module, video: nn.Module, fusion: nn.Module, xa0: torch.Tensor, xv0: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device):
    b = y.numel()
    best_xa = xa0.clone().to(device)
    best_xv = xv0.clone().to(device)
    best_success = torch.zeros(b, dtype=torch.bool, device=device)
    best_score = torch.full((b,), -1e9, device=device)
    xa_base = xa0.to(device)
    xv_base = xv0.to(device)
    y = y.to(device)
    step_a = cfg.eps_audio * 2.5 / cfg.attack_steps
    step_v = cfg.eps_video * 2.5 / cfg.attack_steps

    for _restart in range(cfg.random_restarts):
        xa = _clip_delta(xa_base + torch.empty_like(xa_base).uniform_(-cfg.eps_audio, cfg.eps_audio), xa_base, cfg.eps_audio).detach()
        xv = _clip_delta(xv_base + torch.empty_like(xv_base).uniform_(-cfg.eps_video, cfg.eps_video), xv_base, cfg.eps_video).detach()
        with _rnn_modules_train_mode(audio, video, fusion):
            for _ in range(cfg.attack_steps):
                _zero_model_grads(audio, video, fusion)
                xa.requires_grad_(True); xv.requires_grad_(True)
                lf = F.cross_entropy(fusion(xa, xv), y, reduction="none")
                la = F.cross_entropy(audio(xa), y, reduction="none")
                lv = F.cross_entropy(video(xv), y, reduction="none")
                # Maximize fusion loss while discouraging unimodal loss growth.
                objective = (lf - cfg.preservation_weight * (la + lv)).mean()
                ga, gv = torch.autograd.grad(objective, [xa, xv], retain_graph=False, create_graph=False)
                xa = _clip_delta(xa.detach() + step_a * ga.sign(), xa_base, cfg.eps_audio)
                xv = _clip_delta(xv.detach() + step_v * gv.sign(), xv_base, cfg.eps_video)
                with torch.no_grad():
                    pa = audio(xa).argmax(1)
                    pv = video(xv).argmax(1)
                    pf = fusion(xa, xv).argmax(1)
                    success = (pa == y) & (pv == y) & (pf != y)
                    score = lf.detach() - 2.0 * ((pa != y).float() + (pv != y).float())
                    update = success & (~best_success | (score > best_score))
                    update |= (~best_success) & (score > best_score)
                    best_xa[update] = xa.detach()[update]
                    best_xv[update] = xv.detach()[update]
                    best_score[update] = score[update]
                    best_success[update] = best_success[update] | success[update]
    return best_xa.detach(), best_xv.detach()


def run_attacks(audio: nn.Module, video: nn.Module, fusion: nn.Module, test_cache: dict[str, torch.Tensor], cfg: Config, device: torch.device) -> dict[str, object]:
    audio.eval(); video.eval(); fusion.eval()
    pool = _collect_clean_pool(audio, video, fusion, test_cache, cfg, device)
    if pool is None:
        return {"attacked_clean_samples": 0, "error": "No all-three-correct clean samples were found."}
    xa_all, xv_all, y_all = pool
    n = int(y_all.numel())
    print(f"[attack] Attacking {n} clean samples with steps={cfg.attack_steps}, restarts={cfg.random_restarts}")

    fg_counts = {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0}
    pgd_counts = {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0}

    for start in tqdm(range(0, n, cfg.attack_batch_size), desc="Attacking", unit="batch"):
        end = min(start + cfg.attack_batch_size, n)
        xa = xa_all[start:end]
        xv = xv_all[start:end]
        y = y_all[start:end]
        xa_fg, xv_fg = _fusiongap(audio, video, fusion, xa, xv, y, cfg, device)
        xa_pgd, xv_pgd = _standard_pgd(audio, video, fusion, xa, xv, y, cfg, device)
        for target, new in [(fg_counts, _counts(audio, video, fusion, xa_fg, xv_fg, y.to(device))), (pgd_counts, _counts(audio, video, fusion, xa_pgd, xv_pgd, y.to(device)))]:
            for k, v in new.items():
                target[k] += v

    return {
        "attacked_clean_samples": n,
        "fusion_gap_counts": fg_counts,
        "fusion_gap_rates": _rates(fg_counts),
        "standard_pgd_counts": pgd_counts,
        "standard_pgd_rates": _rates(pgd_counts),
        "claim_metrics": {
            "FOSR_absolute_gain_over_PGD": _rates(fg_counts)["FOSR"] - _rates(pgd_counts)["FOSR"],
            "audio_preservation_gain_over_PGD": _rates(fg_counts)["audio_preservation_rate"] - _rates(pgd_counts)["audio_preservation_rate"],
            "video_preservation_gain_over_PGD": _rates(fg_counts)["video_preservation_rate"] - _rates(pgd_counts)["video_preservation_rate"],
        },
    }
