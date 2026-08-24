from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config


def _collect_clean_all_three(audio_model, video_model, fusion_model, test_cache: dict, cfg: Config, device: torch.device):
    audio_model.eval()
    video_model.eval()
    fusion_model.eval()

    ds = TensorDataset(test_cache["audio"], test_cache["video"], test_cache["labels"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    audio_list, video_list, label_list = [], [], []
    with torch.no_grad():
        for xa, xv, y in loader:
            xa = xa.to(device)
            xv = xv.to(device)
            y = y.to(device)
            pa = audio_model(xa).argmax(dim=1)
            pv = video_model(xv).argmax(dim=1)
            pf = fusion_model(xa, xv).argmax(dim=1)
            mask = (pa == y) & (pv == y) & (pf == y)
            if mask.any():
                audio_list.append(xa[mask].detach().cpu())
                video_list.append(xv[mask].detach().cpu())
                label_list.append(y[mask].detach().cpu())
    if not audio_list:
        return None
    xa = torch.cat(audio_list, dim=0)
    xv = torch.cat(video_list, dim=0)
    y = torch.cat(label_list, dim=0)
    n = min(int(cfg.attack_samples), xa.shape[0])
    return xa[:n], xv[:n], y[:n]


def _clamp_delta(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)


def _true_minus_other_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Positive means the true class beats every other class."""
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -1e9)
    other_logit = masked.max(dim=1).values
    return true_logit - other_logit


def _kl_to_clean(current_logits: torch.Tensor, clean_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(clean || current), used as a soft logit-preservation term."""
    return F.kl_div(
        F.log_softmax(current_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction="none",
    ).sum(dim=1)


@torch.no_grad()
def _candidate_score(audio_model, video_model, fusion_model, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Decision-aware score for selecting the best point seen during the PGD path.

    The old code selected the final restart by a soft CE objective. On RAVDESS,
    that often passed through useful fusion-only candidates and then continued
    until one or both unimodal predictions flipped. This score gives hard
    priority to the event we actually report: fusion wrong while both unimodal
    predictions remain correct.
    """
    audio_logits = audio_model(xa)
    video_logits = video_model(xv)
    fusion_logits = fusion_model(xa, xv)

    pa = audio_logits.argmax(dim=1)
    pv = video_logits.argmax(dim=1)
    pf = fusion_logits.argmax(dim=1)

    audio_ok = pa == y
    video_ok = pv == y
    fusion_fail = pf != y
    fusion_only = fusion_fail & audio_ok & video_ok
    one_unimodal_ok = fusion_fail & (audio_ok | video_ok)

    fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
    a_margin = _true_minus_other_margin(audio_logits, y)
    v_margin = _true_minus_other_margin(video_logits, y)

    return (
        fusion_only.float() * 1_000_000.0
        + one_unimodal_ok.float() * 100_000.0
        + fusion_fail.float() * 10_000.0
        + audio_ok.float() * 100.0
        + video_ok.float() * 100.0
        + fusion_ce
        + 0.01 * (a_margin + v_margin)
    )


def _standard_pgd(audio_model, video_model, fusion_model, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device):
    xa0 = xa.detach()
    xv0 = xv.detach()
    best_xa = xa0.clone()
    best_xv = xv0.clone()
    best_loss = torch.full((xa0.shape[0],), -1e9, device=device)

    for restart in range(cfg.random_restarts):
        if restart == 0 and cfg.clean_start_first_restart:
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _clamp_delta(xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio), xa0, cfg.eps_audio).detach()
            xv_adv = _clamp_delta(xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video), xv0, cfg.eps_video).detach()

        for _ in range(cfg.attack_steps):
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)
            logits = fusion_model(xa_adv, xv_adv)
            loss_vec = F.cross_entropy(logits, y, reduction="none")
            loss = loss_vec.mean()
            ga, gv = torch.autograd.grad(loss, [xa_adv, xv_adv])
            with torch.no_grad():
                xa_adv = _clamp_delta(xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio).detach()
                xv_adv = _clamp_delta(xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video).detach()

        with torch.no_grad():
            final_loss = F.cross_entropy(fusion_model(xa_adv, xv_adv), y, reduction="none")
            improve = final_loss > best_loss
            best_loss[improve] = final_loss[improve]
            best_xa[improve] = xa_adv[improve]
            best_xv[improve] = xv_adv[improve]

    return best_xa.detach(), best_xv.detach()


def _fusiongap(audio_model, video_model, fusion_model, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device):
    """Decision-preserving FusionGap attack.

    Fixes the previous RAVDESS behavior in two ways:
    1. The first restart begins at the clean sample, not a random point. Random
       starts often already flip fragile RAVDESS unimodal classifiers.
    2. The best adversarial example is selected from the whole optimization path
       by the actual reported event: fusion failure with both unimodal decisions
       preserved. The old code selected only by a soft CE tradeoff at the end.
    """
    xa0 = xa.detach()
    xv0 = xv.detach()

    with torch.no_grad():
        clean_audio_logits = audio_model(xa0).detach()
        clean_video_logits = video_model(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio_model, video_model, fusion_model, best_xa, best_xv, y)

    def maybe_update_best(xa_cand: torch.Tensor, xv_cand: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score(audio_model, video_model, fusion_model, xa_cand, xv_cand, y)
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = xa_cand[improve].detach()
                best_xv[improve] = xv_cand[improve].detach()

    for restart in range(cfg.random_restarts):
        if restart == 0 and cfg.clean_start_first_restart:
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _clamp_delta(xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio), xa0, cfg.eps_audio).detach()
            xv_adv = _clamp_delta(xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video), xv0, cfg.eps_video).detach()

        maybe_update_best(xa_adv, xv_adv)

        for step in range(1, cfg.attack_steps + 1):
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)

            fusion_logits = fusion_model(xa_adv, xv_adv)
            audio_logits = audio_model(xa_adv)
            video_logits = video_model(xv_adv)

            fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
            # Positive when fusion is already wrong; maximize this.
            fusion_attack_margin = -_true_minus_other_margin(fusion_logits, y)

            # Penalize any reduction of unimodal true-class margins below cfg.preserve_margin.
            audio_margin = _true_minus_other_margin(audio_logits, y)
            video_margin = _true_minus_other_margin(video_logits, y)
            preservation_hinge = (
                F.relu(cfg.preserve_margin - audio_margin)
                + F.relu(cfg.preserve_margin - video_margin)
            )

            # Also keep unimodal logits close to clean logits. This is smoother than
            # argmax correctness and helps on small, fragile RAVDESS models.
            logit_distill = _kl_to_clean(audio_logits, clean_audio_logits) + _kl_to_clean(video_logits, clean_video_logits)

            objective_vec = (
                fusion_ce
                + cfg.lambda_fusion_margin * fusion_attack_margin
                - cfg.lambda_preserve_margin * preservation_hinge
                - cfg.lambda_logit_distill * logit_distill
            )
            objective = objective_vec.mean()
            ga, gv = torch.autograd.grad(objective, [xa_adv, xv_adv])

            with torch.no_grad():
                xa_adv = _clamp_delta(xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio).detach()
                xv_adv = _clamp_delta(xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video).detach()

            if cfg.candidate_eval_interval <= 1 or step % cfg.candidate_eval_interval == 0 or step == cfg.attack_steps:
                maybe_update_best(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


@torch.no_grad()
def _counts(audio_model, video_model, fusion_model, xa: torch.Tensor, xv: torch.Tensor, y: torch.Tensor) -> dict[str, int]:
    pa = audio_model(xa).argmax(dim=1)
    pv = video_model(xv).argmax(dim=1)
    pf = fusion_model(xa, xv).argmax(dim=1)
    fusion_fail = pf != y
    audio_preserved = pa == y
    video_preserved = pv == y
    fusion_only = fusion_fail & audio_preserved & video_preserved
    n = int(y.numel())
    return {
        "n": n,
        "fusion_fail": int(fusion_fail.sum().item()),
        "audio_preserved": int(audio_preserved.sum().item()),
        "video_preserved": int(video_preserved.sum().item()),
        "fusion_only_success": int(fusion_only.sum().item()),
    }


def _rates(counts: dict[str, int]) -> dict[str, float]:
    n = max(1, counts["n"])
    return {
        "fusion_ASR": counts["fusion_fail"] / n,
        "audio_preservation_rate": counts["audio_preserved"] / n,
        "video_preservation_rate": counts["video_preserved"] / n,
        "FOSR": counts["fusion_only_success"] / n,
    }


def run_attacks(audio_model, video_model, fusion_model, test_cache: dict, cfg: Config, device: torch.device) -> dict[str, object]:
    audio_model.eval()
    video_model.eval()
    fusion_model.eval()

    clean = _collect_clean_all_three(audio_model, video_model, fusion_model, test_cache, cfg, device)
    if clean is None:
        return {"attacked_clean_samples": 0, "message": "No all-three-correct samples available."}

    xa_all, xv_all, y_all = clean
    n = int(y_all.numel())
    print(
        f"[attack] Attacking {n} clean samples with steps={cfg.attack_steps}, "
        f"restarts={cfg.random_restarts}, attack_version={cfg.attack_version}"
    )

    fg_total = {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0}
    pgd_total = {"n": 0, "fusion_fail": 0, "audio_preserved": 0, "video_preserved": 0, "fusion_only_success": 0}

    ds = TensorDataset(xa_all, xv_all, y_all)
    loader = DataLoader(ds, batch_size=cfg.attack_batch_size, shuffle=False, num_workers=0)
    for xa, xv, y in tqdm(loader, desc="Attacking", unit="batch"):
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)

        xa_fg, xv_fg = _fusiongap(audio_model, video_model, fusion_model, xa, xv, y, cfg, device)
        xa_pgd, xv_pgd = _standard_pgd(audio_model, video_model, fusion_model, xa, xv, y, cfg, device)

        fg = _counts(audio_model, video_model, fusion_model, xa_fg, xv_fg, y)
        pgd = _counts(audio_model, video_model, fusion_model, xa_pgd, xv_pgd, y)
        for k in fg_total:
            fg_total[k] += fg[k]
            pgd_total[k] += pgd[k]

    fg_rates = _rates(fg_total)
    pgd_rates = _rates(pgd_total)
    return {
        "attacked_clean_samples": n,
        "attack_version": cfg.attack_version,
        "fusion_gap_counts": fg_total,
        "fusion_gap_rates": fg_rates,
        "standard_pgd_counts": pgd_total,
        "standard_pgd_rates": pgd_rates,
        "claim_metrics": {
            "FOSR_absolute_gain_over_PGD": fg_rates["FOSR"] - pgd_rates["FOSR"],
            "audio_preservation_gain_over_PGD": fg_rates["audio_preservation_rate"] - pgd_rates["audio_preservation_rate"],
            "video_preservation_gain_over_PGD": fg_rates["video_preservation_rate"] - pgd_rates["video_preservation_rate"],
        },
    }
