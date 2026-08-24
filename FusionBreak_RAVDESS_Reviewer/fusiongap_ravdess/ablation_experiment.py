
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .attacks import (
    _clamp_delta,
    _collect_clean_all_three,
    _counts,
    _rates,
)
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .models import AudioNet, FusionNet, VideoNet
from .utils import get_device, save_json, set_seed, write_csv


DATASET_NAME = "RAVDESS"
DEFAULT_SEED = 42
ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")


@dataclass(frozen=True)
class AblationVariant:
    """One controlled change relative to the complete FusionBreak attack."""

    name: str
    family: str
    description: str

    use_fusion_ce: bool = True
    use_fusion_margin: bool = True
    use_preservation: bool = True
    use_distillation: bool = True
    decision_aware_selection: bool = True

    eps_scale: float = 1.0
    alpha_scale: float = 1.0
    restart_override: int | None = None
    step_override: int | None = None


def publication_variants() -> list[AblationVariant]:
    """Return a one-factor-at-a-time publication ablation plan.

    The complete method is the common reference. Default budget, restart, and
    step settings are represented by ``full_fusionbreak`` and are not rerun
    under duplicate names.
    """
    return [
        AblationVariant(
            name="full_fusionbreak",
            family="reference",
            description="Complete FusionBreak objective and decision-aware path selection.",
        ),
        AblationVariant(
            name="fusion_ce_only",
            family="objective",
            description="Fused cross-entropy only; removes fusion margin, preservation, and distillation.",
            use_fusion_margin=False,
            use_preservation=False,
            use_distillation=False,
        ),
        AblationVariant(
            name="fusion_ce_plus_margin",
            family="objective",
            description="Fused cross-entropy and fused attack margin only.",
            use_preservation=False,
            use_distillation=False,
        ),
        AblationVariant(
            name="without_fusion_margin",
            family="objective",
            description="Complete method without the fused attack-margin term.",
            use_fusion_margin=False,
        ),
        AblationVariant(
            name="without_preservation",
            family="objective",
            description="Complete method without the unimodal margin-preservation term.",
            use_preservation=False,
        ),
        AblationVariant(
            name="without_distillation",
            family="objective",
            description="Complete method without unimodal logit distillation.",
            use_distillation=False,
        ),
        AblationVariant(
            name="final_iterate_selection",
            family="candidate_selection",
            description="Complete objective, but selects only the terminal iterate of each restart.",
            decision_aware_selection=False,
        ),
        AblationVariant(
            name="budget_half",
            family="budget",
            description="Half of each modality-specific perturbation budget; step sizes scaled proportionally.",
            eps_scale=0.5,
            alpha_scale=0.5,
        ),
        AblationVariant(
            name="budget_one_and_half",
            family="budget",
            description="One-and-a-half times each modality-specific perturbation budget; step sizes scaled proportionally.",
            eps_scale=1.5,
            alpha_scale=1.5,
        ),
        AblationVariant(
            name="one_restart",
            family="restarts",
            description="Complete method with one restart.",
            restart_override=1,
        ),
        AblationVariant(
            name="five_restarts",
            family="restarts",
            description="Complete method with five restarts.",
            restart_override=5,
        ),
        AblationVariant(
            name="half_steps",
            family="steps",
            description="Complete method with half of the default optimization steps.",
            step_override=40,
        ),
        AblationVariant(
            name="one_and_half_steps",
            family="steps",
            description="Complete method with one-and-a-half times the default optimization steps.",
            step_override=120,
        ),
    ]


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    return (
        cfg.checkpoint_dir / f"audio_seed{seed}.pt",
        cfg.checkpoint_dir / f"video_seed{seed}.pt",
        cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt",
    )


def _load_checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}: {type(obj)}")
    for key in ("model", "model_state_dict", "net", "weights"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if any(isinstance(k, str) and k.startswith("module.") for k in obj):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def _load_models(seed: int, arch: str, cfg: Config, device: torch.device):
    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    missing = [str(p) for p in (audio_path, video_path, fusion_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s). Run the main RAVDESS experiments first. Missing: "
            + "; ".join(missing)
        )

    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio.load_state_dict(_load_checkpoint_state_dict(audio_path, device))
    video.load_state_dict(_load_checkpoint_state_dict(video_path, device))

    fusion_audio = copy.deepcopy(audio).to(device)
    fusion_video = copy.deepcopy(video).to(device)
    fusion = FusionNet(
        fusion_audio,
        fusion_video,
        arch,
        num_classes=cfg.num_classes,
    ).to(device)
    fusion.load_state_dict(_load_checkpoint_state_dict(fusion_path, device))

    audio.eval()
    video.eval()
    fusion.eval()
    return audio, video, fusion


def _zero_model_grads(*models: torch.nn.Module) -> None:
    for model in models:
        for parameter in model.parameters():
            parameter.grad = None


def _true_minus_other_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -1e9)
    strongest_other = masked.max(dim=1).values
    return true_logit - strongest_other


def _kl_to_clean(current_logits: torch.Tensor, clean_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(current_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction="none",
    ).sum(dim=1)


@torch.no_grad()
def _decision_aware_score(
    audio_model,
    video_model,
    fusion_model,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    audio_logits = audio_model(xa)
    video_logits = video_model(xv)
    fusion_logits = fusion_model(xa, xv)

    audio_ok = audio_logits.argmax(dim=1) == y
    video_ok = video_logits.argmax(dim=1) == y
    fusion_fail = fusion_logits.argmax(dim=1) != y
    fusion_only = fusion_fail & audio_ok & video_ok
    one_unimodal_ok = fusion_fail & (audio_ok | video_ok)

    fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
    audio_margin = _true_minus_other_margin(audio_logits, y)
    video_margin = _true_minus_other_margin(video_logits, y)

    return (
        fusion_only.float() * 1_000_000.0
        + one_unimodal_ok.float() * 100_000.0
        + fusion_fail.float() * 10_000.0
        + audio_ok.float() * 100.0
        + video_ok.float() * 100.0
        + fusion_ce
        + 0.01 * (audio_margin + video_margin)
    )


@torch.no_grad()
def _terminal_score(
    fusion_model,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Select the strongest terminal point across restarts without path tracking."""
    return F.cross_entropy(fusion_model(xa, xv), y, reduction="none")


def _effective_settings(cfg: Config, variant: AblationVariant) -> dict[str, Any]:
    steps = int(variant.step_override if variant.step_override is not None else cfg.attack_steps)
    restarts = int(
        variant.restart_override
        if variant.restart_override is not None
        else cfg.random_restarts
    )
    return {
        "attack_steps": steps,
        "random_restarts": restarts,
        "eps_audio": float(cfg.eps_audio) * variant.eps_scale,
        "eps_video": float(cfg.eps_video) * variant.eps_scale,
        "alpha_audio": float(cfg.alpha_audio) * variant.alpha_scale,
        "alpha_video": float(cfg.alpha_video) * variant.alpha_scale,
    }


def _attack_batch(
    audio_model,
    video_model,
    fusion_model,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    variant: AblationVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    settings = _effective_settings(cfg, variant)
    steps = settings["attack_steps"]
    restarts = settings["random_restarts"]
    eps_audio = settings["eps_audio"]
    eps_video = settings["eps_video"]
    alpha_audio = settings["alpha_audio"]
    alpha_video = settings["alpha_video"]

    xa0 = xa.detach()
    xv0 = xv.detach()

    with torch.no_grad():
        clean_audio_logits = audio_model(xa0).detach()
        clean_video_logits = video_model(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        if variant.decision_aware_selection:
            best_score = _decision_aware_score(
                audio_model, video_model, fusion_model, best_xa, best_xv, y
            )
        else:
            best_score = torch.full(
                (xa0.shape[0],), -1e9, device=xa0.device, dtype=torch.float32
            )

    def update_decision_aware(xa_candidate: torch.Tensor, xv_candidate: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _decision_aware_score(
                audio_model,
                video_model,
                fusion_model,
                xa_candidate,
                xv_candidate,
                y,
            )
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = xa_candidate[improve].detach()
                best_xv[improve] = xv_candidate[improve].detach()

    for restart in range(restarts):
        if restart == 0 and bool(getattr(cfg, "clean_start_first_restart", True)):
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _clamp_delta(
                xa0 + torch.empty_like(xa0).uniform_(-eps_audio, eps_audio),
                xa0,
                eps_audio,
            ).detach()
            xv_adv = _clamp_delta(
                xv0 + torch.empty_like(xv0).uniform_(-eps_video, eps_video),
                xv0,
                eps_video,
            ).detach()

        if variant.decision_aware_selection:
            update_decision_aware(xa_adv, xv_adv)

        for step in range(1, steps + 1):
            _zero_model_grads(audio_model, video_model, fusion_model)
            xa_adv.requires_grad_(True)
            xv_adv.requires_grad_(True)

            fusion_logits = fusion_model(xa_adv, xv_adv)
            audio_logits = audio_model(xa_adv)
            video_logits = video_model(xv_adv)

            objective_vec = torch.zeros(
                y.shape[0], device=y.device, dtype=fusion_logits.dtype
            )

            if variant.use_fusion_ce:
                objective_vec = objective_vec + F.cross_entropy(
                    fusion_logits, y, reduction="none"
                )

            if variant.use_fusion_margin:
                fusion_attack_margin = -_true_minus_other_margin(fusion_logits, y)
                objective_vec = (
                    objective_vec
                    + float(cfg.lambda_fusion_margin) * fusion_attack_margin
                )

            if variant.use_preservation:
                audio_margin = _true_minus_other_margin(audio_logits, y)
                video_margin = _true_minus_other_margin(video_logits, y)
                preservation_hinge = (
                    F.relu(float(cfg.preserve_margin) - audio_margin)
                    + F.relu(float(cfg.preserve_margin) - video_margin)
                )
                objective_vec = (
                    objective_vec
                    - float(cfg.lambda_preserve_margin) * preservation_hinge
                )

            if variant.use_distillation:
                distillation = _kl_to_clean(
                    audio_logits, clean_audio_logits
                ) + _kl_to_clean(video_logits, clean_video_logits)
                objective_vec = (
                    objective_vec
                    - float(cfg.lambda_logit_distill) * distillation
                )

            objective = objective_vec.mean()
            grad_audio, grad_video = torch.autograd.grad(
                objective,
                [xa_adv, xv_adv],
                retain_graph=False,
                create_graph=False,
            )

            with torch.no_grad():
                xa_adv = _clamp_delta(
                    xa_adv + alpha_audio * grad_audio.sign(),
                    xa0,
                    eps_audio,
                ).detach()
                xv_adv = _clamp_delta(
                    xv_adv + alpha_video * grad_video.sign(),
                    xv0,
                    eps_video,
                ).detach()

            if variant.decision_aware_selection:
                interval = max(1, int(getattr(cfg, "candidate_eval_interval", 1)))
                if step % interval == 0 or step == steps:
                    update_decision_aware(xa_adv, xv_adv)

        if not variant.decision_aware_selection:
            with torch.no_grad():
                terminal = _terminal_score(fusion_model, xa_adv, xv_adv, y)
                improve = terminal > best_score
                if improve.any():
                    best_score[improve] = terminal[improve]
                    best_xa[improve] = xa_adv[improve].detach()
                    best_xv[improve] = xv_adv[improve].detach()

    return best_xa.detach(), best_xv.detach()


def _zero_counts() -> dict[str, int]:
    return {
        "n": 0,
        "fusion_fail": 0,
        "audio_preserved": 0,
        "video_preserved": 0,
        "fusion_only_success": 0,
    }


def _add_counts(total: dict[str, int], batch: dict[str, int]) -> None:
    for key in total:
        total[key] += int(batch[key])


@torch.no_grad()
def _perturbation_statistics(
    xa_adv: torch.Tensor,
    xv_adv: torch.Tensor,
    xa: torch.Tensor,
    xv: torch.Tensor,
) -> dict[str, float]:
    da = (xa_adv - xa).flatten(1)
    dv = (xv_adv - xv).flatten(1)
    return {
        "audio_linf_sum": float(da.abs().max(dim=1).values.sum().item()),
        "video_linf_sum": float(dv.abs().max(dim=1).values.sum().item()),
        "audio_l2_sum": float(torch.linalg.vector_norm(da, ord=2, dim=1).sum().item()),
        "video_l2_sum": float(torch.linalg.vector_norm(dv, ord=2, dim=1).sum().item()),
        "n": int(xa.shape[0]),
    }


def _evaluate_variant(
    audio,
    video,
    fusion,
    clean_pool,
    cfg: Config,
    variant: AblationVariant,
    seed: int,
    arch: str,
    device: torch.device,
) -> dict[str, Any]:
    xa_all, xv_all, y_all = clean_pool
    loader = DataLoader(
        TensorDataset(xa_all.float(), xv_all.float(), y_all.long()),
        batch_size=int(cfg.attack_batch_size),
        shuffle=False,
        num_workers=0,
    )

    counts = _zero_counts()
    perturb = {
        "audio_linf_sum": 0.0,
        "video_linf_sum": 0.0,
        "audio_l2_sum": 0.0,
        "video_l2_sum": 0.0,
        "n": 0,
    }

    start = time.perf_counter()
    progress = tqdm(
        loader,
        desc=f"{arch} | {variant.name}",
        unit="batch",
        leave=False,
    )
    for batch_index, (xa, xv, y) in enumerate(progress):
        # Reset before each batch so every variant sees matched random starts.
        set_seed(seed * 100_000 + batch_index)
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)

        xa_adv, xv_adv = _attack_batch(
            audio,
            video,
            fusion,
            xa,
            xv,
            y,
            cfg,
            variant,
        )

        batch_counts = _counts(audio, video, fusion, xa_adv, xv_adv, y)
        _add_counts(counts, batch_counts)

        batch_perturb = _perturbation_statistics(xa_adv, xv_adv, xa, xv)
        for key in perturb:
            perturb[key] += batch_perturb[key]

        rates = _rates(counts)
        progress.set_postfix(
            {
                "ASR": f"{100.0 * rates['fusion_ASR']:.1f}%",
                "FOSR": f"{100.0 * rates['FOSR']:.1f}%",
            }
        )

    elapsed = time.perf_counter() - start
    rates = _rates(counts)
    n = max(1, int(perturb["n"]))
    settings = _effective_settings(cfg, variant)

    return {
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "fusion_architecture": arch,
        "variant": variant.name,
        "ablation_family": variant.family,
        "description": variant.description,
        "attacked_clean_samples": int(counts["n"]),
        "counts": counts,
        "rates": rates,
        "perturbation": {
            "audio_linf_mean": perturb["audio_linf_sum"] / n,
            "video_linf_mean": perturb["video_linf_sum"] / n,
            "audio_l2_mean": perturb["audio_l2_sum"] / n,
            "video_l2_mean": perturb["video_l2_sum"] / n,
        },
        "effective_settings": {
            **settings,
            "lambda_fusion_margin": (
                float(cfg.lambda_fusion_margin)
                if variant.use_fusion_margin
                else 0.0
            ),
            "lambda_preserve_margin": (
                float(cfg.lambda_preserve_margin)
                if variant.use_preservation
                else 0.0
            ),
            "lambda_logit_distill": (
                float(cfg.lambda_logit_distill)
                if variant.use_distillation
                else 0.0
            ),
            "preserve_margin": float(cfg.preserve_margin),
            "decision_aware_selection": bool(variant.decision_aware_selection),
            "use_fusion_ce": bool(variant.use_fusion_ce),
            "use_fusion_margin": bool(variant.use_fusion_margin),
            "use_preservation": bool(variant.use_preservation),
            "use_distillation": bool(variant.use_distillation),
        },
        "elapsed_seconds": float(elapsed),
    }


def _result_path(cfg: Config, seed: int, arch: str, variant: str) -> Path:
    return (
        cfg.log_dir
        / "ravdess_ablations"
        / f"ablation_seed{seed}_{arch}_{variant}.json"
    )


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    rates = result["rates"]
    perturb = result["perturbation"]
    settings = result["effective_settings"]
    return {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "fusion_architecture": result["fusion_architecture"],
        "variant": result["variant"],
        "ablation_family": result["ablation_family"],
        "attacked_clean_samples": result["attacked_clean_samples"],
        "fusion_ASR": rates["fusion_ASR"],
        "audio_preservation_rate": rates["audio_preservation_rate"],
        "video_preservation_rate": rates["video_preservation_rate"],
        "FOSR": rates["FOSR"],
        "audio_linf_mean": perturb["audio_linf_mean"],
        "video_linf_mean": perturb["video_linf_mean"],
        "audio_l2_mean": perturb["audio_l2_mean"],
        "video_l2_mean": perturb["video_l2_mean"],
        "attack_steps": settings["attack_steps"],
        "random_restarts": settings["random_restarts"],
        "eps_audio": settings["eps_audio"],
        "eps_video": settings["eps_video"],
        "alpha_audio": settings["alpha_audio"],
        "alpha_video": settings["alpha_video"],
        "decision_aware_selection": settings["decision_aware_selection"],
        "elapsed_seconds": result["elapsed_seconds"],
        "description": result["description"],
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = {row["variant"] for row in rows}
    aggregated = []
    for variant in sorted(variants):
        group = [row for row in rows if row["variant"] == variant]
        aggregated.append(
            {
                "variant": variant,
                "ablation_family": group[0]["ablation_family"],
                "architectures": len(group),
                "total_attacked_clean_samples": sum(
                    int(row["attacked_clean_samples"]) for row in group
                ),
                "mean_fusion_ASR": _mean(float(row["fusion_ASR"]) for row in group),
                "mean_audio_preservation_rate": _mean(
                    float(row["audio_preservation_rate"]) for row in group
                ),
                "mean_video_preservation_rate": _mean(
                    float(row["video_preservation_rate"]) for row in group
                ),
                "mean_FOSR": _mean(float(row["FOSR"]) for row in group),
                "total_elapsed_seconds": sum(
                    float(row["elapsed_seconds"]) for row in group
                ),
            }
        )
    return aggregated


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    order = {v.name: index for index, v in enumerate(publication_variants())}
    rows = sorted(rows, key=lambda row: order.get(row["variant"], 10_000))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RAVDESS ablation results averaged across the four fusion architectures using seed 42.}",
        r"\label{tab:ravdess-complete-ablation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Family & Variant & Fusion ASR & Audio Pres. & Video Pres. & FOSR & Time (min) \\",
        r"\midrule",
    ]
    previous_family = None
    for row in rows:
        family = str(row["ablation_family"])
        if previous_family is not None and family != previous_family:
            lines.append(r"\midrule")
        previous_family = family
        lines.append(
            "{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.1f} \\\\".format(
                _latex_escape(family),
                _latex_escape(str(row["variant"])),
                100.0 * float(row["mean_fusion_ASR"]),
                100.0 * float(row["mean_audio_preservation_rate"]),
                100.0 * float(row["mean_video_preservation_rate"]),
                100.0 * float(row["mean_FOSR"]),
                float(row["total_elapsed_seconds"]) / 60.0,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(cfg: Config, seed: int) -> None:
    result_dir = cfg.log_dir / "ravdess_ablations"
    files = sorted(result_dir.glob(f"ablation_seed{seed}_*.json"))
    results = []
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[summary] Skipping unreadable file {path}: {exc}")
            continue
        if obj.get("dataset") == DATASET_NAME and int(obj.get("seed", -1)) == seed:
            results.append(obj)

    if not results:
        print("[summary] No completed ablation results found.")
        return

    rows = [_flatten_result(result) for result in results]
    aggregate = _aggregate_rows(rows)

    artifact_dir = cfg.artifact_dir / "ravdess_ablations"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_csv(artifact_dir / f"ravdess_ablation_by_run_seed{seed}.csv", rows)
    write_csv(
        artifact_dir / f"ravdess_ablation_aggregate_seed{seed}.csv",
        aggregate,
    )
    save_json(
        artifact_dir / f"ravdess_ablation_complete_seed{seed}.json",
        {"rows": rows, "aggregate": aggregate},
    )
    _write_latex_table(
        artifact_dir / f"ravdess_ablation_table_seed{seed}.tex",
        aggregate,
    )

    print(f"[summary] Wrote ablation artifacts to: {artifact_dir}")


def run_complete_ravdess_ablations(
    cfg: Config,
    *,
    seed: int = DEFAULT_SEED,
    architectures: Iterable[str] = ARCHITECTURES,
    variant_names: set[str] | None = None,
    overwrite: bool = False,
) -> None:
    cfg.ensure_dirs()
    variants = publication_variants()
    if variant_names is not None:
        variants = [variant for variant in variants if variant.name in variant_names]
        missing = variant_names - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"Unknown ablation variants: {sorted(missing)}")

    architectures = tuple(architectures)
    unknown_architectures = set(architectures) - set(ARCHITECTURES)
    if unknown_architectures:
        raise ValueError(
            f"Unknown fusion architectures: {sorted(unknown_architectures)}"
        )

    print("RAVDESS complete FusionBreak ablations")
    print("Project root:", cfg.project_root)
    print("Seed:", seed)
    print("Architectures:", list(architectures))
    print("Variants:", [variant.name for variant in variants])
    print("CUDA available:", torch.cuda.is_available())

    device = get_device()
    records = scan_records(cfg)
    _, _, test_records = split_records(records, seed, cfg)
    test_cache = build_or_load_packed_cache(
        test_records, "test", seed, cfg, device
    )

    plan = [
        (architecture, variant)
        for architecture in architectures
        for variant in variants
    ]

    for architecture, variant in tqdm(
        plan, desc="RAVDESS ablation plan", unit="run"
    ):
        output = _result_path(cfg, seed, architecture, variant.name)
        if output.exists() and not overwrite:
            print(f"[skip] Existing result: {output}")
            continue

        set_seed(seed)
        audio, video, fusion = _load_models(
            seed, architecture, cfg, device
        )
        clean_pool = _collect_clean_all_three(
            audio, video, fusion, test_cache, cfg, device
        )
        if clean_pool is None:
            result = {
                "dataset": DATASET_NAME,
                "seed": seed,
                "fusion_architecture": architecture,
                "variant": variant.name,
                "ablation_family": variant.family,
                "description": variant.description,
                "attacked_clean_samples": 0,
                "message": "No all-three-correct clean samples.",
            }
        else:
            result = _evaluate_variant(
                audio,
                video,
                fusion,
                clean_pool,
                cfg,
                variant,
                seed,
                architecture,
                device,
            )

        save_json(output, result)
        print(f"[done] Saved: {output}")
        write_summaries(cfg, seed)

        del audio, video, fusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summaries(cfg, seed)
    print("All requested RAVDESS ablations completed.")


def _parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run publication-ready FusionBreak ablations on RAVDESS using "
            "existing seed-42 checkpoints."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--architectures",
        type=str,
        default=",".join(ARCHITECTURES),
        help="Comma-separated fusion architectures.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Optional comma-separated subset of ablation variant names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite completed per-run JSON files.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print all available variants and exit.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debugging limit; do not use for final paper runs.",
    )
    args = parser.parse_args()

    if args.list_variants:
        for variant in publication_variants():
            print(f"{variant.name:30s} [{variant.family}] {variant.description}")
        return

    cfg = Config.create_default(args.project_root)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        cfg.attack_samples = int(args.max_samples)
        print(
            "WARNING: sample limit enabled. Results are for debugging, not publication."
        )

    architectures = tuple(
        part.strip()
        for part in args.architectures.split(",")
        if part.strip()
    )
    variants = _parse_csv_set(args.variants)

    run_complete_ravdess_ablations(
        cfg,
        seed=args.seed,
        architectures=architectures,
        variant_names=variants,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
