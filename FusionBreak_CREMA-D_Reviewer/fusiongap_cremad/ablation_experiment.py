
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .attacks import (
    _clip_delta,
    _collect_clean_pool,
    _counts,
    _rates,
    _rnn_modules_train_mode,
    _zero_model_grads,
)
from .config import Config
from .data import build_or_load_packed_cache, scan_records, split_records
from .models import AudioNet, FusionNet, VideoNet
from .utils import get_device, save_json, set_seed


DATASET_NAME = "CREMA-D"
DEFAULT_SEED = 42
ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    family: str
    description: str

    use_fusion_ce: bool = True
    use_fusion_margin: bool = True
    use_preservation: bool = True
    use_distillation: bool = True
    decision_aware_selection: bool = True

    eps_scale: float = 1.0
    restart_override: int | None = None
    step_override: int | None = None


def publication_variants() -> list[AblationVariant]:
    return [
        AblationVariant(
            "full_fusionbreak",
            "reference",
            "Complete margin-preserving FusionBreak objective with decision-aware path selection.",
        ),
        AblationVariant(
            "cremad_ce_reference",
            "reference",
            "Original CREMA-D cross-entropy preservation objective with decision-aware path selection.",
            use_fusion_margin=False,
            use_distillation=False,
        ),
        AblationVariant(
            "fusion_ce_only",
            "objective",
            "Fused cross-entropy only.",
            use_fusion_margin=False,
            use_preservation=False,
            use_distillation=False,
        ),
        AblationVariant(
            "fusion_ce_plus_margin",
            "objective",
            "Fused cross-entropy and fused margin only.",
            use_preservation=False,
            use_distillation=False,
        ),
        AblationVariant(
            "without_fusion_margin",
            "objective",
            "Complete objective without the fused margin term.",
            use_fusion_margin=False,
        ),
        AblationVariant(
            "without_preservation",
            "objective",
            "Complete objective without unimodal margin preservation.",
            use_preservation=False,
        ),
        AblationVariant(
            "without_distillation",
            "objective",
            "Complete objective without unimodal logit distillation.",
            use_distillation=False,
        ),
        AblationVariant(
            "final_iterate_selection",
            "candidate_selection",
            "Complete objective with terminal-iterate selection instead of path-wise decision-aware selection.",
            decision_aware_selection=False,
        ),
        AblationVariant(
            "budget_half",
            "budget",
            "Half of each modality-specific perturbation budget.",
            eps_scale=0.5,
        ),
        AblationVariant(
            "budget_one_and_half",
            "budget",
            "One-and-a-half times each modality-specific perturbation budget.",
            eps_scale=1.5,
        ),
        AblationVariant(
            "one_restart",
            "restarts",
            "Complete method with one restart.",
            restart_override=1,
        ),
        AblationVariant(
            "five_restarts",
            "restarts",
            "Complete method with five restarts.",
            restart_override=5,
        ),
        AblationVariant(
            "half_steps",
            "steps",
            "Complete method with half of the default optimization steps.",
            step_override=40,
        ),
        AblationVariant(
            "one_and_half_steps",
            "steps",
            "Complete method with one-and-a-half times the default optimization steps.",
            step_override=120,
        ),
    ]


def _checkpoint_paths(cfg: Config, seed: int, arch: str) -> tuple[Path, Path, Path]:
    return (
        cfg.checkpoint_dir / f"audio_seed{seed}.pt",
        cfg.checkpoint_dir / f"video_seed{seed}.pt",
        cfg.checkpoint_dir / f"fusion_seed{seed}_{arch}.pt",
    )


def _load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
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
            "Missing checkpoint(s). Run the main CREMA-D experiments first: "
            + "; ".join(missing)
        )

    audio = AudioNet(num_classes=cfg.num_classes).to(device)
    video = VideoNet(num_classes=cfg.num_classes).to(device)
    audio.load_state_dict(_load_state(audio_path, device))
    video.load_state_dict(_load_state(video_path, device))

    fusion = FusionNet(
        copy.deepcopy(audio).to(device),
        copy.deepcopy(video).to(device),
        arch,
        num_classes=cfg.num_classes,
    ).to(device)
    fusion.load_state_dict(_load_state(fusion_path, device))

    audio.eval()
    video.eval()
    fusion.eval()
    return audio, video, fusion


def _true_minus_other_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -1e9)
    return true_logit - masked.max(dim=1).values


def _kl_to_clean(current: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(current, dim=1),
        F.softmax(clean.detach(), dim=1),
        reduction="none",
    ).sum(dim=1)


@torch.no_grad()
def _candidate_score(audio, video, fusion, xa, xv, y) -> torch.Tensor:
    la = audio(xa)
    lv = video(xv)
    lf = fusion(xa, xv)

    audio_ok = la.argmax(1) == y
    video_ok = lv.argmax(1) == y
    fusion_fail = lf.argmax(1) != y
    fusion_only = fusion_fail & audio_ok & video_ok
    one_ok = fusion_fail & (audio_ok | video_ok)

    return (
        fusion_only.float() * 1_000_000.0
        + one_ok.float() * 100_000.0
        + fusion_fail.float() * 10_000.0
        + audio_ok.float() * 100.0
        + video_ok.float() * 100.0
        + F.cross_entropy(lf, y, reduction="none")
        + 0.01 * (
            _true_minus_other_margin(la, y)
            + _true_minus_other_margin(lv, y)
        )
    )


def _effective(cfg: Config, variant: AblationVariant) -> dict[str, Any]:
    steps = int(variant.step_override or cfg.attack_steps)
    restarts = int(
        variant.restart_override
        if variant.restart_override is not None
        else cfg.random_restarts
    )
    eps_a = float(cfg.eps_audio) * variant.eps_scale
    eps_v = float(cfg.eps_video) * variant.eps_scale
    # Preserve the original CREMA-D convention: step = 2.5*epsilon/steps.
    alpha_a = eps_a * 2.5 / steps
    alpha_v = eps_v * 2.5 / steps
    return {
        "attack_steps": steps,
        "random_restarts": restarts,
        "eps_audio": eps_a,
        "eps_video": eps_v,
        "alpha_audio": alpha_a,
        "alpha_video": alpha_v,
    }


def _objective(
    audio_logits: torch.Tensor,
    video_logits: torch.Tensor,
    fusion_logits: torch.Tensor,
    clean_audio_logits: torch.Tensor,
    clean_video_logits: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    variant: AblationVariant,
) -> torch.Tensor:
    # The original CREMA-D reference is retained as a named variant.
    if variant.name == "cremad_ce_reference":
        lf = F.cross_entropy(fusion_logits, y, reduction="none")
        la = F.cross_entropy(audio_logits, y, reduction="none")
        lv = F.cross_entropy(video_logits, y, reduction="none")
        return lf - float(cfg.preservation_weight) * (la + lv)

    obj = torch.zeros(y.shape[0], device=y.device, dtype=fusion_logits.dtype)

    if variant.use_fusion_ce:
        obj = obj + F.cross_entropy(fusion_logits, y, reduction="none")

    if variant.use_fusion_margin:
        lambda_f = float(getattr(cfg, "lambda_fusion_margin", 0.5))
        obj = obj + lambda_f * (-_true_minus_other_margin(fusion_logits, y))

    if variant.use_preservation:
        rho = float(getattr(cfg, "preserve_margin", 0.05))
        lambda_m = float(getattr(cfg, "lambda_preserve_margin", 5.0))
        preservation = (
            F.relu(rho - _true_minus_other_margin(audio_logits, y))
            + F.relu(rho - _true_minus_other_margin(video_logits, y))
        )
        obj = obj - lambda_m * preservation

    if variant.use_distillation:
        lambda_k = float(getattr(cfg, "lambda_logit_distill", 0.25))
        obj = obj - lambda_k * (
            _kl_to_clean(audio_logits, clean_audio_logits)
            + _kl_to_clean(video_logits, clean_video_logits)
        )

    return obj


def _attack_batch(audio, video, fusion, xa, xv, y, cfg, variant):
    settings = _effective(cfg, variant)
    xa0 = xa.detach()
    xv0 = xv.detach()

    with torch.no_grad():
        clean_a = audio(xa0).detach()
        clean_v = video(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = (
            _candidate_score(audio, video, fusion, xa0, xv0, y)
            if variant.decision_aware_selection
            else torch.full((y.shape[0],), -1e9, device=y.device)
        )

    def update_best(ca, cv):
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score(audio, video, fusion, ca, cv, y)
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = ca[improve].detach()
                best_xv[improve] = cv[improve].detach()

    for restart in range(settings["random_restarts"]):
        # The first clean start makes the full-margin method consistent with the
        # current adaptive implementation; later restarts are random.
        if restart == 0:
            xa_adv = xa0.clone()
            xv_adv = xv0.clone()
        else:
            xa_adv = _clip_delta(
                xa0 + torch.empty_like(xa0).uniform_(
                    -settings["eps_audio"], settings["eps_audio"]
                ),
                xa0,
                settings["eps_audio"],
            ).detach()
            xv_adv = _clip_delta(
                xv0 + torch.empty_like(xv0).uniform_(
                    -settings["eps_video"], settings["eps_video"]
                ),
                xv0,
                settings["eps_video"],
            ).detach()

        if variant.decision_aware_selection:
            update_best(xa_adv, xv_adv)

        with _rnn_modules_train_mode(audio, video, fusion):
            for _ in range(settings["attack_steps"]):
                _zero_model_grads(audio, video, fusion)
                xa_adv.requires_grad_(True)
                xv_adv.requires_grad_(True)

                audio_logits = audio(xa_adv)
                video_logits = video(xv_adv)
                fusion_logits = fusion(xa_adv, xv_adv)

                objective = _objective(
                    audio_logits,
                    video_logits,
                    fusion_logits,
                    clean_a,
                    clean_v,
                    y,
                    cfg,
                    variant,
                ).mean()

                ga, gv = torch.autograd.grad(
                    objective,
                    [xa_adv, xv_adv],
                    retain_graph=False,
                    create_graph=False,
                )

                xa_adv = _clip_delta(
                    xa_adv.detach() + settings["alpha_audio"] * ga.sign(),
                    xa0,
                    settings["eps_audio"],
                )
                xv_adv = _clip_delta(
                    xv_adv.detach() + settings["alpha_video"] * gv.sign(),
                    xv0,
                    settings["eps_video"],
                )

                if variant.decision_aware_selection:
                    update_best(xa_adv, xv_adv)

        if not variant.decision_aware_selection:
            with torch.no_grad():
                terminal = F.cross_entropy(
                    fusion(xa_adv, xv_adv), y, reduction="none"
                )
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
def _perturbation_stats(xa_adv, xv_adv, xa, xv):
    da = (xa_adv - xa).flatten(1)
    dv = (xv_adv - xv).flatten(1)
    return {
        "n": int(xa.shape[0]),
        "audio_linf_sum": float(da.abs().max(1).values.sum().item()),
        "video_linf_sum": float(dv.abs().max(1).values.sum().item()),
        "audio_l2_sum": float(torch.linalg.vector_norm(da, dim=1).sum().item()),
        "video_l2_sum": float(torch.linalg.vector_norm(dv, dim=1).sum().item()),
    }


def _evaluate_variant(audio, video, fusion, clean_pool, cfg, variant, seed, arch, device):
    xa_all, xv_all, y_all = clean_pool
    loader = DataLoader(
        TensorDataset(xa_all.float(), xv_all.float(), y_all.long()),
        batch_size=int(cfg.attack_batch_size),
        shuffle=False,
        num_workers=0,
    )

    counts = _zero_counts()
    perturb = {
        "n": 0,
        "audio_linf_sum": 0.0,
        "video_linf_sum": 0.0,
        "audio_l2_sum": 0.0,
        "video_l2_sum": 0.0,
    }

    start_time = time.perf_counter()
    progress = tqdm(loader, desc=f"{arch} | {variant.name}", leave=False)

    for batch_index, (xa, xv, y) in enumerate(progress):
        # Matched random starts for every variant.
        set_seed(seed * 100_000 + batch_index)
        xa = xa.to(device)
        xv = xv.to(device)
        y = y.to(device)

        xa_adv, xv_adv = _attack_batch(
            audio, video, fusion, xa, xv, y, cfg, variant
        )
        batch_counts = _counts(audio, video, fusion, xa_adv, xv_adv, y)
        _add_counts(counts, batch_counts)

        batch_perturb = _perturbation_stats(xa_adv, xv_adv, xa, xv)
        for key in perturb:
            perturb[key] += batch_perturb[key]

        rates = _rates(counts)
        progress.set_postfix(
            ASR=f"{100*rates['fusion_ASR']:.1f}%",
            FOSR=f"{100*rates['FOSR']:.1f}%",
        )

    elapsed = time.perf_counter() - start_time
    n = max(1, int(perturb["n"]))

    return {
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "fusion_architecture": arch,
        "variant": variant.name,
        "ablation_family": variant.family,
        "description": variant.description,
        "attacked_clean_samples": int(counts["n"]),
        "counts": counts,
        "rates": _rates(counts),
        "perturbation": {
            "audio_linf_mean": perturb["audio_linf_sum"] / n,
            "video_linf_mean": perturb["video_linf_sum"] / n,
            "audio_l2_mean": perturb["audio_l2_sum"] / n,
            "video_l2_mean": perturb["video_l2_sum"] / n,
        },
        "effective_settings": _effective(cfg, variant),
        "elapsed_seconds": float(elapsed),
    }


def _result_path(cfg: Config, seed: int, arch: str, variant: str) -> Path:
    return (
        cfg.log_dir
        / "cremad_ablations"
        / f"ablation_seed{seed}_{arch}_{variant}.json"
    )


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
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
        "elapsed_seconds": result["elapsed_seconds"],
        "description": result["description"],
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        output.append({
            "variant": variant,
            "ablation_family": group[0]["ablation_family"],
            "architectures": len(group),
            "total_attacked_clean_samples": sum(int(r["attacked_clean_samples"]) for r in group),
            "mean_fusion_ASR": _mean(float(r["fusion_ASR"]) for r in group),
            "mean_audio_preservation_rate": _mean(float(r["audio_preservation_rate"]) for r in group),
            "mean_video_preservation_rate": _mean(float(r["video_preservation_rate"]) for r in group),
            "mean_FOSR": _mean(float(r["FOSR"]) for r in group),
            "total_elapsed_seconds": sum(float(r["elapsed_seconds"]) for r in group),
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _write_latex(path: Path, aggregate: list[dict[str, Any]]) -> None:
    order = {v.name: i for i, v in enumerate(publication_variants())}
    aggregate = sorted(aggregate, key=lambda x: order.get(x["variant"], 999))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CREMA-D ablation results averaged across the four fusion architectures using seed 42.}",
        r"\label{tab:cremad-complete-ablation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Family & Variant & Fusion ASR & Audio Pres. & Video Pres. & FOSR & Time (min) \\",
        r"\midrule",
    ]
    previous = None
    for row in aggregate:
        family = str(row["ablation_family"])
        if previous is not None and family != previous:
            lines.append(r"\midrule")
        previous = family
        lines.append(
            "{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.1f} \\\\".format(
                _latex_escape(family),
                _latex_escape(str(row["variant"])),
                100 * float(row["mean_fusion_ASR"]),
                100 * float(row["mean_audio_preservation_rate"]),
                100 * float(row["mean_video_preservation_rate"]),
                100 * float(row["mean_FOSR"]),
                float(row["total_elapsed_seconds"]) / 60,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(cfg: Config, seed: int) -> None:
    result_dir = cfg.log_dir / "cremad_ablations"
    results = []
    for path in sorted(result_dir.glob(f"ablation_seed{seed}_*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[summary] skipped {path}: {exc}")
            continue
        if obj.get("rates"):
            results.append(obj)

    if not results:
        return

    rows = [_flatten(r) for r in results]
    aggregate = _aggregate(rows)
    out = cfg.artifact_dir / "cremad_ablations"
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / f"cremad_ablation_by_run_seed{seed}.csv", rows)
    _write_csv(out / f"cremad_ablation_aggregate_seed{seed}.csv", aggregate)
    save_json(
        out / f"cremad_ablation_complete_seed{seed}.json",
        {"rows": rows, "aggregate": aggregate},
    )
    _write_latex(out / f"cremad_ablation_table_seed{seed}.tex", aggregate)
    print(f"[summary] Wrote: {out}")


def run_complete_cremad_ablations(
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
        variants = [v for v in variants if v.name in variant_names]
        unknown = variant_names - {v.name for v in variants}
        if unknown:
            raise ValueError(f"Unknown variants: {sorted(unknown)}")

    architectures = tuple(architectures)
    unknown_arch = set(architectures) - set(ARCHITECTURES)
    if unknown_arch:
        raise ValueError(f"Unknown architectures: {sorted(unknown_arch)}")

    device = get_device()
    print("CREMA-D complete FusionBreak ablations")
    print("Seed:", seed)
    print("Architectures:", architectures)
    print("Device:", device)

    records = scan_records(cfg)
    _, _, test_records = split_records(records, seed, cfg)
    test_cache = build_or_load_packed_cache(
        test_records, "test", seed, cfg, device
    )

    for arch in architectures:
        set_seed(seed)
        audio, video, fusion = _load_models(seed, arch, cfg, device)
        clean_pool = _collect_clean_pool(
            audio, video, fusion, test_cache, cfg, device
        )
        if clean_pool is None:
            raise RuntimeError(
                f"No all-three-correct eligible samples for architecture={arch}."
            )

        for variant in variants:
            out = _result_path(cfg, seed, arch, variant.name)
            if out.exists() and not overwrite:
                print(f"[skip] {out}")
                continue

            result = _evaluate_variant(
                audio,
                video,
                fusion,
                clean_pool,
                cfg,
                variant,
                seed,
                arch,
                device,
            )
            save_json(out, result)
            print(f"[done] {out}")
            write_summaries(cfg, seed)

        del audio, video, fusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summaries(cfg, seed)
    print("All requested CREMA-D ablations completed.")


def _csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete CREMA-D FusionBreak ablations."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--architectures",
        type=str,
        default=",".join(ARCHITECTURES),
    )
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.list_variants:
        for variant in publication_variants():
            print(f"{variant.name:30s} [{variant.family}] {variant.description}")
        return

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path.cwd().resolve()
    )
    cfg = Config(project_root=project_root)

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        cfg.attack_samples = args.max_samples
        print("WARNING: debugging sample limit enabled.")

    architectures = tuple(
        x.strip() for x in args.architectures.split(",") if x.strip()
    )

    run_complete_cremad_ablations(
        cfg,
        seed=args.seed,
        architectures=architectures,
        variant_names=_csv_set(args.variants),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
