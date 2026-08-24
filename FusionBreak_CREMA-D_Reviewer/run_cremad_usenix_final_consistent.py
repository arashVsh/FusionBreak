from __future__ import annotations

"""
Final consistency-locked CREMA-D runner for the USENIX FusionBreak paper.

Purpose
-------
Run the experiments that were missing from the completed USENIX-extra sweep
without silently changing the dataset, checkpoints, attack budget, or FusionBreak
configuration.

This runner evaluates, on the SAME eligible pool for every seed/architecture:
  1) FusionBreak (reference rerun),
  2) AdaptiveFusionBreak,
  3) standard fused-model PGD.

The reference FusionBreak and AdaptiveFusionBreak use exactly the implementation
from run_cremad_controlled_full_vs_adaptive.py.  The only optimization-objective
difference between them is the detector-evasion penalty.  Adaptive-only extra
stealth refinement is explicitly disabled.

Strong consistency guard
------------------------
Before attacking, this script REQUIRES the completed
usenix_extras_v1_cremad_seed{seed}_{arch}.json file and verifies:
  - the exact paper attack settings,
  - test-set size,
  - eligible/all-three-correct count,
  - attacked eligible count,
  - clean validation consistency counts,
  - clean test consistency counts.

If any of these differ, the run aborts before producing attack results.  This is
intended to prevent accidental mixing of incompatible CREMA-D experiments.

The script never trains missing checkpoints.  Missing checkpoints are an error.
That prevents a resumed run from silently creating a different model.

Default GPU: cuda:1. Override with FUSIONGAP_DEVICE.

PowerShell:
    $env:FUSIONGAP_DEVICE="cuda:1"
    python .\run_cremad_usenix_final_consistent.py

Optional subset:
    $env:FUSIONGAP_SEEDS="42"
    $env:FUSIONGAP_ARCHS="concat"
    python .\run_cremad_usenix_final_consistent.py

The run is resumable at seed x architecture level. Existing final-consistent
JSON files are skipped unless FUSIONGAP_FORCE=1.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fusiongap_cremad.attacks import (
    _collect_clean_pool,
    _counts,
    _rates,
    _rnn_modules_train_mode,
    _zero_model_grads,
)
from fusiongap_cremad.config import Config
from fusiongap_cremad.data import build_or_load_packed_cache, scan_records, split_records
from fusiongap_cremad.detectors import (
    collect_detector_scores,
    detection_metrics,
    fit_detector_pack,
)
from fusiongap_cremad.train import clean_evaluation
from fusiongap_cremad.utils import save_json, set_seed
from fusiongap_cremad.adaptive_detector_experiment import _threshold_stats

# IMPORTANT: reuse the same controlled implementation imported by the completed
# USENIX-extra runner. Do not duplicate the FusionBreak objective here.
from run_cremad_controlled_full_vs_adaptive import (
    ARCHITECTURES,
    _capture_attack_rng,
    _checkpoint_paths,
    _configure_controlled_attack,
    _full_objective_attack,
    _load_controlled_models,
    _project_audio,
    _project_video,
    _restore_attack_rng,
)

DATASET_NAME = "CREMA-D"
RESULT_VERSION = 1
REFERENCE_EXTRA_VERSION = 1
ATTACK_NAMES = ("FusionBreak", "AdaptiveFusionBreak", "PGD")

# These are the exact common settings used by the completed USENIX extras.
EXPECTED_BASE_SETTINGS = {
    "attack_steps": 80,
    "random_restarts": 10,
    "attack_samples_cap": 500,
    "eps_audio": 0.08,
    "eps_video": 0.06,
    "alpha_audio": 0.004,
    "alpha_video": 0.003,
    "lambda_fusion_margin": 0.5,
    "lambda_preserve_margin": 5.0,
    "lambda_logit_distill": 0.25,
    "preserve_margin": 0.05,
}

EXPECTED_ADAPTIVE_SETTINGS = {
    "lambda_detector_evasion": 2.0,
    "lambda_detector_msp": 0.5,
    "lambda_detector_mahalanobis": 3.0,
    "lambda_detector_feature_squeezing": 5.0,
    "lambda_detector_lid": 0.0,
    "adaptive_stealth_steps": 0,
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _select_device() -> torch.device:
    requested = os.environ.get("FUSIONGAP_DEVICE", "cuda:1").strip() or "cuda:1"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {requested}") from exc
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {requested}, but only {torch.cuda.device_count()} CUDA GPU(s) are visible."
                )
        torch.cuda.set_device(torch.device(requested))
    device = torch.device(requested)
    print(f"Requested device: {requested}")
    if device.type == "cuda":
        print(f"Using GPU {device.index}: {torch.cuda.get_device_name(device)}")
    return device


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(
            f"CONSISTENCY GUARD FAILED for {name}: actual={actual!r}, expected={expected!r}"
        )


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(
            f"CONSISTENCY GUARD FAILED for {name}: actual={actual!r}, expected={expected!r}"
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_manifest(cfg: Config, seed: int, arch: str) -> dict[str, Any]:
    audio_path, video_path, fusion_path = _checkpoint_paths(cfg, seed, arch)
    paths = {"audio": audio_path, "video": video_path, "fusion": fusion_path}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Final USENIX consistency runner will NOT train missing checkpoints. "
            "Missing: " + "; ".join(missing)
        )
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _config_snapshot(cfg: Config) -> dict[str, Any]:
    return {
        "attack_steps": int(cfg.attack_steps),
        "random_restarts": int(cfg.random_restarts),
        "attack_samples_cap": int(cfg.attack_samples),
        "eps_audio": float(cfg.eps_audio),
        "eps_video": float(cfg.eps_video),
        "alpha_audio": float(cfg.alpha_audio),
        "alpha_video": float(cfg.alpha_video),
        "lambda_fusion_margin": float(cfg.lambda_fusion_margin),
        "lambda_preserve_margin": float(cfg.lambda_preserve_margin),
        "lambda_logit_distill": float(cfg.lambda_logit_distill),
        "preserve_margin": float(cfg.preserve_margin),
        "candidate_eval_interval": int(cfg.candidate_eval_interval),
        "clean_start_first_restart": bool(cfg.clean_start_first_restart),
        "lambda_detector_evasion": float(cfg.lambda_detector_evasion),
        "lambda_detector_msp": float(cfg.lambda_detector_msp),
        "lambda_detector_mahalanobis": float(cfg.lambda_detector_mahalanobis),
        "lambda_detector_feature_squeezing": float(cfg.lambda_detector_feature_squeezing),
        "lambda_detector_lid": float(cfg.lambda_detector_lid),
        "adaptive_detector_names": list(cfg.adaptive_detector_names),
        "adaptive_stealth_steps": int(cfg.adaptive_stealth_steps),
        "detector_target_fpr": float(cfg.detector_target_fpr),
    }


def _config_fingerprint(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_pinned_config(cfg: Config) -> None:
    snap = _config_snapshot(cfg)
    for key, expected in EXPECTED_BASE_SETTINGS.items():
        _assert_close(f"config.{key}", snap[key], expected)
    for key, expected in EXPECTED_ADAPTIVE_SETTINGS.items():
        _assert_close(f"config.{key}", snap[key], expected)
    _assert_equal("config.candidate_eval_interval", snap["candidate_eval_interval"], 1)
    _assert_equal("config.clean_start_first_restart", snap["clean_start_first_restart"], True)
    _assert_equal(
        "config.adaptive_detector_names",
        snap["adaptive_detector_names"],
        ["msp_hendrycks_gimpel", "mahalanobis_lee", "feature_squeezing_xu"],
    )


def _zero_counts() -> dict[str, int]:
    return {
        "n": 0,
        "fusion_fail": 0,
        "audio_preserved": 0,
        "video_preserved": 0,
        "fusion_only_success": 0,
    }


def _add_counts(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] += int(v)


def _zero_norms() -> dict[str, float]:
    return {
        "audio_linf_max": 0.0,
        "video_linf_max": 0.0,
        "audio_linf_sum": 0.0,
        "video_linf_sum": 0.0,
        "n": 0,
    }


@torch.no_grad()
def _add_norms(total, xa_adv, xv_adv, xa0, xv0) -> None:
    an = (xa_adv - xa0).abs().flatten(1).max(1).values
    vn = (xv_adv - xv0).abs().flatten(1).max(1).values
    total["audio_linf_max"] = max(total["audio_linf_max"], float(an.max().item()))
    total["video_linf_max"] = max(total["video_linf_max"], float(vn.max().item()))
    total["audio_linf_sum"] += float(an.sum().item())
    total["video_linf_sum"] += float(vn.sum().item())
    total["n"] += int(an.numel())


def _finish_norms(total) -> dict[str, float]:
    n = max(1, int(total["n"]))
    return {
        "audio_linf_max": float(total["audio_linf_max"]),
        "video_linf_max": float(total["video_linf_max"]),
        "audio_linf_mean": float(total["audio_linf_sum"]) / n,
        "video_linf_mean": float(total["video_linf_sum"]) / n,
    }


def _zero_consistency() -> dict[str, int]:
    return {
        "n": 0,
        "flagged": 0,
        "unimodal_agree": 0,
        "fusion_only_n": 0,
        "fusion_only_flagged": 0,
    }


@torch.no_grad()
def _consistency_counts(audio, video, fusion, xa, xv, y) -> dict[str, int]:
    pa = audio(xa).argmax(1)
    pv = video(xv).argmax(1)
    pf = fusion(xa, xv).argmax(1)
    agree = pa == pv
    flagged = agree & (pf != pa)
    fusion_only = (pa == y) & (pv == y) & (pf != y)
    return {
        "n": int(y.numel()),
        "flagged": int(flagged.sum().item()),
        "unimodal_agree": int(agree.sum().item()),
        "fusion_only_n": int(fusion_only.sum().item()),
        "fusion_only_flagged": int((flagged & fusion_only).sum().item()),
    }


def _add_consistency(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] += int(v)


def _consistency_rates(c: dict[str, int]) -> dict[str, float]:
    n = max(1, int(c["n"]))
    agree_n = max(1, int(c["unimodal_agree"]))
    fos_n = max(1, int(c["fusion_only_n"]))
    return {
        "flag_rate": c["flagged"] / n,
        "unimodal_agreement_rate": c["unimodal_agree"] / n,
        "flag_rate_given_unimodal_agreement": c["flagged"] / agree_n,
        "fusion_only_detection_rate": c["fusion_only_flagged"] / fos_n,
    }


@torch.no_grad()
def _cache_consistency(audio, video, fusion, cache, cfg: Config, device: torch.device) -> dict[str, Any]:
    ds = TensorDataset(cache["audio"].float(), cache["video"].float(), cache["labels"].long())
    loader = DataLoader(
        ds,
        batch_size=getattr(cfg, "eval_batch_size", cfg.batch_size),
        shuffle=False,
        num_workers=0,
    )
    total = _zero_consistency()
    all_three = 0
    for xa_cpu, xv_cpu, y_cpu in loader:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)
        pa = audio(xa).argmax(1)
        pv = video(xv).argmax(1)
        pf = fusion(xa, xv).argmax(1)
        all_three += int(((pa == y) & (pv == y) & (pf == y)).sum().item())
        _add_consistency(total, _consistency_counts(audio, video, fusion, xa, xv, y))
    return {
        "counts": total,
        "rates": _consistency_rates(total),
        "all_three_correct_count": int(all_three),
        "all_three_correct_rate": float(all_three / max(1, total["n"])),
    }


def _reference_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"usenix_extras_v{REFERENCE_EXTRA_VERSION}_cremad_seed{seed}_{arch}.json"


def _validate_reference_settings(reference: dict[str, Any]) -> None:
    settings = reference.get("attack_settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Reference USENIX-extra JSON has no attack_settings object.")
    for key, expected in EXPECTED_BASE_SETTINGS.items():
        if key not in settings:
            raise RuntimeError(f"Reference USENIX-extra JSON is missing attack setting: {key}")
        _assert_close(f"reference.attack_settings.{key}", settings[key], expected)


def _reference_guard(
    *,
    reference: dict[str, Any],
    seed: int,
    arch: str,
    test_records,
    val_cons: dict[str, Any],
    test_cons: dict[str, Any],
    clean_pool,
) -> dict[str, Any]:
    _assert_equal("reference.dataset", reference.get("dataset"), DATASET_NAME)
    _assert_equal("reference.seed", int(reference.get("seed")), int(seed))
    _assert_equal("reference.fusion_architecture", reference.get("fusion_architecture"), arch)
    _validate_reference_settings(reference)

    elig = reference.get("eligibility", {})
    ref_clean = reference.get("fusion_consistency_defense_clean", {})
    ref_val = ref_clean.get("validation", {})
    ref_test = ref_clean.get("test", {})

    _assert_equal("test_samples", len(test_records), int(elig["test_samples"]))
    _assert_equal(
        "eligible_count",
        int(test_cons["all_three_correct_count"]),
        int(elig["all_three_correct_count"]),
    )
    attacked_n = int(clean_pool[2].numel())
    _assert_equal("attacked_eligible_samples", attacked_n, int(elig["attacked_eligible_samples"]))

    # Exact clean prediction fingerprints. These are much stronger than checking
    # only scalar accuracy and will catch changed checkpoints/caches in practice.
    _assert_equal("clean_val_consistency_counts", val_cons["counts"], ref_val["counts"])
    _assert_equal("clean_test_consistency_counts", test_cons["counts"], ref_test["counts"])
    _assert_equal(
        "clean_val_all_three_correct_count",
        int(val_cons["all_three_correct_count"]),
        int(ref_val["all_three_correct_count"]),
    )
    _assert_equal(
        "clean_test_all_three_correct_count",
        int(test_cons["all_three_correct_count"]),
        int(ref_test["all_three_correct_count"]),
    )

    return {
        "passed": True,
        "reference_file": f"usenix_extras_v{REFERENCE_EXTRA_VERSION}_cremad_seed{seed}_{arch}.json",
        "test_samples": len(test_records),
        "eligible_count": int(test_cons["all_three_correct_count"]),
        "attacked_eligible_samples": attacked_n,
        "clean_val_consistency_counts_match": True,
        "clean_test_consistency_counts_match": True,
    }


def _standard_pgd(
    audio,
    video,
    fusion,
    xa: torch.Tensor,
    xv: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    device: torch.device,
):
    """Conventional fused-model PGD under the SAME numerical attack budget.

    Objective: fused cross-entropy only.
    Selection: highest final fused CE across the 10 restarts.
    It does NOT use FOSR-aware candidate selection.
    """
    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_loss = F.cross_entropy(fusion(xa0, xv0), y, reduction="none")

    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and bool(getattr(cfg, "clean_start_first_restart", True)):
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = _project_audio(
                xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio),
                xa0,
                cfg.eps_audio,
            ).detach()
            xv_adv = _project_video(
                xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video),
                xv0,
                cfg.eps_video,
            ).detach()

        with _rnn_modules_train_mode(audio, video, fusion):
            for _ in range(int(cfg.attack_steps)):
                _zero_model_grads(audio, video, fusion)
                xa_adv.requires_grad_(True)
                xv_adv.requires_grad_(True)
                loss = F.cross_entropy(fusion(xa_adv, xv_adv), y)
                ga, gv = torch.autograd.grad(loss, [xa_adv, xv_adv])
                with torch.no_grad():
                    xa_adv = _project_audio(
                        xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                    ).detach()
                    xv_adv = _project_video(
                        xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                    ).detach()

        with torch.no_grad():
            final_loss = F.cross_entropy(fusion(xa_adv, xv_adv), y, reduction="none")
            improve = final_loss > best_loss
            if improve.any():
                best_loss[improve] = final_loss[improve]
                best_xa[improve] = xa_adv[improve].detach()
                best_xv[improve] = xv_adv[improve].detach()

    return best_xa.detach(), best_xv.detach()


def _concat_score_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _run_matched_attacks(
    audio,
    video,
    fusion,
    clean_pool,
    detector_pack,
    detector_stats,
    cfg: Config,
    device: torch.device,
):
    xa_all, xv_all, y_all = clean_pool
    loader = DataLoader(
        TensorDataset(xa_all.float(), xv_all.float(), y_all.long()),
        batch_size=cfg.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    counts = {name: _zero_counts() for name in ATTACK_NAMES}
    norms = {name: _zero_norms() for name in ATTACK_NAMES}
    consistency = {name: _zero_consistency() for name in ATTACK_NAMES}
    detector_parts = {name: {} for name in ATTACK_NAMES}

    bar = tqdm(loader, desc="USENIX final matched CREMA-D", unit="batch")
    for xa_cpu, xv_cpu, y_cpu in bar:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)

        # Every method receives exactly the same random restart initializations
        # for this batch.
        matched_state = _capture_attack_rng(device)

        xa_full, xv_full = _full_objective_attack(
            audio, video, fusion, xa, xv, y, cfg, adaptive=False
        )

        _restore_attack_rng(matched_state, device)
        xa_adapt, xv_adapt = _full_objective_attack(
            audio,
            video,
            fusion,
            xa,
            xv,
            y,
            cfg,
            detector_pack=detector_pack,
            detector_stats=detector_stats,
            adaptive=True,
        )

        _restore_attack_rng(matched_state, device)
        xa_pgd, xv_pgd = _standard_pgd(audio, video, fusion, xa, xv, y, cfg, device)

        outputs = {
            "FusionBreak": (xa_full, xv_full),
            "AdaptiveFusionBreak": (xa_adapt, xv_adapt),
            "PGD": (xa_pgd, xv_pgd),
        }

        for name, (xa_adv, xv_adv) in outputs.items():
            _add_counts(counts[name], _counts(audio, video, fusion, xa_adv, xv_adv, y))
            _add_norms(norms[name], xa_adv, xv_adv, xa, xv)
            _add_consistency(
                consistency[name],
                _consistency_counts(audio, video, fusion, xa_adv, xv_adv, y),
            )
            scores = collect_detector_scores(
                fusion,
                detector_pack,
                xa_adv.detach().cpu(),
                xv_adv.detach().cpu(),
                cfg,
                device,
            )
            for det_name, arr in scores.items():
                detector_parts[name].setdefault(det_name, []).append(arr)

        fr = _rates(counts["FusionBreak"])
        ar = _rates(counts["AdaptiveFusionBreak"])
        pr = _rates(counts["PGD"])
        bar.set_postfix(
            full=f"{100*fr['FOSR']:.1f}%",
            adapt=f"{100*ar['FOSR']:.1f}%",
            pgd=f"{100*pr['FOSR']:.1f}%",
        )

    return {
        name: {
            "counts": counts[name],
            "rates": _rates(counts[name]),
            "perturbation_norms": _finish_norms(norms[name]),
            "consistency_detector": {
                "counts": consistency[name],
                "rates": _consistency_rates(consistency[name]),
            },
            "detector_scores": _concat_score_parts(detector_parts[name]),
        }
        for name in ATTACK_NAMES
    }


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"usenix_final_controlled_v{RESULT_VERSION}_cremad_seed{seed}_{arch}.json"


def run_one(seed: int, arch: str, cfg: Config, device: torch.device) -> dict[str, Any]:
    out = _result_path(cfg, seed, arch)
    if out.exists() and not _env_flag("FUSIONGAP_FORCE", False):
        print(f"[skip] {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"USENIX FINAL CONSISTENT CREMA-D: seed={seed}, arch={arch}")
    print("=" * 100)

    set_seed(seed)
    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", seed, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", seed, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", seed, cfg, device)

    # Never train here. These must be the exact checkpoints already used by the
    # completed USENIX-extra run.
    checkpoint_manifest = _checkpoint_manifest(cfg, seed, arch)

    set_seed(seed)
    audio, video, fusion = _load_controlled_models(seed, arch, cfg, device)
    clean_eval = clean_evaluation(audio, video, fusion, test_cache, cfg, device)
    val_cons = _cache_consistency(audio, video, fusion, val_cache, cfg, device)
    test_cons = _cache_consistency(audio, video, fusion, test_cache, cfg, device)
    clean_pool = _collect_clean_pool(audio, video, fusion, test_cache, cfg, device)
    if clean_pool is None:
        raise RuntimeError("No all-three-correct clean samples.")

    ref_path = _reference_path(cfg, seed, arch)
    if not ref_path.exists():
        raise FileNotFoundError(
            "CONSISTENCY GUARD requires the completed USENIX-extra reference JSON: "
            f"{ref_path}"
        )
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    guard = _reference_guard(
        reference=reference,
        seed=seed,
        arch=arch,
        test_records=test_records,
        val_cons=val_cons,
        test_cons=test_cons,
        clean_pool=clean_pool,
    )
    print(
        "[consistency guard] PASS | "
        f"test={guard['test_samples']} | eligible={guard['eligible_count']} | "
        f"attacked={guard['attacked_eligible_samples']}"
    )

    detector_pack = fit_detector_pack(fusion, train_cache, cfg, device)
    val_xa = val_cache["audio"].float()
    val_xv = val_cache["video"].float()
    val_clean_scores = collect_detector_scores(
        fusion, detector_pack, val_xa, val_xv, cfg, device
    )
    target_fpr = float(cfg.detector_target_fpr)
    detector_stats = _threshold_stats(val_clean_scores, target_fpr, device)

    attack_out = _run_matched_attacks(
        audio,
        video,
        fusion,
        clean_pool,
        detector_pack,
        detector_stats,
        cfg,
        device,
    )

    xa_clean, xv_clean, _ = clean_pool
    clean_pool_scores = collect_detector_scores(
        fusion, detector_pack, xa_clean, xv_clean, cfg, device
    )

    detector_metrics: dict[str, dict[str, Any]] = {name: {} for name in ATTACK_NAMES}
    for attack_name in ATTACK_NAMES:
        for detector_name in sorted(val_clean_scores.keys()):
            detector_metrics[attack_name][detector_name] = detection_metrics(
                clean_pool_scores[detector_name],
                attack_out[attack_name]["detector_scores"][detector_name],
                val_clean_scores[detector_name],
                target_fpr,
            )
        # Raw arrays are no longer needed in the saved result.
        attack_out[attack_name].pop("detector_scores", None)

    snapshot = _config_snapshot(cfg)
    result = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "seed": seed,
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "attacked_clean_samples": int(clean_pool[2].numel()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "reference_guard": guard,
        "checkpoint_manifest": checkpoint_manifest,
        "clean_eval": clean_eval,
        "clean_consistency_defense": {
            "definition": "flag if audio and video predictions agree but fusion predicts another class",
            "validation": val_cons,
            "test": test_cons,
        },
        "comparison_design": {
            "same_eligible_samples": True,
            "same_checkpoints": True,
            "same_steps_restarts_budgets_alphas": True,
            "matched_random_restart_initializations_within_each_batch": True,
            "same_candidate_selection_for_fusionbreak_and_adaptive": True,
            "fusionbreak_vs_adaptive_only_objective_difference": "AdaptiveFusionBreak adds detector-evasion penalty",
            "adaptive_extra_stealth_steps": 0,
            "pgd_objective": "fused cross-entropy only",
            "pgd_selection": "highest final fused CE across restarts; no FOSR-aware candidate selection",
        },
        "config_snapshot": snapshot,
        "config_sha256": _config_fingerprint(snapshot),
        "attack_metrics": attack_out,
        "detector_metrics": detector_metrics,
    }

    save_json(out, result)
    print(f"[saved] {out}")
    for name in ATTACK_NAMES:
        r = attack_out[name]["rates"]
        print(
            f"{name}: ASR={100*r['fusion_ASR']:.2f}% | "
            f"Audio={100*r['audio_preservation_rate']:.2f}% | "
            f"Video={100*r['video_preservation_rate']:.2f}% | "
            f"FOSR={100*r['FOSR']:.2f}%"
        )
    return result


def _mean_generic_detection(result: dict[str, Any], attack_name: str) -> float | None:
    dets = result.get("detector_metrics", {}).get(attack_name, {})
    vals = []
    for det in dets.values():
        if "attack_detection_rate_at_threshold" in det:
            vals.append(float(det["attack_detection_rate_at_threshold"]))
    return float(np.mean(vals)) if vals else None


def _flatten_row(result: dict[str, Any], attack_name: str) -> dict[str, Any]:
    obj = result["attack_metrics"][attack_name]
    r = obj["rates"]
    c = obj["consistency_detector"]["rates"]
    n = obj["perturbation_norms"]
    row = {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "fusion_architecture": result["fusion_architecture"],
        "attack": attack_name,
        "test_samples": result["test_samples"],
        "eligible_count": result["reference_guard"]["eligible_count"],
        "attacked_eligible_samples": result["attacked_clean_samples"],
        "fusion_ASR": r["fusion_ASR"],
        "audio_preservation_rate": r["audio_preservation_rate"],
        "video_preservation_rate": r["video_preservation_rate"],
        "FOSR": r["FOSR"],
        "clean_val_consistency_FPR": result["clean_consistency_defense"]["validation"]["rates"]["flag_rate"],
        "clean_test_consistency_FPR": result["clean_consistency_defense"]["test"]["rates"]["flag_rate"],
        "attack_consistency_detection_rate": c["flag_rate"],
        "fusion_only_consistency_detection_rate": c["fusion_only_detection_rate"],
        "audio_linf_max": n["audio_linf_max"],
        "video_linf_max": n["video_linf_max"],
        "mean_generic_detection_rate": _mean_generic_detection(result, attack_name),
        "config_sha256": result["config_sha256"],
        "reference_guard_passed": bool(result["reference_guard"]["passed"]),
    }
    for det_name, det in result.get("detector_metrics", {}).get(attack_name, {}).items():
        rate = det.get("attack_detection_rate_at_threshold")
        if rate is not None:
            row[f"det_{det_name}"] = float(rate)
    return row


def write_summary(cfg: Config) -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(cfg.log_dir.glob(f"usenix_final_controlled_v{RESULT_VERSION}_cremad_seed*_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[summary warning] {p}: {exc}")
            continue
        if "attack_metrics" not in r:
            continue
        for name in ATTACK_NAMES:
            rows.append(_flatten_row(r, name))

    if not rows:
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    csv_path = cfg.log_dir / f"usenix_final_controlled_summary_v{RESULT_VERSION}_cremad.csv"
    json_path = cfg.log_dir / f"usenix_final_controlled_summary_v{RESULT_VERSION}_cremad.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[summary] {csv_path}")
    print(f"[summary] {json_path}")


def main() -> None:
    cfg = Config(project_root=Path(__file__).resolve().parent)
    _configure_controlled_attack(cfg)
    cfg.ensure_dirs()
    _validate_pinned_config(cfg)
    device = _select_device()

    seed_env = os.environ.get("FUSIONGAP_SEEDS", "").strip()
    arch_env = os.environ.get("FUSIONGAP_ARCHS", "").strip()
    seeds = [int(x.strip()) for x in seed_env.split(",") if x.strip()] if seed_env else list(cfg.seeds)
    archs = [x.strip() for x in arch_env.split(",") if x.strip()] if arch_env else list(ARCHITECTURES)
    bad = [a for a in archs if a not in ARCHITECTURES]
    if bad:
        raise ValueError(f"Unsupported architectures: {bad}")

    snapshot = _config_snapshot(cfg)
    print("Pinned final config:")
    print(json.dumps(snapshot, indent=2))
    print("Config SHA256:", _config_fingerprint(snapshot))
    print("Seeds:", seeds)
    print("Architectures:", archs)

    plan = [(seed, arch) for seed in seeds for arch in archs]
    for seed, arch in tqdm(plan, desc="CREMA-D USENIX final controlled", unit="run"):
        run_one(seed, arch, cfg, device)
        write_summary(cfg)

    write_summary(cfg)
    print("All requested final-consistent CREMA-D runs completed.")


if __name__ == "__main__":
    main()
