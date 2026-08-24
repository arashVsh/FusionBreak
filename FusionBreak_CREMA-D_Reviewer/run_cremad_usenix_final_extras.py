from __future__ import annotations

"""
Final consistency-locked CREMA-D extra experiments for the USENIX FusionBreak paper.

This is the LAST CREMA-D experiment runner intended for the paper.

Locked paper scope
------------------
Seeds: 42, 43, 44
Fusion architectures: interaction, concat, gated, group_gated
Extra methods:
  1) CEOnlySelection
  2) AudioOnlyFusionBreak
  3) VideoOnlyFusionBreak

A FusionBreak reference rerun is included ONLY as a hard consistency check. It must
exactly reproduce the already-completed final controlled FusionBreak counts for every
seed x architecture configuration. If it does not, this runner aborts and does not
write a paper-usable result for that configuration.

Why this runner is consistency-locked
-------------------------------------
The earlier CREMA-D extras were generated before the final controlled protocol was
frozen. In particular, fitting the detector pack before the attacks can advance the
RNG state. This runner reproduces the exact pre-attack sequence from
run_cremad_usenix_final_consistent.py, then gives FusionBreak and all three extras the
same per-batch restart RNG state.

The runner also preserves the same RNG progression between attack batches as the
completed final main runner. It verifies that a FusionBreak run consumes the same RNG
as the final PGD restart-initialization pattern, then restores that post-PGD-equivalent
state before the next batch.

No generic detector metrics are recomputed for the three extras because the paper's
detector comparison is already complete for FusionBreak, AdaptiveFusionBreak, and PGD.
The detector pack is still fitted before attacking solely to reproduce the final main
runner's pre-attack RNG sequence.

Requirements
------------
Place this file in the CREMA-D project root beside:
  run_cremad_usenix_final_consistent.py
  run_cremad_controlled_full_vs_adaptive.py
  fusiongap_cremad/
  CREMA-D/

The completed final controlled files
  usenix_final_controlled_v1_cremad_seed{seed}_{arch}.json
must already exist.

PowerShell smoke test:
  $env:FUSIONGAP_DEVICE="cuda:0"
  $env:FUSIONGAP_SEEDS="42"
  $env:FUSIONGAP_ARCHS="concat"
  Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
  python .\run_cremad_usenix_final_extras.py

Full locked run:
  $env:FUSIONGAP_DEVICE="cuda:0"
  $env:FUSIONGAP_SEEDS="42,43,44"
  $env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
  Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
  python .\run_cremad_usenix_final_extras.py

Existing completed per-config files are skipped unless FUSIONGAP_FORCE=1.
"""

import csv
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
from fusiongap_cremad.detectors import collect_detector_scores, fit_detector_pack
from fusiongap_cremad.train import clean_evaluation
from fusiongap_cremad.utils import save_json, set_seed
from fusiongap_cremad.adaptive_detector_experiment import (
    _kl_to_clean,
    _threshold_stats,
    _true_minus_other_margin,
)

# Exact attack implementation used by the final controlled CREMA-D experiment.
from run_cremad_controlled_full_vs_adaptive import (
    ARCHITECTURES,
    _candidate_score,
    _capture_attack_rng,
    _configure_controlled_attack,
    _full_objective_attack,
    _load_controlled_models,
    _project_audio,
    _project_video,
    _restore_attack_rng,
)

# The completed final main runner is the source of truth for configuration,
# checkpoints, eligible samples, and the FusionBreak reference counts.
from run_cremad_usenix_final_consistent import (
    _cache_consistency,
    _checkpoint_manifest,
    _config_fingerprint,
    _config_snapshot,
    _result_path as _final_main_result_path,
    _select_device,
    _validate_pinned_config,
)

DATASET_NAME = "CREMA-D"
RESULT_VERSION = 1
PAPER_SEEDS = (42, 43, 44)
PAPER_ARCHITECTURES = ("interaction", "concat", "gated", "group_gated")

REFERENCE_NAME = "FusionBreakReference"
EXTRA_ATTACK_NAMES = (
    "CEOnlySelection",
    "AudioOnlyFusionBreak",
    "VideoOnlyFusionBreak",
)
ALL_ATTACK_NAMES = (REFERENCE_NAME,) + EXTRA_ATTACK_NAMES


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _selected_seeds() -> list[int]:
    raw = os.environ.get("FUSIONGAP_SEEDS", "").strip()
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()] if raw else list(PAPER_SEEDS)
    bad = [s for s in seeds if s not in PAPER_SEEDS]
    if bad:
        raise ValueError(
            f"This final extras runner is locked to paper seeds {PAPER_SEEDS}. Invalid seed(s): {bad}"
        )
    return seeds


def _selected_architectures() -> list[str]:
    raw = os.environ.get("FUSIONGAP_ARCHS", "").strip()
    archs = [x.strip() for x in raw.split(",") if x.strip()] if raw else list(PAPER_ARCHITECTURES)
    bad = [a for a in archs if a not in PAPER_ARCHITECTURES]
    if bad:
        raise ValueError(
            f"This final extras runner is locked to architectures {PAPER_ARCHITECTURES}. Invalid: {bad}"
        )
    return archs


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(
            f"FINAL EXTRAS CONSISTENCY CHECK FAILED for {name}: "
            f"actual={actual!r}, expected={expected!r}"
        )


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-8) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(
            f"FINAL EXTRAS CONSISTENCY CHECK FAILED for {name}: "
            f"actual={actual!r}, expected={expected!r}, tol={tol}"
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
    for key, value in new.items():
        total[key] += int(value)


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
    a = (xa_adv - xa0).abs().flatten(1).max(1).values
    v = (xv_adv - xv0).abs().flatten(1).max(1).values
    total["audio_linf_max"] = max(float(total["audio_linf_max"]), float(a.max().item()))
    total["video_linf_max"] = max(float(total["video_linf_max"]), float(v.max().item()))
    total["audio_linf_sum"] += float(a.sum().item())
    total["video_linf_sum"] += float(v.sum().item())
    total["n"] += int(a.numel())


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
    for key, value in new.items():
        total[key] += int(value)


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


def _ce_only_selection_attack(audio, video, fusion, xa, xv, y, cfg: Config):
    """Fused CE only with the same restarts and decision-aware path selection.

    This is the fair-search baseline. It changes the optimization objective but keeps
    the strong intermediate candidate search used by FusionBreak.
    """
    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a: torch.Tensor, v: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score(audio, video, fusion, a, v, y)
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = a[improve].detach()
                best_xv[improve] = v[improve].detach()

    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and bool(cfg.clean_start_first_restart):
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

        maybe_update(xa_adv, xv_adv)

        with _rnn_modules_train_mode(audio, video, fusion):
            for step in range(1, int(cfg.attack_steps) + 1):
                _zero_model_grads(audio, video, fusion)
                xa_adv.requires_grad_(True)
                xv_adv.requires_grad_(True)
                loss = F.cross_entropy(fusion(xa_adv, xv_adv), y)
                ga, gv = torch.autograd.grad(
                    loss, [xa_adv, xv_adv], retain_graph=False, create_graph=False
                )
                with torch.no_grad():
                    xa_adv = _project_audio(
                        xa_adv + cfg.alpha_audio * ga.sign(), xa0, cfg.eps_audio
                    ).detach()
                    xv_adv = _project_video(
                        xv_adv + cfg.alpha_video * gv.sign(), xv0, cfg.eps_video
                    ).detach()
                if (
                    cfg.candidate_eval_interval <= 1
                    or step % cfg.candidate_eval_interval == 0
                    or step == cfg.attack_steps
                ):
                    maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _single_modality_fusionbreak(
    audio, video, fusion, xa, xv, y, cfg: Config, mode: str
):
    """Full FusionBreak objective while perturbing only one modality."""
    if mode not in {"audio", "video"}:
        raise ValueError(mode)

    xa0, xv0 = xa.detach(), xv.detach()
    with torch.no_grad():
        clean_audio_logits = audio(xa0).detach()
        clean_video_logits = video(xv0).detach()
        best_xa = xa0.clone()
        best_xv = xv0.clone()
        best_score = _candidate_score(audio, video, fusion, best_xa, best_xv, y)

    def maybe_update(a: torch.Tensor, v: torch.Tensor) -> None:
        nonlocal best_xa, best_xv, best_score
        with torch.no_grad():
            score = _candidate_score(audio, video, fusion, a, v, y)
            improve = score > best_score
            if improve.any():
                best_score[improve] = score[improve]
                best_xa[improve] = a[improve].detach()
                best_xv[improve] = v[improve].detach()

    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and bool(cfg.clean_start_first_restart):
            xa_adv = xa0.clone().detach()
            xv_adv = xv0.clone().detach()
        elif mode == "audio":
            xa_adv = _project_audio(
                xa0 + torch.empty_like(xa0).uniform_(-cfg.eps_audio, cfg.eps_audio),
                xa0,
                cfg.eps_audio,
            ).detach()
            xv_adv = xv0.clone().detach()
        else:
            xa_adv = xa0.clone().detach()
            xv_adv = _project_video(
                xv0 + torch.empty_like(xv0).uniform_(-cfg.eps_video, cfg.eps_video),
                xv0,
                cfg.eps_video,
            ).detach()

        maybe_update(xa_adv, xv_adv)

        with _rnn_modules_train_mode(audio, video, fusion):
            for step in range(1, int(cfg.attack_steps) + 1):
                _zero_model_grads(audio, video, fusion)
                if mode == "audio":
                    xa_adv.requires_grad_(True)
                else:
                    xv_adv.requires_grad_(True)

                fusion_logits = fusion(xa_adv, xv_adv)
                audio_logits = audio(xa_adv)
                video_logits = video(xv_adv)

                fusion_ce = F.cross_entropy(fusion_logits, y, reduction="none")
                fusion_margin = -_true_minus_other_margin(fusion_logits, y)
                audio_margin = _true_minus_other_margin(audio_logits, y)
                video_margin = _true_minus_other_margin(video_logits, y)
                preserve = (
                    F.relu(cfg.preserve_margin - audio_margin)
                    + F.relu(cfg.preserve_margin - video_margin)
                )
                distill = (
                    _kl_to_clean(audio_logits, clean_audio_logits)
                    + _kl_to_clean(video_logits, clean_video_logits)
                )
                objective = (
                    fusion_ce
                    + cfg.lambda_fusion_margin * fusion_margin
                    - cfg.lambda_preserve_margin * preserve
                    - cfg.lambda_logit_distill * distill
                ).mean()

                if mode == "audio":
                    (g,) = torch.autograd.grad(
                        objective, [xa_adv], retain_graph=False, create_graph=False
                    )
                    with torch.no_grad():
                        xa_adv = _project_audio(
                            xa_adv + cfg.alpha_audio * g.sign(), xa0, cfg.eps_audio
                        ).detach()
                        xv_adv = xv0.clone().detach()
                else:
                    (g,) = torch.autograd.grad(
                        objective, [xv_adv], retain_graph=False, create_graph=False
                    )
                    with torch.no_grad():
                        xv_adv = _project_video(
                            xv_adv + cfg.alpha_video * g.sign(), xv0, cfg.eps_video
                        ).detach()
                        xa_adv = xa0.clone().detach()

                if (
                    cfg.candidate_eval_interval <= 1
                    or step % cfg.candidate_eval_interval == 0
                    or step == cfg.attack_steps
                ):
                    maybe_update(xa_adv, xv_adv)

    return best_xa.detach(), best_xv.detach()


def _rng_states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _simulate_final_pgd_rng_consumption(
    matched_state: dict[str, torch.Tensor],
    xa: torch.Tensor,
    xv: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Advance RNG exactly as final PGD's random restart initialization does.

    The final PGD uses a clean first restart and, for each later restart, draws one
    full audio tensor and one full video tensor from Uniform(-eps, eps). Its gradient
    loop contains no stochastic layers in this CREMA-D model setup.
    """
    _restore_attack_rng(matched_state, device)
    for restart in range(int(cfg.random_restarts)):
        if restart == 0 and bool(cfg.clean_start_first_restart):
            continue
        torch.empty_like(xa).uniform_(-cfg.eps_audio, cfg.eps_audio)
        torch.empty_like(xv).uniform_(-cfg.eps_video, cfg.eps_video)
    return _capture_attack_rng(device)


def _run_matched_extras(
    audio,
    video,
    fusion,
    clean_pool,
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

    counts = {name: _zero_counts() for name in ALL_ATTACK_NAMES}
    norms = {name: _zero_norms() for name in ALL_ATTACK_NAMES}
    consistency = {name: _zero_consistency() for name in ALL_ATTACK_NAMES}
    rng_checks: list[bool] = []

    bar = tqdm(loader, desc="USENIX final matched CREMA-D extras", unit="batch")
    for xa_cpu, xv_cpu, y_cpu in bar:
        xa, xv, y = xa_cpu.to(device), xv_cpu.to(device), y_cpu.to(device)
        matched_state = _capture_attack_rng(device)

        # 1) Reference rerun. This must reproduce the already completed final main
        # FusionBreak counts over the full eligible pool.
        xa_ref, xv_ref = _full_objective_attack(
            audio, video, fusion, xa, xv, y, cfg, adaptive=False
        )
        state_after_reference = _capture_attack_rng(device)

        # Verify that the reference attack consumes exactly the same RNG as the
        # final main PGD's restart initialization pattern. This lets us restore the
        # correct post-batch RNG state after running the three extras.
        simulated_after_pgd = _simulate_final_pgd_rng_consumption(
            matched_state, xa, xv, cfg, device
        )
        rng_ok = _rng_states_equal(state_after_reference, simulated_after_pgd)
        if not rng_ok:
            raise RuntimeError(
                "RNG progression check failed: FusionBreak reference and the final "
                "PGD restart-initialization pattern did not leave the same RNG state. "
                "Do not use these extras."
            )
        rng_checks.append(True)

        # 2) Fair-search CE baseline with the exact same starting restart RNG.
        _restore_attack_rng(matched_state, device)
        xa_ce, xv_ce = _ce_only_selection_attack(
            audio, video, fusion, xa, xv, y, cfg
        )

        # 3) Audio-only FusionBreak, same starting RNG state.
        _restore_attack_rng(matched_state, device)
        xa_audio, xv_audio = _single_modality_fusionbreak(
            audio, video, fusion, xa, xv, y, cfg, "audio"
        )

        # 4) Video-only FusionBreak, same starting RNG state.
        _restore_attack_rng(matched_state, device)
        xa_video, xv_video = _single_modality_fusionbreak(
            audio, video, fusion, xa, xv, y, cfg, "video"
        )

        outputs = {
            REFERENCE_NAME: (xa_ref, xv_ref),
            "CEOnlySelection": (xa_ce, xv_ce),
            "AudioOnlyFusionBreak": (xa_audio, xv_audio),
            "VideoOnlyFusionBreak": (xa_video, xv_video),
        }

        for name, (xa_adv, xv_adv) in outputs.items():
            _add_counts(counts[name], _counts(audio, video, fusion, xa_adv, xv_adv, y))
            _add_norms(norms[name], xa_adv, xv_adv, xa, xv)
            _add_consistency(
                consistency[name],
                _consistency_counts(audio, video, fusion, xa_adv, xv_adv, y),
            )

        # Critical: leave this batch in the same RNG state as the completed final
        # main runner after PGD, not in the state left by the last extra method.
        _restore_attack_rng(simulated_after_pgd, device)

        rr = _rates(counts[REFERENCE_NAME])
        cr = _rates(counts["CEOnlySelection"])
        ar = _rates(counts["AudioOnlyFusionBreak"])
        vr = _rates(counts["VideoOnlyFusionBreak"])
        bar.set_postfix(
            ref=f"{100*rr['FOSR']:.1f}%",
            ce=f"{100*cr['FOSR']:.1f}%",
            audio=f"{100*ar['FOSR']:.1f}%",
            video=f"{100*vr['FOSR']:.1f}%",
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
        }
        for name in ALL_ATTACK_NAMES
    }, {
        "per_batch_rng_progression_checks": len(rng_checks),
        "all_rng_progression_checks_passed": all(rng_checks),
    }


def _result_path(cfg: Config, seed: int, arch: str) -> Path:
    return cfg.log_dir / f"usenix_final_extras_v{RESULT_VERSION}_cremad_seed{seed}_{arch}.json"


def _load_final_reference(cfg: Config, seed: int, arch: str) -> dict[str, Any]:
    path = _final_main_result_path(cfg, seed, arch)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing completed final main result: {path}. "
            "Run run_cremad_usenix_final_consistent.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = _config_fingerprint(_config_snapshot(cfg))
    _assert_equal("final_main.config_sha256", data.get("config_sha256"), expected_hash)
    return data


def _checkpoint_sha_only(manifest: dict[str, Any]) -> dict[str, str]:
    return {name: str(obj["sha256"]) for name, obj in manifest.items()}


def run_one(seed: int, arch: str, cfg: Config, device: torch.device) -> dict[str, Any]:
    out = _result_path(cfg, seed, arch)
    if out.exists() and not _env_flag("FUSIONGAP_FORCE", False):
        print(f"[skip] {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print(f"USENIX FINAL CONSISTENT CREMA-D EXTRAS: seed={seed}, arch={arch}")
    print("=" * 100)

    final_ref = _load_final_reference(cfg, seed, arch)

    # Reproduce the exact final-main pre-attack sequence.
    set_seed(seed)
    records = scan_records(cfg)
    train_records, val_records, test_records = split_records(records, seed, cfg)
    train_cache = build_or_load_packed_cache(train_records, "train", seed, cfg, device)
    val_cache = build_or_load_packed_cache(val_records, "val", seed, cfg, device)
    test_cache = build_or_load_packed_cache(test_records, "test", seed, cfg, device)

    checkpoint_manifest = _checkpoint_manifest(cfg, seed, arch)
    _assert_equal(
        "checkpoint SHA manifest",
        _checkpoint_sha_only(checkpoint_manifest),
        _checkpoint_sha_only(final_ref["checkpoint_manifest"]),
    )

    set_seed(seed)
    audio, video, fusion = _load_controlled_models(seed, arch, cfg, device)
    clean_eval = clean_evaluation(audio, video, fusion, test_cache, cfg, device)
    val_cons = _cache_consistency(audio, video, fusion, val_cache, cfg, device)
    test_cons = _cache_consistency(audio, video, fusion, test_cache, cfg, device)
    clean_pool = _collect_clean_pool(audio, video, fusion, test_cache, cfg, device)
    if clean_pool is None:
        raise RuntimeError("No all-three-correct clean samples.")

    # Hard guards against any changed data/model/eligibility state.
    _assert_equal("train_samples", len(train_records), int(final_ref["train_samples"]))
    _assert_equal("val_samples", len(val_records), int(final_ref["val_samples"]))
    _assert_equal("test_samples", len(test_records), int(final_ref["test_samples"]))
    _assert_equal(
        "eligible_count",
        int(clean_pool[2].numel()),
        int(final_ref["attacked_clean_samples"]),
    )
    _assert_equal(
        "clean validation consistency counts",
        val_cons["counts"],
        final_ref["clean_consistency_defense"]["validation"]["counts"],
    )
    _assert_equal(
        "clean test consistency counts",
        test_cons["counts"],
        final_ref["clean_consistency_defense"]["test"]["counts"],
    )

    # IMPORTANT: the completed final main runner fit the detector pack and computed
    # validation detector scores BEFORE attacking. Reproduce that sequence even
    # though the three extras do not need generic detector metrics themselves.
    detector_pack = fit_detector_pack(fusion, train_cache, cfg, device)
    val_clean_scores = collect_detector_scores(
        fusion,
        detector_pack,
        val_cache["audio"].float(),
        val_cache["video"].float(),
        cfg,
        device,
    )
    _ = _threshold_stats(val_clean_scores, float(cfg.detector_target_fpr), device)

    attack_out, rng_audit = _run_matched_extras(
        audio, video, fusion, clean_pool, cfg, device
    )

    # Reference check against the already completed authoritative FusionBreak run.
    expected_ref = final_ref["attack_metrics"]["FusionBreak"]
    actual_ref = attack_out[REFERENCE_NAME]
    reference_exact = actual_ref["counts"] == expected_ref["counts"]
    if not reference_exact:
        raise RuntimeError(
            "FUSIONBREAK REFERENCE REPRODUCTION FAILED.\n"
            f"Expected final-main counts: {expected_ref['counts']}\n"
            f"Actual rerun counts:       {actual_ref['counts']}\n"
            "Do not use these extras until the RNG/pre-attack sequence is reconciled."
        )

    _assert_close(
        "reference audio_linf_max",
        actual_ref["perturbation_norms"]["audio_linf_max"],
        expected_ref["perturbation_norms"]["audio_linf_max"],
        tol=1e-6,
    )
    _assert_close(
        "reference video_linf_max",
        actual_ref["perturbation_norms"]["video_linf_max"],
        expected_ref["perturbation_norms"]["video_linf_max"],
        tol=1e-6,
    )

    print(
        f"[reference PASS] seed={seed}, arch={arch}: FusionBreak exactly reproduces "
        f"final main counts | FOSR={100*actual_ref['rates']['FOSR']:.2f}%"
    )

    # Threat-model and implementation checks.
    tol = 1e-5
    for name, obj in attack_out.items():
        norms = obj["perturbation_norms"]
        if norms["audio_linf_max"] > float(cfg.eps_audio) + tol:
            raise RuntimeError(f"Audio budget violation in {name}: {norms['audio_linf_max']}")
        if norms["video_linf_max"] > float(cfg.eps_video) + tol:
            raise RuntimeError(f"Video budget violation in {name}: {norms['video_linf_max']}")
        if int(obj["counts"]["n"]) != int(clean_pool[2].numel()):
            raise RuntimeError(f"Eligible coverage mismatch in {name}")

    audio_untouched = attack_out["AudioOnlyFusionBreak"]["perturbation_norms"]["video_linf_max"]
    video_untouched = attack_out["VideoOnlyFusionBreak"]["perturbation_norms"]["audio_linf_max"]
    if audio_untouched > 1e-8:
        raise RuntimeError(
            f"Audio-only attack changed video tensor: max delta={audio_untouched}"
        )
    if video_untouched > 1e-8:
        raise RuntimeError(
            f"Video-only attack changed audio tensor: max delta={video_untouched}"
        )

    test_n = int(final_ref["test_samples"])
    eligible_n = int(clean_pool[2].numel())
    for obj in attack_out.values():
        obj["test_set_fusion_only_yield"] = (
            int(obj["counts"]["fusion_only_success"]) / max(1, test_n)
        )

    snapshot = _config_snapshot(cfg)
    result = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "seed": int(seed),
        "fusion_architecture": arch,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "eligible_count": eligible_n,
        "eligible_rate": eligible_n / max(1, test_n),
        "attacked_eligible_samples": eligible_n,
        "eligible_attack_coverage": 1.0,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "final_main_reference_path": str(_final_main_result_path(cfg, seed, arch)),
        "reference_fusionbreak_exact_match": True,
        "reference_fusionbreak_expected_counts": expected_ref["counts"],
        "reference_fusionbreak_rerun_counts": actual_ref["counts"],
        "checkpoint_manifest": checkpoint_manifest,
        "clean_eval": clean_eval,
        "clean_consistency_defense": {
            "definition": "flag if audio and video predictions agree but fusion predicts another class",
            "validation": val_cons,
            "test": test_cons,
        },
        "comparison_design": {
            "paper_seeds": list(PAPER_SEEDS),
            "paper_architectures": list(PAPER_ARCHITECTURES),
            "same_final_checkpoints": True,
            "same_final_eligible_pool": True,
            "same_steps_restarts_budgets_alphas": True,
            "same_decision_aware_candidate_selection": True,
            "same_per_batch_restart_rng_as_reference_fusionbreak": True,
            "pre_attack_detector_fit_reproduced_for_rng_alignment": True,
            "generic_detector_metrics_for_extras_computed": False,
            "fusionbreak_reference_is_check_only": True,
            "ce_only_objective": "fused cross-entropy only",
            "audio_only_untouched_video": True,
            "video_only_untouched_audio": True,
        },
        "rng_audit": rng_audit,
        "config_snapshot": snapshot,
        "config_sha256": _config_fingerprint(snapshot),
        "attack_metrics": attack_out,
    }

    save_json(out, result)
    print(f"[saved] {out}")
    for name in EXTRA_ATTACK_NAMES:
        r = attack_out[name]["rates"]
        print(
            f"{name}: ASR={100*r['fusion_ASR']:.2f}% | "
            f"Audio={100*r['audio_preservation_rate']:.2f}% | "
            f"Video={100*r['video_preservation_rate']:.2f}% | "
            f"FOSR={100*r['FOSR']:.2f}%"
        )
    return result


def _flatten_row(result: dict[str, Any], attack_name: str) -> dict[str, Any]:
    obj = result["attack_metrics"][attack_name]
    rates = obj["rates"]
    cons = obj["consistency_detector"]["rates"]
    norms = obj["perturbation_norms"]
    return {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "fusion_architecture": result["fusion_architecture"],
        "attack": attack_name,
        "is_reference_only": attack_name == REFERENCE_NAME,
        "test_samples": result["test_samples"],
        "eligible_count": result["eligible_count"],
        "eligible_rate": result["eligible_rate"],
        "attacked_eligible_samples": result["attacked_eligible_samples"],
        "eligible_attack_coverage": result["eligible_attack_coverage"],
        "fusion_ASR": rates["fusion_ASR"],
        "audio_preservation_rate": rates["audio_preservation_rate"],
        "video_preservation_rate": rates["video_preservation_rate"],
        "FOSR": rates["FOSR"],
        "clean_val_consistency_FPR": result["clean_consistency_defense"]["validation"]["rates"]["flag_rate"],
        "clean_test_consistency_FPR": result["clean_consistency_defense"]["test"]["rates"]["flag_rate"],
        "attack_consistency_detection_rate": cons["flag_rate"],
        "fusion_only_consistency_detection_rate": cons["fusion_only_detection_rate"],
        "test_set_fusion_only_yield": obj["test_set_fusion_only_yield"],
        "audio_linf_max": norms["audio_linf_max"],
        "video_linf_max": norms["video_linf_max"],
        "config_sha256": result["config_sha256"],
        "reference_fusionbreak_exact_match": result["reference_fusionbreak_exact_match"],
        "rng_progression_passed": result["rng_audit"]["all_rng_progression_checks_passed"],
    }


def _load_completed_results(cfg: Config) -> list[dict[str, Any]]:
    results = []
    for seed in PAPER_SEEDS:
        for arch in PAPER_ARCHITECTURES:
            path = _result_path(cfg, seed, arch)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[summary warning] {path}: {exc}")
                continue
            if "attack_metrics" in data:
                results.append(data)
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "fusion_ASR",
        "audio_preservation_rate",
        "video_preservation_rate",
        "FOSR",
        "attack_consistency_detection_rate",
        "fusion_only_consistency_detection_rate",
        "test_set_fusion_only_yield",
    ]
    out = []
    for attack_name in ALL_ATTACK_NAMES:
        sub = [r for r in rows if r["attack"] == attack_name]
        if not sub:
            continue
        row: dict[str, Any] = {
            "attack": attack_name,
            "is_reference_only": attack_name == REFERENCE_NAME,
            "config_count": len(sub),
            "total_eligible": int(sum(int(r["eligible_count"]) for r in sub)),
        }
        weights = np.asarray([float(r["eligible_count"]) for r in sub], dtype=float)
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in sub], dtype=float)
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"weighted_{metric}"] = float(np.average(vals, weights=weights))
        out.append(row)
    return out


def write_outputs(cfg: Config) -> None:
    results = _load_completed_results(cfg)
    rows = [
        _flatten_row(result, attack_name)
        for result in results
        for attack_name in ALL_ATTACK_NAMES
    ]

    summary_csv = cfg.log_dir / f"usenix_final_extras_summary_v{RESULT_VERSION}_cremad.csv"
    summary_json = cfg.log_dir / f"usenix_final_extras_summary_v{RESULT_VERSION}_cremad.json"
    _write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    aggregate = _aggregate_rows(rows)
    aggregate_csv = cfg.log_dir / f"usenix_final_extras_aggregate_v{RESULT_VERSION}_cremad.csv"
    aggregate_json = cfg.log_dir / f"usenix_final_extras_aggregate_v{RESULT_VERSION}_cremad.json"
    _write_csv(aggregate_csv, aggregate)
    aggregate_json.write_text(json.dumps({"rows": aggregate}, indent=2), encoding="utf-8")

    present = sorted(
        [(int(r["seed"]), str(r["fusion_architecture"])) for r in results]
    )
    expected = sorted(
        [(seed, arch) for seed in PAPER_SEEDS for arch in PAPER_ARCHITECTURES]
    )
    hashes = sorted({str(r["config_sha256"]) for r in results})

    audit = {
        "dataset": DATASET_NAME,
        "result_version": RESULT_VERSION,
        "paper_seeds": list(PAPER_SEEDS),
        "paper_architectures": list(PAPER_ARCHITECTURES),
        "expected_configs": len(expected),
        "present_configs": len(present),
        "configs_present": present,
        "missing_configs": [list(x) for x in expected if x not in present],
        "complete": present == expected,
        "unique_config_sha256": hashes,
        "all_reference_fusionbreak_exact_match": bool(results) and all(
            bool(r.get("reference_fusionbreak_exact_match")) for r in results
        ),
        "all_rng_progression_checks_passed": bool(results) and all(
            bool(r.get("rng_audit", {}).get("all_rng_progression_checks_passed"))
            for r in results
        ),
        "all_full_eligible_coverage": bool(results) and all(
            abs(float(r.get("eligible_attack_coverage", 0.0)) - 1.0) < 1e-12
            for r in results
        ),
        "all_audio_only_video_untouched": bool(results) and all(
            float(r["attack_metrics"]["AudioOnlyFusionBreak"]["perturbation_norms"]["video_linf_max"]) <= 1e-8
            for r in results
        ),
        "all_video_only_audio_untouched": bool(results) and all(
            float(r["attack_metrics"]["VideoOnlyFusionBreak"]["perturbation_norms"]["audio_linf_max"]) <= 1e-8
            for r in results
        ),
        "generic_detector_metrics_for_extras_computed": False,
        "note": (
            "FusionBreakReference is a consistency check only. Paper-facing extra methods are "
            "CEOnlySelection, AudioOnlyFusionBreak, and VideoOnlyFusionBreak."
        ),
    }
    audit_path = cfg.log_dir / f"usenix_final_extras_audit_v{RESULT_VERSION}_cremad.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Paper-friendly compact output. Mean +/- sample SD is across the 12
    # seed x architecture configurations. Weighted values are also provided.
    paper_rows = []
    for row in aggregate:
        if row["is_reference_only"]:
            continue
        paper_rows.append({
            "attack": row["attack"],
            "configs": row["config_count"],
            "total_eligible": row["total_eligible"],
            "Fusion_ASR_mean_percent": 100 * row["mean_fusion_ASR"],
            "Fusion_ASR_sd_percent": 100 * row["sd_fusion_ASR"],
            "Audio_Pres_mean_percent": 100 * row["mean_audio_preservation_rate"],
            "Audio_Pres_sd_percent": 100 * row["sd_audio_preservation_rate"],
            "Video_Pres_mean_percent": 100 * row["mean_video_preservation_rate"],
            "Video_Pres_sd_percent": 100 * row["sd_video_preservation_rate"],
            "FOSR_mean_percent": 100 * row["mean_FOSR"],
            "FOSR_sd_percent": 100 * row["sd_FOSR"],
            "weighted_FOSR_percent": 100 * row["weighted_FOSR"],
            "weighted_Fusion_ASR_percent": 100 * row["weighted_fusion_ASR"],
            "weighted_Audio_Pres_percent": 100 * row["weighted_audio_preservation_rate"],
            "weighted_Video_Pres_percent": 100 * row["weighted_video_preservation_rate"],
        })

    paper_path = cfg.log_dir / f"usenix_final_extras_paper_table_v{RESULT_VERSION}_cremad.json"
    paper_path.write_text(
        json.dumps(
            {
                "aggregation_note": (
                    "mean/sd are unweighted across the 12 seed x architecture configurations; "
                    "weighted values pool by eligible-count weights"
                ),
                "rows": paper_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[summary] {summary_csv}")
    print(f"[summary] {summary_json}")
    print(f"[aggregate] {aggregate_csv}")
    print(f"[aggregate] {aggregate_json}")
    print(f"[paper table] {paper_path}")
    print(f"[audit] complete={audit['complete']} | {audit_path}")


def main() -> None:
    cfg = Config(project_root=Path(__file__).resolve().parent)
    _configure_controlled_attack(cfg)
    cfg.ensure_dirs()
    _validate_pinned_config(cfg)

    device = _select_device()
    seeds = _selected_seeds()
    archs = _selected_architectures()

    snapshot = _config_snapshot(cfg)
    print("\nLOCKED CREMA-D FINAL EXTRAS PROTOCOL")
    print("Paper seeds:", PAPER_SEEDS)
    print("Architectures:", PAPER_ARCHITECTURES)
    print("Requested seeds:", seeds)
    print("Requested architectures:", archs)
    print("Config SHA-256:", _config_fingerprint(snapshot))
    print("Extra methods:", EXTRA_ATTACK_NAMES)
    print("FusionBreak reference rerun: consistency check only")
    print("Generic detector scoring for extras: disabled")

    plan = [(seed, arch) for seed in seeds for arch in archs]
    for seed, arch in tqdm(plan, desc="CREMA-D final extras configs", unit="config"):
        run_one(seed, arch, cfg, device)
        write_outputs(cfg)

    write_outputs(cfg)
    print("All requested final-consistent CREMA-D extras completed.")


if __name__ == "__main__":
    main()
