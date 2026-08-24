from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _safe_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _pct(x: float | None) -> float | None:
    return None if x is None else 100.0 * float(x)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def write_summary(log_dir: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("fusion_gap_ravdess_seed*_ravdess_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attack = data.get("attack_results", {}) if isinstance(data.get("attack_results", {}), dict) else {}
        row = {
            "file": path.name,
            "seed": data.get("seed"),
            "architecture": data.get("fusion_architecture"),
            "train_samples": data.get("train_samples"),
            "val_samples": data.get("val_samples"),
            "test_samples": data.get("test_samples"),
            "audio_acc": _safe_get(data, ["clean_eval", "audio_acc"]),
            "video_acc": _safe_get(data, ["clean_eval", "video_acc"]),
            "fusion_acc": _safe_get(data, ["clean_eval", "fusion_acc"]),
            "all_three_correct_rate": _safe_get(data, ["clean_eval", "all_three_correct_rate"]),
            "all_three_correct_count": _safe_get(data, ["clean_eval", "all_three_correct_count"]),
            "attacked_clean_samples": attack.get("attacked_clean_samples"),
            "fg_fusion_ASR": _safe_get(attack, ["fusion_gap_rates", "fusion_ASR"]),
            "fg_audio_preservation": _safe_get(attack, ["fusion_gap_rates", "audio_preservation_rate"]),
            "fg_video_preservation": _safe_get(attack, ["fusion_gap_rates", "video_preservation_rate"]),
            "fg_FOSR": _safe_get(attack, ["fusion_gap_rates", "FOSR"]),
            "pgd_fusion_ASR": _safe_get(attack, ["standard_pgd_rates", "fusion_ASR"]),
            "pgd_audio_preservation": _safe_get(attack, ["standard_pgd_rates", "audio_preservation_rate"]),
            "pgd_video_preservation": _safe_get(attack, ["standard_pgd_rates", "video_preservation_rate"]),
            "pgd_FOSR": _safe_get(attack, ["standard_pgd_rates", "FOSR"]),
        }
        rows.append(row)

    if not rows:
        print("[summary] No RAVDESS result JSON files found yet.")
        return

    run_csv = artifact_dir / "ravdess_summary_by_run.csv"
    with run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arch[str(row["architecture"])].append(row)
        by_seed[str(row["seed"])].append(row)

    def aggregate(group_rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
        out: dict[str, Any] = {"group": group_name, "runs": len(group_rows)}
        out["attacked_clean_samples_total"] = sum(int(r.get("attacked_clean_samples") or 0) for r in group_rows)
        for key in [
            "audio_acc", "video_acc", "fusion_acc", "all_three_correct_rate",
            "fg_fusion_ASR", "fg_audio_preservation", "fg_video_preservation", "fg_FOSR",
            "pgd_fusion_ASR", "pgd_audio_preservation", "pgd_video_preservation", "pgd_FOSR",
        ]:
            vals = [float(r[key]) for r in group_rows if r.get(key) is not None]
            m, s = _mean_std(vals)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = s
        return out

    arch_rows = [aggregate(v, k) for k, v in sorted(by_arch.items())]
    seed_rows = [aggregate(v, k) for k, v in sorted(by_seed.items())]
    overall = aggregate(rows, "overall")

    for name, table_rows in [
        ("ravdess_summary_by_architecture.csv", arch_rows),
        ("ravdess_summary_by_seed.csv", seed_rows),
    ]:
        path = artifact_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)

    (artifact_dir / "ravdess_summary_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(f"[summary] Wrote {run_csv}")
    print(f"[summary] Wrote {artifact_dir / 'ravdess_summary_by_architecture.csv'}")
    print(f"[summary] Wrote {artifact_dir / 'ravdess_summary_by_seed.csv'}")
    print(f"[summary] Wrote {artifact_dir / 'ravdess_summary_overall.json'}")
