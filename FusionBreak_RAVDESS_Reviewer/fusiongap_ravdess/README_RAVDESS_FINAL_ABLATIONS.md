# RAVDESS final-consistent ablation rerun

This companion runner reruns the RAVDESS ablation experiments reported in the FusionBreak paper using the exact final RAVDESS base protocol and existing final checkpoints.

## Why rerun these

The final main RAVDESS evaluation now uses the controlled three-seed runner in `run_ravdess_usenix_final_consistent.py`. The old manuscript ablations were already close to the same settings, but they were produced in an earlier execution pipeline. This runner removes that remaining provenance mismatch.

## What stays identical to the final main run

- seed 42 for the ablation study
- the same four fusion architectures
- the same RAVDESS split/cache
- the same existing audio, video, and fusion checkpoints
- the same eligible samples
- default 80 steps and 10 restarts
- default epsilons 0.08 audio and 0.06 video
- default step sizes 0.004 audio and 0.003 video
- the same FusionBreak objective and candidate ranking

The runner also reproduces the final runner's detector-fitting stage before the attacks because that stage affects the RNG state. It verifies that its `Full FusionBreak` counts exactly reproduce the already completed final-consistent seed-42 result for every architecture. If that check fails, it stops.

## Run

Place both files in the RAVDESS project root beside:

```text
run_ravdess_usenix_final_consistent.py
fusiongap_ravdess/
RAVDESS/
```

PowerShell:

```powershell
$env:FUSIONGAP_DEVICE="cuda:1"   # use cuda:0 if needed
Remove-Item Env:FUSIONGAP_ARCHS -ErrorAction SilentlyContinue
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_ravdess_usenix_final_ablations.py
```

Optional one-architecture validation first:

```powershell
$env:FUSIONGAP_ARCHS="concat"
python .\run_ravdess_usenix_final_ablations.py
```

After it prints `[reference PASS]`, clear the subset and run all architectures.

## Outputs

Per architecture:

```text
usenix_final_ablation_v1_ravdess_seed42_{arch}.json
```

Final summaries:

```text
usenix_final_ablation_summary_v1_ravdess.csv
usenix_final_ablation_aggregate_v1_ravdess.csv
usenix_final_ablation_paper_tables_v1_ravdess.json
usenix_final_ablation_audit_v1_ravdess.json
```

Use the values in `usenix_final_ablation_paper_tables_v1_ravdess.json` to replace the old manuscript ablation tables if any value changes.
