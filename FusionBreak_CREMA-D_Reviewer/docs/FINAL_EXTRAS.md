# FusionBreak USENIX: final CREMA-D extras

This package implements the **last remaining CREMA-D experiments** for the paper under the locked 3-seed protocol.

## What it runs

Paper seeds: `42, 43, 44`

Architectures: `interaction`, `concat`, `gated`, `group_gated`

Paper-facing extra methods:

1. `CEOnlySelection`
2. `AudioOnlyFusionBreak`
3. `VideoOnlyFusionBreak`

The script also reruns `FusionBreakReference`, but **only as a hard consistency check**. Its counts must exactly reproduce the already-completed `usenix_final_controlled_v1_cremad_seed{seed}_{arch}.json` FusionBreak counts. If the reference does not match, the script aborts instead of producing a usable result.

## Why this version is final-consistent

The old CREMA-D extras were generated outside the final controlled runner. The final main experiment fits the detector pack before attacking, which can affect RNG state. This runner reproduces that same pre-attack sequence even though the three extras do not need generic detector scores.

For every attack batch, FusionBreak reference and all three extras start from the same captured RNG state. The runner also verifies and restores the RNG state expected after the final-main PGD path before continuing to the next batch.

The script checks:

- final config SHA-256
- final checkpoint SHA-256 hashes
- train, validation, and test sizes
- exact eligible sample count
- exact clean consistency-defense counts
- exact FusionBreak reference counts
- full eligible-set coverage
- audio/video `L_inf` budgets
- zero video perturbation for audio-only FusionBreak
- zero audio perturbation for video-only FusionBreak

## Required project files

Place `run_cremad_usenix_final_extras.py` in the same project root as:

```text
run_cremad_usenix_final_consistent.py
run_cremad_controlled_full_vs_adaptive.py
fusiongap_cremad/
CREMA-D/
```

The completed final main JSON files must already exist in the normal log directory.

## Smoke test

On your single-GPU RTX 3070 Laptop GPU:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_SEEDS="42"
$env:FUSIONGAP_ARCHS="concat"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_cremad_usenix_final_extras.py
```

The important line is:

```text
[reference PASS] seed=42, arch=concat: FusionBreak exactly reproduces final main counts
```

Do not continue with the full run if the reference check fails.

## Full locked run

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_SEEDS="42,43,44"
$env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_cremad_usenix_final_extras.py
```

Completed configurations are skipped automatically. Do not set `FUSIONGAP_FORCE=1` when resuming.

## Final files to send back

After the full run, send these files back to ChatGPT:

```text
usenix_final_extras_summary_v1_cremad.csv
usenix_final_extras_aggregate_v1_cremad.csv
usenix_final_extras_paper_table_v1_cremad.json
usenix_final_extras_audit_v1_cremad.json
```

The audit should report:

```json
"complete": true
```

and:

```json
"all_reference_fusionbreak_exact_match": true
```

## No more experiments after this

Once this audit is complete and clean, the experimental phase should be frozen. The remaining work is paper writing, replacing obsolete 5-seed numbers, updating figures/tables, and tightening the threat model and limitations.
