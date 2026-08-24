# RAVDESS final-consistent ablations, V2

This is the corrected ablation runner for the FusionBreak USENIX experiments.

## What V2 fixes

V1 correctly reproduced the final main FusionBreak reference, but its `final_iterate_selection` ablation still used the hard decision-aware candidate score to choose among restart terminal points. That did not fully remove decision-aware candidate selection.

V2 fixes only that issue.

For `final_iterate_selection`, V2 now:

- never evaluates intermediate candidates for selection
- never uses the hard decision-aware candidate score
- does not allow the clean sample as a fallback candidate
- runs the same full FusionBreak objective for the same steps and restarts
- after each restart, considers only that restart's terminal iterate
- selects the terminal iterate with the highest value of the same continuous FusionBreak objective used for optimization

This cleanly isolates the effect of intermediate decision-aware candidate selection.

## Consistency safeguards

The runner still:

- uses seed 42
- uses the final RAVDESS checkpoints
- uses the exact final eligible pool
- uses the same preprocessing and attack settings as the final main runner
- matches the final runner's pre-attack detector fitting and RNG progression
- verifies Full FusionBreak exactly matches the completed final main counts
- aborts on checkpoint, eligible-pool, reference-count, or perturbation-budget mismatches

## Efficient reuse of your completed concat V1 run

Your completed concat V1 run is still useful. If the V1 JSON is present in the normal logs folder and its reference check passed, V2 reuses all 12 unaffected rows and reruns only the corrected `final_iterate_selection` row.

Therefore the concat correction should be much faster than repeating the full 28-minute run.

V2 writes new files with `v2` in their names. It does not overwrite the V1 results.

## First run: corrected concat check

Place `run_ravdess_usenix_final_ablations.py` in the project root beside `run_ravdess_usenix_final_consistent.py`.

PowerShell:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_ARCHS="concat"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue

python .\run_ravdess_usenix_final_ablations.py
```

If the V1 concat result is in the logs folder, you should see a message similar to:

```text
[reuse v1] concat: reusing 12 verified rows; rerunning corrected final_iterate_selection only.
```

At the end you must still see:

```text
[reference PASS] concat
```

The new result will be:

```text
usenix_final_ablation_v2_ravdess_seed42_concat.json
```

Please send that V2 concat JSON back before running the other three architectures. We should inspect the corrected final-iterate result first.

## After concat is approved

Run all remaining architectures:

```powershell
Remove-Item Env:FUSIONGAP_ARCHS -ErrorAction SilentlyContinue
python .\run_ravdess_usenix_final_ablations.py
```

The existing V2 concat result will be skipped. The other three architectures will run normally.

## Final V2 outputs

```text
usenix_final_ablation_summary_v2_ravdess.csv
usenix_final_ablation_aggregate_v2_ravdess.csv
usenix_final_ablation_paper_tables_v2_ravdess.json
usenix_final_ablation_audit_v2_ravdess.json
```

Use only the V2 paper-table JSON for the final manuscript ablation values.
