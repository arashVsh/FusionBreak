# FusionBreak — RAVDESS Reviewer Artifact

This directory contains the RAVDESS implementation and paper-facing outputs
for the final FusionBreak evaluation. It is intentionally limited to the
frozen paper protocol; older development runners, caches, and obsolete results
are excluded.

## Final protocol

The final RAVDESS evaluation uses:

- seeds `42`, `43`, and `44`;
- fusion architectures `interaction`, `concat`, `gated`, and `group_gated`;
- FusionBreak;
- AdaptiveFusionBreak;
- standard fused-model PGD;
- CE-only with the same decision-aware candidate search;
- audio-only FusionBreak;
- video-only FusionBreak;
- the four generic detectors reported in the paper;
- the hard-label fusion-consistency defense.

All six main attack/control methods use the same eligible samples and
checkpoints within each seed/architecture configuration.

The paper ablations are the final **V2** RAVDESS ablations using seed `42`
and all four fusion architectures.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── configs/
├── fusiongap_ravdess/
├── run_ravdess_usenix_final_consistent.py
├── run_ravdess_usenix_final_ablations.py
├── audit_ravdess_clean_accuracies_fixed.py
├── plot_final_results.py
└── fusion_gap_ravdess_multimodal_outputs/
    ├── logs/
    │   ├── usenix_final_allmethods_v1_ravdess_seed*_*.json
    │   ├── usenix_final_*_v1_ravdess.*
    │   ├── usenix_final_ablation_v2_ravdess_seed42_*.json
    │   ├── usenix_final_ablation_*_v2_ravdess.*
    │   └── usenix_clean_accuracies_audit_v1_ravdess.json
    └── large_files/
        └── checkpoints/
```

Packed tensor caches are omitted because they can be rebuilt from the public
dataset.

## Dataset

RAVDESS is not redistributed with this artifact.

Obtain the public RAVDESS speech data and arrange the paired files as expected
by the released configuration:

```text
RAVDESS/
├── Audio/
│   ├── Actor_01/*.wav
│   ├── ...
│   └── Actor_24/*.wav
└── Video/
    ├── Actor_01/*.mp4
    ├── ...
    └── Actor_24/*.mp4
```

The final pipeline uses the speech recordings and pairs audio-only and
audio-video files by RAVDESS filename metadata.

## Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU execution, use a PyTorch build compatible with the local CUDA driver.
OpenCV must be able to decode the RAVDESS video files.

The final preprocessing path reads WAV files through SciPy rather than
`torchaudio`. `torchvision` and `torchaudio` remain in `requirements.txt`
only because the retained modular package contains older import paths that may
be loaded during package initialization.

## Final checkpoints

The final paper runner **never trains missing checkpoints**. The artifact
therefore includes the exact paper checkpoints:

```text
audio_seed{seed}.pt
video_seed{seed}.pt
fusion_seed{seed}_{architecture}.pt
```

for seeds `42`, `43`, and `44` and all four fusion architectures.

The runner records SHA-256 hashes for the checkpoints and aborts if required
files are missing.

## Reproduce the main RAVDESS evaluation

From the artifact root:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_SEEDS="42,43,44"
$env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_ravdess_usenix_final_consistent.py
```

Linux/macOS:

```bash
FUSIONGAP_DEVICE=cuda:0 \
FUSIONGAP_SEEDS=42,43,44 \
FUSIONGAP_ARCHS=interaction,concat,gated,group_gated \
python run_ravdess_usenix_final_consistent.py
```

Existing completed seed × architecture files are skipped unless
`FUSIONGAP_FORCE=1` is deliberately set.

Paper-facing per-configuration results are:

```text
fusion_gap_ravdess_multimodal_outputs/logs/
    usenix_final_allmethods_v1_ravdess_seed{seed}_{architecture}.json
```

The runner also writes paper-level aggregate summaries and a consistency audit.

## Reproduce the final RAVDESS ablations

The final ablation runner is **V2**:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
python .\run_ravdess_usenix_final_ablations.py
```

Final ablation outputs are:

```text
usenix_final_ablation_v2_ravdess_seed42_{architecture}.json
usenix_final_ablation_summary_v2_ravdess.csv
usenix_final_ablation_aggregate_v2_ravdess.csv
usenix_final_ablation_paper_tables_v2_ravdess.json
usenix_final_ablation_audit_v2_ravdess.json
```

V1 ablation results are not paper-facing. V2 corrects the
final-iterate-selection definition.

## Audit clean accuracy

The final result JSONs already contain the clean test metrics. To aggregate
them independently:

```bash
python audit_ravdess_clean_accuracies_fixed.py
```

Expected 12-configuration aggregate:

```text
Audio clean accuracy : 47.22 +/- 3.81%
Video clean accuracy : 33.49 +/- 5.55%
Fusion clean accuracy: 63.12 +/- 5.07%

Eligible count: 33.25 +/- 4.11
Eligible rate : 15.39 +/- 1.91%
Eligible-rate range: 13.89%-18.52%
```

The audit should end with:

```text
Paper eligibility consistency: PASS
```

## Paper-result sanity check

Main paper-facing mean FOSR:

```text
FusionBreak          77.09%
AdaptiveFusionBreak  75.99%
PGD                   0.00%
```

The structured JSON/CSV outputs are authoritative; the values above are only
a quick sanity check.

## Reproducibility notes

- Perturbation constraints are applied in processed model-input space.
- The audio attack operates on the time-frequency representation, not directly
  on raw waveform samples.
- All main methods attack the full eligible set for each configuration.
- The final runner records configuration fingerprints, checkpoint hashes,
  perturbation norms, eligible-set coverage, detector results, and
  consistency-defense results.
- Dataset partitions are utterance-level rather than speaker-disjoint.

## Intentionally omitted

The reviewer artifact excludes:

- the RAVDESS dataset;
- `.venv/`, `__pycache__/`, and other Python caches;
- packed tensor caches;
- old development/selection/five-seed runners;
- old detector-only runners;
- the buggy V1 ablation backup;
- obsolete result files and machine-specific logs.

These omissions do not change the final paper protocol.
