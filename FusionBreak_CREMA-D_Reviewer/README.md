# FusionBreak — CREMA-D Reviewer Artifact

This directory contains the CREMA-D implementation and paper-facing outputs
for the final FusionBreak evaluation. It is restricted to the frozen
three-seed paper protocol; older five-seed pipelines, pre-final controls,
caches, and development outputs are excluded.

## Final protocol

The final CREMA-D evaluation uses:

- seeds `42`, `43`, and `44`;
- fusion architectures `interaction`, `concat`, `gated`, and `group_gated`;
- FusionBreak;
- AdaptiveFusionBreak;
- standard fused-model PGD;
- CE-only with the same decision-aware search;
- audio-only FusionBreak;
- video-only FusionBreak;
- the generic detector evaluation reported in the paper;
- the hard-label fusion-consistency defense.

`run_cremad_usenix_final_consistent.py` is the final main runner.

`run_cremad_usenix_final_extras.py` is the final consistency-locked runner for
the three paper-facing controlled extras:
`CEOnlySelection`, `AudioOnlyFusionBreak`, and `VideoOnlyFusionBreak`.

`run_cremad_controlled_full_vs_adaptive.py` is retained because the final
runners import the exact FusionBreak/AdaptiveFusionBreak implementation,
projection functions, RNG helpers, and checkpoint loader from it.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── fusiongap_cremad/
├── run_cremad_controlled_full_vs_adaptive.py
├── run_cremad_usenix_final_consistent.py
├── run_cremad_usenix_final_extras.py
├── compute_cremad_clean_accuracies.py
└── fusion_gap_cremad_modular_outputs/
    ├── logs/
    │   ├── usenix_extras_v1_cremad_seed*_*.json
    │   ├── usenix_final_controlled_v1_cremad_seed*_*.json
    │   ├── usenix_final_extras_v1_cremad_seed*_*.json
    │   ├── usenix_final_controlled_summary_v1_cremad.*
    │   ├── usenix_final_extras_*_v1_cremad.*
    │   └── usenix_clean_accuracies_v1_cremad.json
    └── large_files/
        └── checkpoints/
```

Packed tensor caches are omitted because they can be rebuilt.

## Dataset

CREMA-D is not redistributed with this artifact.

Obtain the public dataset and place it under the project root. The recommended
layout is:

```text
CREMA-D/
├── AudioWAV/
│   └── *.wav
└── VideoFlash/
    └── *.flv
```

The scanner pairs audio and video by their common utterance stem. If a local
copy uses a different dataset-folder name, update the corresponding field in
`fusiongap_cremad/config.py`.

The final deduplicated preprocessing produces 7,442 paired utterances. For
each paper seed, the final split contains 5,211 training, 744 validation, and
1,487 test utterances.

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
OpenCV must be able to decode the CREMA-D video files.

The final preprocessing uses SciPy for WAV loading/resampling and OpenCV for
video decoding; `torchaudio` is not required by the final CREMA-D path.

## Final checkpoints

The final main runner never trains missing checkpoints. The artifact therefore
includes the exact final controlled checkpoints:

```text
audio_cremad_v2_seed{seed}.pt
video_cremad_v2_seed{seed}.pt
fusion_cremad_v2_seed{seed}_{architecture}.pt
```

for seeds `42`, `43`, and `44` and all four fusion architectures.

The final runners verify checkpoint/configuration consistency before accepting
paper-facing results.

## Why `usenix_extras_v1_...` files are present

Files named:

```text
usenix_extras_v1_cremad_seed{seed}_{architecture}.json
```

are retained **only as consistency anchors** required by the final main
runner. They preserve the frozen test-set/eligibility state used by the final
evaluation.

Their older attack metrics are **not paper-facing results**.

Paper-facing main files are:

```text
usenix_final_controlled_v1_cremad_seed{seed}_{architecture}.json
```

Paper-facing controlled-extra files are:

```text
usenix_final_extras_v1_cremad_seed{seed}_{architecture}.json
```

## Reproduce the final main CREMA-D evaluation

From the artifact root:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_SEEDS="42,43,44"
$env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_cremad_usenix_final_consistent.py
```

Linux/macOS:

```bash
FUSIONGAP_DEVICE=cuda:0 \
FUSIONGAP_SEEDS=42,43,44 \
FUSIONGAP_ARCHS=interaction,concat,gated,group_gated \
python run_cremad_usenix_final_consistent.py
```

The consistency guard must pass before results are accepted. It checks the
frozen attack settings, test size, eligible count, attacked eligible count,
clean validation/test consistency counts, and final checkpoints.

## Reproduce the final controlled extras

Run this after the final main result files are present:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
$env:FUSIONGAP_SEEDS="42,43,44"
$env:FUSIONGAP_ARCHS="interaction,concat,gated,group_gated"
Remove-Item Env:FUSIONGAP_FORCE -ErrorAction SilentlyContinue
python .\run_cremad_usenix_final_extras.py
```

The runner also reruns FusionBreak as a reference check. Its counts must
exactly reproduce the completed final-main FusionBreak counts; otherwise the
runner aborts.

The final extras audit should report:

```json
"complete": true
```

and:

```json
"all_reference_fusionbreak_exact_match": true
```

## Audit clean accuracy

To recompute clean test performance using the exact final checkpoints:

```powershell
$env:FUSIONGAP_DEVICE="cuda:0"
python .\compute_cremad_clean_accuracies.py
```

Expected 12-configuration aggregate:

```text
Audio clean accuracy : 46.92 +/- 3.05%
Video clean accuracy : 45.62 +/- 4.06%
Fusion clean accuracy: 64.25 +/- 4.93%

Eligible count: 299.83 +/- 42.41
Eligible rate : 20.16 +/- 2.85%
Eligible-rate range: 15.13%-23.67%
```

The script should end with:

```text
Eligibility consistency guard: PASS
```

## Paper-result sanity check

Main paper-facing mean FOSR:

```text
FusionBreak          97.67%
AdaptiveFusionBreak  97.64%
PGD                   0.00%
```

The structured JSON/CSV outputs are authoritative; the values above are only
a quick sanity check.

## Reproducibility notes

- Perturbation constraints are applied in processed model-input space.
- CREMA-D audio perturbations operate on the log-magnitude time-frequency
  representation rather than the raw waveform.
- Audio is projected into the specified L-infinity ball without an artificial
  `[0,1]` clamp; video is projected into the L-infinity ball and valid
  video-input range.
- Paper-facing comparisons use the same fixed checkpoints and eligible samples
  within a configuration.
- Dataset partitions are utterance-level rather than speaker-disjoint.

## Intentionally omitted

The reviewer artifact excludes:

- the CREMA-D dataset;
- `.venv/`, `__pycache__/`, and other Python caches;
- packed tensor caches and detector caches;
- old five-seed publication scripts;
- pre-final extras/control runners;
- old ablation manifests and unrelated ablation outputs;
- the development project ZIP;
- obsolete result files and machine-specific logs.

These omissions do not alter the final frozen paper protocol.
