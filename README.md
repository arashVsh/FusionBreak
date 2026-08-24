# FusionBreak Reviewer Artifact

This repository contains the anonymous reproducibility artifact for **FusionBreak: Attacking Multimodal Fusion While Preserving Unimodal Decisions**.

FusionBreak studies a stricter multimodal adversarial failure condition: the fused classifier becomes incorrect while the corresponding unimodal classifiers remain correct.

The repository contains two dataset-specific implementations:

```text
FusionBreak/
├── README.md
├── requirements.txt
├── .gitignore
├── FusionBreak-RAVDESS_Reviewer/
└── FusionBreak-CREMA-D_Reviewer/
```

Each dataset directory contains its own detailed README with dataset setup, checkpoint layout, reproduction commands, output descriptions, and paper-facing sanity checks.

## Evaluated datasets

### RAVDESS

Directory:

```text
FusionBreak-RAVDESS_Reviewer/
```

The final RAVDESS evaluation uses:

- seeds `42`, `43`, and `44`;
- four fusion architectures: `interaction`, `concat`, `gated`, and `group_gated`;
- FusionBreak;
- AdaptiveFusionBreak;
- standard fused-model PGD;
- CE-only with decision-aware candidate selection;
- audio-only FusionBreak;
- video-only FusionBreak;
- generic detector evaluation;
- the hard-label fusion-consistency defense;
- the final seed-42 ablation suite.

Main paper-facing mean FOSR:

```text
FusionBreak          77.09%
AdaptiveFusionBreak  75.99%
PGD                   0.00%
```

See `FusionBreak-RAVDESS_Reviewer/README.md` for the complete reproduction procedure.

### CREMA-D

Directory:

```text
FusionBreak-CREMA-D_Reviewer/
```

The final CREMA-D evaluation uses the same three seeds and four fusion architectures, together with the same primary attacks and controlled comparisons.

Main paper-facing mean FOSR:

```text
FusionBreak          97.67%
AdaptiveFusionBreak  97.64%
PGD                   0.00%
```

See `FusionBreak-CREMA-D_Reviewer/README.md` for the complete reproduction procedure.

## Installation

Python 3.10-3.12 is recommended.

Create and activate a virtual environment:

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

The root `requirements.txt` installs the requirements of both dataset-specific subprojects.

For GPU execution, install a PyTorch build appropriate for the machine's CUDA driver if the default package installation is not suitable.

## Datasets

The datasets are **not redistributed** in this repository.

Reviewers should obtain RAVDESS and CREMA-D from their public sources and place them inside the corresponding dataset project as described in the dataset-specific README files.

## Checkpoints

The reviewer artifact is designed to use the exact frozen checkpoints used for the reported experiments.

The final paper runners do not silently train missing checkpoints. Required checkpoint names and locations are documented in each dataset-specific README.

## Reproducing the results

Run each experiment from inside its corresponding dataset directory.

RAVDESS:

```powershell
cd FusionBreak-RAVDESS_Reviewer
$env:FUSIONGAP_DEVICE="cuda:0"
python .\run_ravdess_usenix_final_consistent.py
```

CREMA-D:

```powershell
cd FusionBreak-CREMA-D_Reviewer
$env:FUSIONGAP_DEVICE="cuda:0"
python .\run_cremad_usenix_final_consistent.py
```

Additional controlled experiments, ablations, clean-accuracy audits, expected output files, and consistency checks are documented in the respective subproject README files.

## Experimental scope

The reported paper protocol uses:

- white-box attacks;
- processed audio/video model-input space;
- modality-specific `L_inf` perturbation constraints;
- the same eligible samples and frozen checkpoints for matched comparisons;
- utterance-level dataset partitions rather than speaker-disjoint partitions.

The audio perturbation operates on the processed time-frequency representation rather than directly on the raw waveform.

## Repository organization

The two dataset directories are intentionally kept separate because their preprocessing, checkpoints, result files, and reproduction commands differ.

The root README provides only the shared overview. Dataset-specific technical details should remain in the two subfolder README files rather than being duplicated here.

## Anonymous artifact

This repository is intended for anonymous review. Before publishing it for double-blind review, verify that it contains no author names, institutional identifiers, personal usernames, local absolute paths, Git metadata revealing identity, or tracking URLs.
