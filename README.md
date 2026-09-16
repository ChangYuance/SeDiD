# SeDiD

**SeDiD**: Learning from Clinician Severity Disagreement for Speech-Based Dysarthria Detection.

This repository provides the reference implementation for the ICASSP 2027 submission "Learning from Clinician Severity Disagreement for Speech-Based Dysarthria Detection".

## Method

The model fine-tunes the full SenseVoice encoder (221M parameters) on raw speech and combines four objectives:

1. **SeDiD multi-annotator fusion** — an instance-dependent confusion matrix `M^k(x)` is generated per utterance for each annotator `k`, and each annotator's prediction is obtained as `q_k = pᵀ M^k(x)`, where `p` is the 4-class model output. The cross-entropy between `q_k` and annotator `k`'s label `y_k` aligns the model with each annotator's individual labeling behavior, in contrast to global (input-independent) confusion matrices.
2. **Focal loss** on the D2 binary (severe vs. non-severe) head for class imbalance.
3. **CTC auxiliary loss** on the counting task (phonetic regularization).
4. **Trace regularization** on the confusion matrices.

## Results (5-fold, seed 42)

| Method | Pos. F1 | Macro F1 | 4-cls Acc |
|--------|:-------:|:--------:|:---------:|
| *Single-annotator* | | | |
| MFCCStats | 0.4839 | 0.6647 | 0.7137 |
| MFCCFusion | 0.5248 | 0.6033 | 0.5362 |
| WhisperProbe-Mid | 0.6806 | 0.7861 | 0.7740 |
| CoarseToFine | 0.7225 | 0.8178 | -- |
| WhisperFT | 0.6752 | 0.7825 | 0.7724 |
| *Multi-annotator* | | | |
| CrowdLayer | 0.7962 | 0.8659 | 0.8375 |
| CrowdAttention | 0.7760 | 0.8527 | 0.8214 |
| COINNet | 0.8071 | 0.8723 | 0.8473 |
| LFCx | 0.8048 | 0.8667 | 0.8099 |
| Tanno | 0.8011 | 0.8689 | 0.8392 |
| **Ours (SeDiD)** | **0.8331** | **0.8882** | **0.8489** |

## Ablation (5-fold, seed 42)

| Variant | Pos. F1 | Macro F1 | 4-cls Acc |
|---------|:-------:|:--------:|:---------:|
| Ours (full) | 0.8331 | 0.8882 | 0.8489 |
| w/o Multi-Annotator | 0.8022 | 0.8687 | 0.8327 |
| w/o CTC | 0.8021 | 0.8693 | 0.8408 |
| w/o Focal | 0.8035 | 0.8707 | 0.8457 |

## Repository structure

```
model.py                  # SenseVoiceMultiAnnotator: SeDiD fusion model (V2)
dys_model_sensevoice.py   # Vendored SenseVoice encoder module (required import)
data_loader.py            # Audio + multi-annotator label loading
run_5fold.py              # 5-fold training & evaluation for V2 and baselines
fold_indices.json         # The exact 5-fold split used in the paper
requirements.txt          # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Requires CUDA (a single GPU with ~24 GB memory; effective batch size 16 via batch size 4 + 4 gradient accumulation steps).

## Data access

The CSD-615 dataset (615 Mandarin dysarthric speech samples recorded from 10 stroke centers, annotated by 3 clinicians on a 4-level severity scale) is **clinical data and is not publicly distributed**. The code expects the following local layout, configurable via environment variables:

```
CSD615_ROOT/
├── training_data_v2/            # CSD615_TRAINING_DATA
│   └── <sample_id>.wav
├── data/
│   └── labels_三位医生标记.xlsx   # CSD615_LABELS (annotator labels)
└── results/
    └── doctor_review_summary.xlsx  # CSD615_REVIEW (consensus labels)
```

The pre-fine-tuned SenseVoice checkpoint used as the initialization for full fine-tuning is also not distributed; set `SENSEVOICE_CKPT` to a local copy.

## Usage

Reproduce the V2 (full) result (canonical configuration: seed 42, learning rate `5e-5`, batch size 4, gradient accumulation 4, 100 max epochs, early stopping patience 20, warmup ratio 0.2, focal `gamma=2.0`, `lambda_conf=1.0`):

```bash
for fold in 0 1 2 3 4; do
  python run_5fold.py --method V2 --fold $fold
done
```

`--seed` defaults to the canonical seed 42 (recorded as `None` in `analytics/results_5fold.json`). Results for each fold are appended to `analytics/results_5fold.json`.

### Baselines and ablations

```bash
# Multi-annotator baselines (each reported under its proposed objective)
python run_5fold.py --method CrowdLayer --fold 0 --focal_gamma 0
python run_5fold.py --method Tanno      --fold 0

# Ablations of V2
python run_5fold.py --method V2 --fold 0 --ablation w/o_focal
python run_5fold.py --method V2 --fold 0 --ablation w/o_ctc
```

## Citation

The paper is currently under review. If you use this code in your research,
please cite it once the final venue and reference are available.
