# SeDiD

**SeDiD**: Learning from Clinician Severity Disagreement for Speech-Based Dysarthria Detection.

This repository provides the reference implementation for the ICASSP 2027 submission "Learning from Clinician Severity Disagreement for Speech-Based Dysarthria Detection".

## Method

The model fine-tunes the full SenseVoice encoder (221M parameters) on raw speech and combines three objectives (paper Eq. 6, `L = L_CE + L_focal + L_CTC`):

1. **Individual clinician supervision (CICM)** — an instance-dependent confusion matrix `C^k(x)` is generated per utterance for each clinician `k`, and that clinician's predicted rating distribution is `q_k = pᵀ C^k(x)`, where `p` is the 4-class model output. The cross-entropy between `q_k` and clinician `k`'s rating `y_k` aligns the model with each clinician's individual labeling behavior, in contrast to a fixed clinician-level confusion matrix.
2. **Consensus-guided focal supervision** (`L_focal`) — a focal loss on the consensus label (the median of the three ratings), weighted by how unanimous the clinicians were, together with a binary focal loss on the normal-vs-dysarthric decision. Inverse-frequency class weights compensate for the imbalance.
3. **CTC auxiliary supervision** (`L_CTC`) — a CTC branch over the known transcript, used during training only.

The detection task is binary normal-vs-dysarthric (`BIN_THRESH = 1`, i.e. severity 0 vs `{1,2,3}`); the 4-class severity accuracy is reported as a secondary metric.

## Results (5-fold, seed 42)

Mean over the five folds; per-fold numbers are appended to
`analytics/results_5fold.json` by the run itself.

| Method | Pos. F1 | Macro F1 | 4-cls Acc |
|--------|:-------:|:--------:|:---------:|
| **Ours (SeDiD)** | **0.8331** | **0.8882** | **0.8489** |

`run_5fold.py` also contains the nine baseline implementations compared against
in the paper; only the SeDiD configuration is documented below.

## Ablation (5-fold, seed 42)

| Variant | `--ablation` | Pos. F1 | Macro F1 | 4-cls Acc |
|---------|--------------|:-------:|:--------:|:---------:|
| Ours (full) | — | 0.8331 | 0.8882 | 0.8489 |
| w/o CICM | `w/o_multiannotator` | 0.8022 | 0.8687 | 0.8327 |
| w/o L_CTC | `w/o_ctc` | 0.8021 | 0.8693 | 0.8408 |
| w/o L_focal | `w/o_focal` | 0.8035 | 0.8707 | 0.8457 |
| Global C^k | `global_cm` | 0.8123 | 0.8753 | 0.8457 |

`Global C^k` replaces the utterance-dependent confusion matrix `C^k(x)` with a
fixed matrix for each clinician.

## Repository structure

```
model.py                  # SenseVoiceMultiAnnotator: SeDiD fusion model
comparison_models.py      # Baseline models compared against in the paper
dys_model_sensevoice.py   # Vendored SenseVoice encoder module (required import)
data_loader.py            # Audio + multi-annotator label loading
train_utils.py            # Device selection, LR schedule
run_5fold.py              # 5-fold training & evaluation
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

The three annotator label columns of `labels_三位医生标记.xlsx` are headed by the
annotators' names, which are deliberately not committed here. Set
`CSD615_LABEL_COLUMNS` to those three header names, comma-separated in
annotator order (`A,B,C`; the third column is the one that falls back to `A`
when empty):

```bash
export CSD615_LABEL_COLUMNS="<col_a>,<col_b>,<col_c>"
```


## Usage

Reproduce the SeDiD result (canonical configuration: seed 42, learning rate `5e-5`, batch size 4, gradient accumulation 4, 100 max epochs, early stopping patience 20, warmup ratio 0.2, focal `gamma=2.0`, `lambda_conf=1.0`):

```bash
for fold in 0 1 2 3 4; do
  python run_5fold.py --method V2 --fold $fold
done
```

`--seed` defaults to the canonical seed 42 (recorded as `None` in `analytics/results_5fold.json`). Results for each fold are appended to `analytics/results_5fold.json`.

An ablation row is reproduced by adding its `--ablation` value from the table above, e.g.:

```bash
python run_5fold.py --method V2 --fold 0 --ablation global_cm
```

## Citation

The paper is currently under review. If you use this code in your research,
please cite it once the final venue and reference are available.
