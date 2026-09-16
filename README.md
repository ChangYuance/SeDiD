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

The corpus is **CSD-615** — 615 Mandarin dysarthric speech samples from 10 stroke
centers, each independently rated by three senior clinicians on a 4-level severity
scale. It has its own release: **https://github.com/aiot-ssmc/CSD-615**, which
publishes the individual clinician ratings (`rater_a`, `rater_b`, `rater_c`) and the
reference severity label (`median`) for every sample.

The full audio is not redistributed here — request it through the CSD-615 release.
No dataset-internal file name or layout is hard-coded in the source; the corpus is
read through environment variables:

| Variable | Meaning |
|----------|---------|
| `CSD615_DATA_LISTS` | directory holding `train/`, `val/` and `test/`, each with a `data.list` (one JSON object per line, mapping an audio key to its wav path) |
| `CSD615_LABELS` | the per-clinician rating table (`.xlsx`) |
| `CSD615_LABEL_COLUMNS` | the three annotator columns of `CSD615_LABELS`, comma-separated in annotator order (A, B, C); the third column's empty cells fall back to the first |
| `CSD615_REVIEW` | the reference-label table (`.xlsx`), used for samples that carry no individual ratings |
| `CSD615_REVIEW_COLUMN` | the reference-label column of `CSD615_REVIEW` |
| `SENSEVOICE_CKPT` | local copy of the pre-fine-tuned SenseVoice checkpoint used to initialize full fine-tuning (also not distributed) |

```bash
export CSD615_DATA_LISTS=/path/to/data_lists
export CSD615_LABELS=/path/to/per_clinician_ratings.xlsx
export CSD615_LABEL_COLUMNS="<clinician_a>,<clinician_b>,<clinician_c>"
export CSD615_REVIEW=/path/to/reference_labels.xlsx
export CSD615_REVIEW_COLUMN="<reference_column>"
export SENSEVOICE_CKPT=/path/to/sensevoice.pt
```

Column headers are dataset-specific (the rating columns are headed by the clinicians who
produced them), so they are supplied through the environment rather than hard-coded.


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
