#!/usr/bin/env python3
"""Load raw audio and the CSD-615 clinician ratings, and provide DataLoaders.

The corpus is read through environment variables so that no dataset-internal
file name or directory layout is hard-coded here: CSD615_DATA_LISTS points at
the split files, CSD615_LABELS at the per-clinician rating table and
CSD615_REVIEW at the reference-label table (see README.md).

Per sample: speech [B, T_max] @16kHz, speech_lengths [B], labels [B, 3]
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
import torch
import torchaudio
import torchaudio.functional as F_aug
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# CSD-615 is clinical data collected from 10 stroke centers and is not
# publicly distributed. Point the CSD615_* environment variables at your local
# copy of the dataset (see README.md).
TARGET_SR = 16000
SOURCE_SR = 48000


def _env_path(name: str, what: str) -> Path:
    """Resolve a dataset path from the environment; no default is baked in."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Set {name} to {what} (see README.md).")
    return Path(val)


def _load_split_metadata() -> dict[str, dict]:
    """Read key -> {wav, split} for every sample from {split}/data.list."""
    lists_dir = _env_path("CSD615_DATA_LISTS", "the directory holding the split files")
    meta = {}
    for split in ["train", "val", "test"]:
        with open(lists_dir / split / "data.list") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    meta[row["key"]] = {"wav": row["wav"], "split": split}
    return meta


def _load_doctor_labels() -> pd.DataFrame:
    """Read the per-clinician rating table.

    The three annotator label column headers are named after the clinicians who
    produced them, so they are not hard-coded here -- set CSD615_LABEL_COLUMNS
    to the three header names, comma-separated in annotator order.
    """
    cols = [c.strip() for c in os.environ.get("CSD615_LABEL_COLUMNS", "").split(",") if c.strip()]
    if len(cols) != 3:
        raise RuntimeError(
            "Set CSD615_LABEL_COLUMNS to the three annotator label column names, "
            "comma-separated in annotator order (see README.md)."
        )
    col_a, col_b, col_c = cols

    df = pd.read_excel(_env_path("CSD615_LABELS", "the per-clinician rating table"))
    df["key"] = df["ID"].astype(str)
    df["label_a"] = df[col_a].astype(int)
    df["label_b"] = df[col_b].astype(int)
    df["label_c"] = df[col_c].fillna(df[col_a]).astype(int)
    return df[["key", "label_a", "label_b", "label_c", "split"]]


def _load_reference_labels() -> dict[str, int]:
    """Read sample ID -> reference label from the reference-label table.

    The reference-label column is dataset-specific; set CSD615_REVIEW_COLUMN to
    its header name.
    """
    col_name = os.environ.get("CSD615_REVIEW_COLUMN", "").strip()
    if not col_name:
        raise RuntimeError(
            "Set CSD615_REVIEW_COLUMN to the reference-label column name of "
            "CSD615_REVIEW (see README.md)."
        )
    wb = openpyxl.load_workbook(_env_path("CSD615_REVIEW", "the reference-label table"))
    ws = wb.active
    header = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    if col_name not in header:
        raise RuntimeError(f"Column {col_name!r} not found in CSD615_REVIEW.")
    label_col = header.index(col_name) + 1

    labels = {}
    for row in range(2, ws.max_row + 1):
        sample = str(ws.cell(row=row, column=1).value)  # first column: sample ID
        label = ws.cell(row=row, column=label_col).value
        if label is not None:
            labels[sample] = int(label)
    return labels


def build_label_dataframe() -> pd.DataFrame:
    """Merge the label sources; return a DataFrame with key, split, label_a,
    label_b, label_c, wav_path."""
    meta = _load_split_metadata()
    df_doctor = _load_doctor_labels()
    reference = _load_reference_labels()

    rows = []
    for key, info in meta.items():
        wav_path = info["wav"]
        if key in set(df_doctor["key"]):
            r = df_doctor[df_doctor["key"] == key].iloc[0]
            rows.append({
                "key": key, "split": r["split"],
                "label_a": r["label_a"], "label_b": r["label_b"], "label_c": r["label_c"],
                "wav_path": wav_path,
            })
        elif key in reference:
            lbl = reference[key]
            rows.append({
                "key": key, "split": info["split"],
                "label_a": lbl, "label_b": lbl, "label_c": lbl,
                "wav_path": wav_path,
            })
        else:
            continue

    df = pd.DataFrame(rows)
    print(f"Total labeled samples: {len(df)}", flush=True)
    for split in ["train", "val", "test"]:
        cnt = (df["split"] == split).sum()
        print(f"  {split}: {cnt}", flush=True)
    return df


def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform to target sample rate. waveform: [C, T]."""
    if orig_sr == target_sr:
        return waveform
    return torchaudio.transforms.Resample(orig_sr, target_sr).to(waveform.device)(waveform)


def apply_volume_norm(waveform: torch.Tensor, target_dbfs: float = -25.0) -> torch.Tensor:
    """Normalize waveform RMS to target dBFS."""
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-10:
        return waveform
    target_amplitude = 10 ** (target_dbfs / 20)
    gain = target_amplitude / (rms + 1e-10)
    return waveform * gain


_SPEED_RESAMPLERS = {}  # cache: (int(sr*speed), sr) -> Resample

def apply_speed_perturb(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Random speed perturbation, quantized to {0.9, 1.0, 1.1} with cached Resample."""
    speed = round((0.9 + 0.2 * torch.rand(1).item()) * 10) / 10  # 0.9, 1.0, or 1.1
    if speed == 1.0:
        return waveform
    orig_len = waveform.size(-1)
    key = (int(sr * speed), sr)
    if key not in _SPEED_RESAMPLERS:
        _SPEED_RESAMPLERS[key] = torchaudio.transforms.Resample(*key)
    resampled = _SPEED_RESAMPLERS[key](waveform.unsqueeze(0)).squeeze(0)
    if resampled.size(-1) > orig_len:
        resampled = resampled[..., :orig_len]
    elif resampled.size(-1) < orig_len:
        resampled = torch.nn.functional.pad(resampled, (0, orig_len - resampled.size(-1)))
    return resampled


class AudioMultiAnnotatorDataset(Dataset):
    """Load raw audio + the three clinicians' labels. All audio is cached in memory
    on first load."""

    def __init__(self, df: pd.DataFrame, augment: bool = False, extra_augment: bool = False, use_dict_features: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.extra_augment = extra_augment
        self._resampler = torchaudio.transforms.Resample(SOURCE_SR, TARGET_SR)
        # Pre-load all audio into memory
        self._cache = self._preload_all()

        # Load dictionary features if requested
        self.use_dict_features = use_dict_features
        if use_dict_features:
            _dict_path = Path(__file__).resolve().parent / "dict_features.npz"
            _loaded = np.load(_dict_path)
            self._dict_features = {k: _loaded[k] for k in _loaded.files}

    def _preload_all(self) -> list[torch.Tensor]:
        """Preload all audio into memory (mono, 16kHz)."""
        waveforms = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            waveform, sr = torchaudio.load(row["wav_path"])  # [C, T]
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                waveform = self._resampler(waveform)
            waveforms.append(waveform.squeeze(0))  # [T]
        return waveforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._cache[idx].clone()  # [T]

        if self.augment:
            # Speed perturbation
            waveform = apply_speed_perturb(waveform, TARGET_SR)
            # Volume normalization
            waveform = apply_volume_norm(waveform, target_dbfs=-25.0)
            if self.extra_augment:
                # Gaussian noise (low SNR ~26dB = noise at 5% of signal RMS)
                noise_std = waveform.abs().mean() * 0.05
                waveform = waveform + torch.randn_like(waveform) * noise_std
                # Random time masking (zero out random segments)
                if torch.rand(1).item() < 0.5:
                    mask_len = int(waveform.size(-1) * 0.05 * torch.rand(1).item())
                    mask_start = torch.randint(0, max(1, waveform.size(-1) - mask_len), (1,)).item()
                    waveform[mask_start:mask_start + mask_len] = 0.0

        labels = torch.tensor(
            [row["label_a"], row["label_b"], row["label_c"]], dtype=torch.long
        )
        out = {
            "speech": waveform,         # [T] @16kHz
            "speech_length": torch.tensor(waveform.size(0), dtype=torch.long),
            "labels": labels,           # [3]
            "key": row["key"],
            "idx": idx,                 # global index in dataset
        }
        if self.use_dict_features:
            key = row["key"]
            feat = self._dict_features.get(key, None)
            if feat is not None:
                out["dict_features"] = torch.tensor(feat, dtype=torch.float)
            else:
                out["dict_features"] = torch.zeros(126, dtype=torch.float)
        return out


def collate_audio(batch: list[dict]) -> dict:
    """Collate function: pad speech to max length in batch."""
    speeches = [b["speech"] for b in batch]
    speech_lengths = torch.stack([b["speech_length"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    keys = [b["key"] for b in batch]

    # Pad to max length
    speech_padded = pad_sequence(speeches, batch_first=True, padding_value=0.0)

    out = {
        "speech": speech_padded,          # [B, T_max]
        "speech_lengths": speech_lengths,  # [B]
        "labels": labels,                  # [B, 3]
        "keys": keys,
        "indices": torch.tensor([b["idx"] for b in batch]),
    }
    if "dict_features" in batch[0]:
        out["dict_features"] = torch.stack([b["dict_features"] for b in batch])
    return out


def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 2,
) -> tuple[dict[str, DataLoader], pd.DataFrame]:
    """Return {'train': loader, 'val': loader, 'test': loader} and the full DataFrame."""
    df = build_label_dataframe()

    loaders = {}
    for split in ["train", "val", "test"]:
        mask = df["split"] == split
        sub_df = df[mask].reset_index(drop=True)
        ds = AudioMultiAnnotatorDataset(sub_df, augment=(split == "train"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True,
            collate_fn=collate_audio,
        )
        print(f"  {split}: {len(sub_df)} samples", flush=True)

    return loaders, df


if __name__ == "__main__":
    loaders, df = get_dataloaders()
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"{split}: speech={batch['speech'].shape}, lengths={batch['speech_lengths'].shape}, labels={batch['labels'].shape}")
