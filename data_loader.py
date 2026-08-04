#!/usr/bin/env python3
"""加载原始音频 + 三位医生标注，提供 DataLoader。

数据来源：
  1. labels_三位医生标记.xlsx — 543 条，含万/师/第三位医生标记
  2. doctor_review_summary.xlsx — 78 条 v2 患者，含统一标签

每条样本: speech【B, T_max】@16kHz, speech_lengths【B】, labels【B, 3】
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
# publicly distributed. Point these environment variables to your local copy
# of the dataset (see README.md for the expected layout).
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA = Path(os.environ.get("CSD615_TRAINING_DATA", str(BASE_DIR / "training_data_v2")))
EXCEL_LABELS = Path(os.environ.get("CSD615_LABELS", str(BASE_DIR / "data" / "labels_三位医生标记.xlsx")))
EXCEL_REVIEW = Path(os.environ.get("CSD615_REVIEW", str(BASE_DIR / "results" / "doctor_review_summary.xlsx")))

TARGET_SR = 16000
SOURCE_SR = 48000


def _load_v2_metadata() -> dict[str, dict]:
    """从 data_v2/{split}/data.list 读取所有样本的 key→{wav, split}。"""
    meta = {}
    for split in ["train", "val", "test"]:
        path = BASE_DIR / "data_v2" / split / "data.list"
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    meta[row["key"]] = {"wav": row["wav"], "split": split}
    return meta


def _load_doctor_labels() -> pd.DataFrame:
    """读取 labels_三位医生标记.xlsx。"""
    df = pd.read_excel(EXCEL_LABELS)
    df["key"] = df["ID"].astype(str)
    df["label_a"] = df["万_视频标签"].astype(int)
    df["label_b"] = df["师_视频标签"].astype(int)
    df["label_c"] = df["视频标记新"].fillna(df["万_视频标签"]).astype(int)
    return df[["key", "label_a", "label_b", "label_c", "split"]]


def _load_v2_unified_labels() -> dict[str, int]:
    wb = openpyxl.load_workbook(EXCEL_REVIEW)
    ws = wb.active
    labels = {}
    for row in range(2, ws.max_row + 1):
        pid = str(ws.cell(row=row, column=1).value)
        label = ws.cell(row=row, column=12).value
        if label is not None:
            labels[pid] = int(label)
    return labels


def build_label_dataframe() -> pd.DataFrame:
    """合并标签来源，返回 DataFrame: key, split, label_a, label_b, label_c, wav_path。"""
    meta = _load_v2_metadata()
    df_doctor = _load_doctor_labels()
    v2_unified = _load_v2_unified_labels()

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
        elif key in v2_unified:
            lbl = v2_unified[key]
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
    """加载原始音频 + 三位医生标签。首次加载时缓存所有音频到内存。"""

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
        """预加载所有音频到内存（mono, 16kHz）。"""
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
    """返回 {'train': loader, 'val': loader, 'test': loader} 和完整 DataFrame。"""
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
