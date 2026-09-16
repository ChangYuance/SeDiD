#!/usr/bin/env python3
"""Baseline comparison models, each trained under the same 5-fold protocol.

  WhisperProbeMidModel  frozen Whisper-small mid-layer probe  (Yue et al., ICASSP 2026)
  MFCCStatsModel        DNN on MFCC statistics
  MFCCFusionModel       MFCC + LaBSE text-embedding fusion
  CoarseToFineWhisper   Whisper-based coarse-to-fine detection
  WhisperFTModel        fine-tuned Whisper-small encoder + MLP
"""
from __future__ import annotations
import os, torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Offline HuggingFace: use cached safetensors only
os.environ['TRANSFORMERS_OFFLINE'] = '1'
_HF_KWARGS = {"local_files_only": True, "revision": "af46f65f540dc3ca7aa59f46c6c3d5dbb4374fa8"}


# ═══════════════════════════════════════════════════════════════
# 1. Whisper Probe (frozen Whisper + linear head)
# ═══════════════════════════════════════════════════════════════
# Paper: Probing Whisper for Dysarthric Speech in Detection
#        and Assessment (Yue et al., ICASSP 2026)
# Key: frozen Whisper-small encoder + mean pooling + linear FC


class WhisperProbeMidModel(nn.Module):
    """Frozen Whisper-small encoder + configurable-layer probing + MLP classifier.

    Implements the probing scheme from Yue et al. ICASSP 2026:
    Probes hidden states from a specific encoder layer with a deep MLP.
    Default (mid_layer=-1) uses the last hidden state = same as WhisperProbe
    but with a deeper MLP instead of single linear layer.
    For Whisper-small (12 layers, hidden_states=[embed,0,1,...,11]),
    mid_layer=12 (0-indexed) = last layer = 1-indexed 13th hidden state.

    Fix: uses output_hidden_states=True (not forward hooks) for reliability.
    """
    def __init__(self, n_classes=4, mid_layer=-1, linear_probe=False):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small", local_files_only=True)
        self.encoder = WhisperModel.from_pretrained(
            "openai/whisper-small", local_files_only=True).encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768
        self.mid_layer = mid_layer  # -1 = last, 12 = last (0-indexed)

        if linear_probe:
            self.classifier = nn.Linear(hidden_size, n_classes)
        else:
            # Deeper MLP with LayerNorm for stability
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )
        self.feat_len = 3000

    def forward(self, speech, speech_lengths):
        B = speech.size(0)
        mels = []
        for i in range(B):
            wav = speech[i, :speech_lengths[i]].cpu().numpy()
            feats = self.feature_extractor(
                wav, sampling_rate=16000, return_tensors="pt",
            ).input_features  # [1, 80, 3000]
            mels.append(feats.to(speech.device))
        mel = torch.cat(mels, dim=0)  # [B, 80, 3000]

        out = self.encoder(mel, output_hidden_states=True)
        # hidden_states: [embed, layer0, ..., layer11] = 13 entries
        hidden = out.hidden_states[self.mid_layer]  # [B, 1500, 768]

        # Mean pooling
        mask = _make_mask(speech_lengths, hidden.size(1))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled}


# Shared mask helper
def _make_mask(speech_lengths, feat_len=None):
    B = speech_lengths.size(0)
    if feat_len is None:
        feat_len = 1500
    device = speech_lengths.device
    mask = torch.arange(feat_len, device=device).unsqueeze(0).expand(B, -1)
    feat_lens = ((speech_lengths - 1) // 320 + 1).clamp(max=feat_len)
    return (mask < feat_lens.unsqueeze(1)).float()


# ═══════════════════════════════════════════════════════════════
# 2. MFCC Stats baseline (Vishwanath et al., Interspeech 2025)
# ═══════════════════════════════════════════════════════════════
# Paper: Comparison of Acoustic and Textual Features for Dysarthria
#        Severity Classification in ALS (Vishwanath et al., Interspeech 2025)
# Key: 13 MFCC + delta + delta-delta → per-utt stats → linear classifier
#
# Acoustic-only branch: for each utterance, extract 39-D MFCC coefficients,
# compute {mean, std, skewness, kurtosis, min, max} per coefficient,
# concatenate into 234-D vector → linear classifier.

class MFCCStatsModel(nn.Module):
    """MFCC statistics + linear classifier (Vishwanath et al., Interspeech 2025).

    Per utterance:
      12 MFCC + 12 delta + 12 delta-delta = 36 coeffs
      3 stats (mean, SD, median) per coeff = 108-D feature vector
      → Linear(n_classes) classifier

    Note: Paper uses a 5-layer DNN, but it overfits on our smaller dataset
    (615 samples vs paper's ~1800). We use a simple linear classifier instead.
    """
    def __init__(self, n_classes=4, n_mfcc=12, sample_rate=16000,
                 norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

        # Frame-level feature dim: 12 mfcc + 12 delta + 12 delta-delta = 36
        feat_dim = n_mfcc * 3
        # 3 stats per coefficient (mean, SD, median) = 108-D
        self.stats_dim = feat_dim * 3  # 108

        # MFCC transform: 20ms window, 10ms hop (matches paper)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 64,
                       "win_length": 320, "power": 2.0},
        )

        # Feature normalization stats (computed from training set, per paper)
        self.register_buffer("norm_mean", torch.zeros(self.stats_dim))
        self.register_buffer("norm_std", torch.ones(self.stats_dim))
        if norm_stats is not None:
            self.norm_mean.copy_(norm_stats[0])
            self.norm_std.copy_(norm_stats[1].clamp(min=1e-8))

        # 2-layer MLP classifier — linear head underperforms (seed-42 F1=0.446);
        # paper's full 5-layer DNN overfits on MSDM (615 samples).
        # 3-layer MLP tested (fold0=0.340) worse than 2-layer (0.439).
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def _extract_stats(self, speech, speech_lengths):
        """Extract MFCC + delta + delta-delta → per-utterance statistics.

        Returns: [B, 108] feature vectors.
        """
        B = speech.size(0)
        all_stats = []
        for i in range(B):
            wav = speech[i, :speech_lengths[i]]
            if wav.size(0) < 320:
                wav = F.pad(wav, (0, 320 - wav.size(0)))

            # Compute MFCCs: [12, T]
            mfcc = self.mfcc_transform(wav)

            # Delta + delta-delta
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)

            # Concatenate: [36, T]
            feats = torch.cat([mfcc, delta, delta2], dim=0)

            # 3 stats per coefficient (mean, SD, median) — matches paper
            mu = feats.mean(dim=-1)
            std = feats.std(dim=-1)
            med, _ = feats.median(dim=-1)

            stats = torch.stack([mu, std, med], dim=-1).view(-1)
            all_stats.append(stats)

        return torch.stack(all_stats)

    def forward(self, speech, speech_lengths):
        feats = self._extract_stats(speech, speech_lengths)  # [B, 108]
        # Feature normalization (paper: normalize using train mean/std)
        feats = (feats - self.norm_mean) / self.norm_std
        logits = self.classifier(feats)
        return {"logits": logits, "pooled": feats}


# ═══════════════════════════════════════════════════════════════
# 3. MFCC + Text Fusion (Vishwanath et al., Interspeech 2025)
# ═══════════════════════════════════════════════════════════════
# Paper: Comparison of Acoustic and Textual Features for Dysarthria
#        Severity Classification in ALS (Vishwanath et al., Interspeech 2025)
# Key: MFCC stats (234-D) + ASR transcript embeddings → concat → DNN classifier
#
# Pipeline:
#   1. SenseVoice ASR transcripts → LaBSE sentence embeddings (768-D)
#   2. MFCC stats (same as MFCCStatsModel: 234-D)
#   3. Concatenate (1002-D) → BatchNorm → ReLU → linear classifier

class MFCCFusionModel(nn.Module):
    """MFCC statistics + text embeddings + linear classifier.

    Uses pre-computed LaBSE text embeddings (cached to disk).
    MFCC features computed on-the-fly from audio.

    Note: Paper uses a 5-layer DNN, but it overfits on our smaller dataset
    (615 samples vs paper's ~1800). We use a simple linear classifier instead.
    """
    def __init__(self, n_classes=4, n_mfcc=12, sample_rate=16000,
                 emb_path=None,
                 norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

        # MFCC transform (20ms window, 10ms hop — matches paper)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 64,
                       "win_length": 320, "power": 2.0},
        )

        # Text embedding dimension (LaBSE = 768)
        self.text_dim = 768
        stats_dim = n_mfcc * 3 * 3  # 12*3*3 = 108
        fusion_dim = stats_dim + self.text_dim  # 876

        # Feature normalization stats (paper: normalize using train mean/std)
        self.register_buffer("norm_mean", torch.zeros(fusion_dim))
        self.register_buffer("norm_std", torch.ones(fusion_dim))
        if norm_stats is not None:
            self.norm_mean.copy_(norm_stats[0])
            self.norm_std.copy_(norm_stats[1].clamp(min=1e-8))

        # Simple linear classifier
        self.classifier = nn.Linear(fusion_dim, n_classes)

        # Load cached text embeddings if available
        self.text_embeddings: dict[str, np.ndarray] | None = None
        self.register_buffer("_dummy", torch.zeros(0))  # marker buffer for hasattr check
        if emb_path is not None:
            self._load_embeddings(emb_path)

    def _load_embeddings(self, emb_path: str):
        _loaded = np.load(emb_path)
        self.text_embeddings = {k: _loaded[k] for k in _loaded.files}
        # Pre-register a buffer to track device; embeddings moved on-demand
        self.register_buffer("_dummy", torch.zeros(0))

    def _extract_mfcc_stats(self, speech, speech_lengths):
        """Same MFCC stats as MFCCStatsModel: [B, 108]."""
        B = speech.size(0)
        all_stats = []
        for i in range(B):
            wav = speech[i, :speech_lengths[i]]
            if wav.size(0) < 320:
                wav = F.pad(wav, (0, 320 - wav.size(0)))
            mfcc = self.mfcc_transform(wav)
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)
            feats = torch.cat([mfcc, delta, delta2], dim=0)
            mu = feats.mean(dim=-1)
            std = feats.std(dim=-1)
            med, _ = feats.median(dim=-1)
            stats = torch.stack([mu, std, med], dim=-1).view(-1)
            all_stats.append(stats)
        return torch.stack(all_stats)

    def _get_text_embeddings(self, keys: list[str]) -> torch.Tensor:
        """Look up text embeddings by key. Returns [B, 768]."""
        B = len(keys)
        device = self._dummy.device
        emb = torch.zeros(B, self.text_dim, device=device)
        for i, k in enumerate(keys):
            if self.text_embeddings is not None and k in self.text_embeddings:
                emb[i] = torch.tensor(self.text_embeddings[k], device=device)
        return emb

    def forward(self, speech, speech_lengths, keys=None):
        mfcc_stats = self._extract_mfcc_stats(speech, speech_lengths)  # [B, 108]
        if keys is not None:
            text_emb = self._get_text_embeddings(keys)  # [B, 768]
            fusion = torch.cat([mfcc_stats, text_emb], dim=-1)  # [B, 876]
        else:
            B = speech.size(0)
            device = mfcc_stats.device
            text_emb = torch.zeros(B, self.text_dim, device=device)
            fusion = torch.cat([mfcc_stats, text_emb], dim=-1)
        # Feature normalization (paper)
        fusion = (fusion - self.norm_mean) / self.norm_std
        logits = self.classifier(fusion)
        return {"logits": logits, "pooled": fusion}


# ═══════════════════════════════════════════════════════════════
# 4. Coarse-to-Fine Whisper (Paper #2, ICASSP 2026 SAND Challenge)
# ═══════════════════════════════════════════════════════════════
# Paper: A Hierarchical Coarse-to-Fine Whisper Adaptation Framework
#        for ALS Dysarthria Severity Estimation (ICASSP 2026 SAND)
# Key:
#   Stage 1: Pre-train on external neurodegenerative data (skip — not available)
#   Stage 2: Coarse D2 ≥1 gate {0} vs {1,2,3} with enhanced pooling
#   Stage 3: Fine {1,2,3} → Mild/Moderate/Severe on the positive branch
#
# Hierarchical output: 4-class logits = log-softmax(gate) + log-softmax(fine),
# a soft product over the tree, so argmax = 0 iff the gate predicts normal,
# else 1..3 = 1 + argmax(fine).
# Enhanced pooling: mean + std + attention pooling → concat → FC(256).
# Paper uses Whisper-Turbo (hidden=1280). We use Whisper-small (hidden=768) since cached.

class CoarseToFineWhisper(nn.Module):
    """Two-level coarse-to-fine Whisper → 4-class severity (ICASSP 2026 SAND).

    Stage-2 gate head {0} vs {1,2,3} (D2, ≥1 threshold) plus a Stage-3 fine
    head Mild/Moderate/Severe on the positive branch; outputs 4-class logits
    as a soft product over the tree.
    """
    def __init__(self, n_classes=4):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small", local_files_only=True)
        self.encoder = WhisperModel.from_pretrained(
            "openai/whisper-small", local_files_only=True).encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768

        # Stage 2 (gate): enhanced pooling (mean + std + attention) → FC(256) → 2
        #   coarse {0}=Normal vs {1,2,3}=dysarthric (D2, ≥1 threshold)
        self.stage2_attn_vector = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.stage2_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        # Stage 3 (fine): Mild / Moderate / Severe on the ≥1 branch.
        # Own attention vector, like the paper's separate Stage-3 model.
        self.fine_attn_vector = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

        self.feat_len = 3000

    def _enhanced_pooling(self, hidden, mask, attn_vector):
        """Mean + std + attention pooling → [B, D*3]."""
        # Mean pooling
        mean_pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        # Std pooling
        var = ((hidden - mean_pooled.unsqueeze(1)) * mask.unsqueeze(-1)).pow(2).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        std_pooled = var.sqrt()
        # Attention pooling with learned query vector
        attn_scores = torch.matmul(hidden, attn_vector.transpose(-2, -1)).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_pooled = (hidden * attn_weights.unsqueeze(-1)).sum(dim=1)
        return torch.cat([mean_pooled, std_pooled, attn_pooled], dim=-1)

    def _compute_mel(self, speech, speech_lengths):
        """Raw audio → Whisper mel spectrogram [B, 80, 3000]."""
        B = speech.size(0)
        mels = []
        for i in range(B):
            wav = speech[i, :speech_lengths[i]].cpu().numpy()
            feats = self.feature_extractor(
                wav, sampling_rate=16000, return_tensors="pt",
            ).input_features
            mels.append(feats.to(speech.device))
        return torch.cat(mels, dim=0)

    def _encode(self, speech, speech_lengths):
        """Frozen-encoder → (hidden [B,T,768], frame mask)."""
        mel = self._compute_mel(speech, speech_lengths)
        hidden = self.encoder(mel).last_hidden_state
        B, T = hidden.shape[:2]
        feat_lens = ((speech_lengths - 1) // 320 + 1).clamp(max=T)
        mask = torch.arange(T, device=hidden.device).unsqueeze(0).expand(B, -1)
        mask = (mask < feat_lens.unsqueeze(1)).float()
        return hidden, mask

    def _encode_pool(self, speech, speech_lengths, attn_vector):
        """Frozen-encoder features → enhanced pooled repr [B, 768*3]."""
        hidden, mask = self._encode(speech, speech_lengths)
        return self._enhanced_pooling(hidden, mask, attn_vector)

    def forward_stage2(self, speech, speech_lengths):
        """Stage-2 gate logits: {0}=Normal vs {1,2,3}=dysarthric (D2, ≥1)."""
        return self.stage2_head(
            self._encode_pool(speech, speech_lengths, self.stage2_attn_vector))

    def forward(self, speech, speech_lengths):
        """Hierarchical coarse-to-fine → 4-class logits.

        Stage-2 gate (coarse {0} vs {1,2,3}) and Stage-3 fine head
        (Mild/Moderate/Severe) each pool the frozen encoder with their own
        attention vector. logit[c] = log-softmax(gate)[coarse(c)]
        + log-softmax(fine)[fine(c)] — a soft product over the tree, so argmax
        returns class 0 iff the gate says normal, else 1..3 = 1+argmax(fine):
        exactly the D2 ≥1 binarization.
        """
        hidden, mask = self._encode(speech, speech_lengths)
        pooled_g = self._enhanced_pooling(hidden, mask, self.stage2_attn_vector)
        gate = self.stage2_head(pooled_g)
        fine = self.fine_head(self._enhanced_pooling(hidden, mask, self.fine_attn_vector))
        gl = F.log_softmax(gate, dim=-1)
        fl = F.log_softmax(fine, dim=-1)
        logits_4 = torch.cat([gl[:, 0:1], gl[:, 1:2] + fl], dim=-1)  # [B, 4]
        return {"logits": logits_4, "gate": gate, "fine": fine, "pooled": pooled_g}


# ═══════════════════════════════════════════════════════════════
# 5. Whisper-FT: Fine-tuned Whisper encoder + deep MLP classifier
# ═══════════════════════════════════════════════════════════════
# Paper: Clinical assessment and interpretation of dysarthria in ALS
#        using attention based deep learning AI models
#        Merler et al., npj Digital Medicine, 2025
# Key: fine-tune Whisper-small encoder end-to-end + 4-layer MLP
# Paper uses Whisper-base (74M, 512-dim), we use Whisper-small
# (244M, 768-dim) since it's cached locally.

class WhisperFTModel(nn.Module):
    """Whisper-small encoder + deep MLP classifier, fine-tuned end-to-end.

    Based on Merler et al. npj Digital Medicine 2025:
    - Whisper encoder fine-tuned end-to-end (not frozen)
    - Mean pooling over time
    - Deep MLP head: D→2048→1024→512→128→n_classes with Dropout
    - Paper original uses MSE regression; we adapt to 4-class CE.

    To prevent overfitting on small datasets (MSDM: 337 train samples),
    supports freezing bottom k encoder layers via n_freeze_layers.
    """
    def __init__(self, n_classes=4, n_freeze_layers=10):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small", local_files_only=True)
        self.encoder = WhisperModel.from_pretrained(
            "openai/whisper-small", local_files_only=True).encoder
        # Fine-tune only top (12 - n_freeze_layers) encoder layers
        # Whisper-small encoder has 12 layers
        if n_freeze_layers > 0:
            for i in range(min(n_freeze_layers, len(self.encoder.layers))):
                for p in self.encoder.layers[i].parameters():
                    p.requires_grad = False
            # Also freeze conv layers if freezing all encoder layers
            if n_freeze_layers >= len(self.encoder.layers):
                for p in self.encoder.conv1.parameters():
                    p.requires_grad = False
                for p in self.encoder.conv2.parameters():
                    p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768

        # Deep MLP head with Dropout for regularization
        # Paper: 4 linear layers with ReLU activations
        # Adapted from paper's 512→2048→1024→512→128→1 for regression
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        self.feat_len = 3000

    def forward(self, speech, speech_lengths):
        B = speech.size(0)
        # Compute mel spectrograms with Whisper's feature extractor
        mels = []
        for i in range(B):
            wav = speech[i, :speech_lengths[i]].cpu().numpy()
            feats = self.feature_extractor(
                wav, sampling_rate=16000, return_tensors="pt",
            ).input_features  # [1, 80, 3000]
            mels.append(feats.to(speech.device))
        mel = torch.cat(mels, dim=0)  # [B, 80, 3000]

        out = self.encoder(mel)
        hidden = out.last_hidden_state  # [B, 1500, 768]

        # Mean pooling over time
        mask = _make_mask(speech_lengths, hidden.size(1))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled}


