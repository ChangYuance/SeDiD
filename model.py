#!/usr/bin/env python3
"""端到端多标注者学习模型，完全匹配 sensevoice_multitask_unfreeze_all_v2 配置。

Architecture:
  Raw audio → WavFrontend (mel 80 + LFR m=7,n=6 → 560-dim)
  → SenseVoiceEncoderSmall (full finetune) → [B, T', 512]
  → AttentionPool → [B, 512]
  → Head A: CTC ASR (digit recognition, auxiliary)
  → Head B: FC(512→64) → ReLU → Dropout → FC(64→4) → logits
      → p = softmax(logits)
  → Head C: MLP(512→64→48) → 3 × 4×4 logits → row-softmax → M_k
      → q_k = p @ M_k (instance-dependent confusion matrices)
"""
from __future__ import annotations
import torch.utils.checkpoint as checkpoint
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# dys_model_sensevoice.py is vendored alongside this file.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in os.environ.get("PYTHONPATH", ""):
    import sys; sys.path.insert(0, str(_SRC_DIR))

from funasr.frontends.wav_frontend import WavFrontend
from funasr.models.sense_voice.model import SenseVoiceEncoderSmall
from funasr.models.transformer.embedding import SinusoidalPositionEncoder


# ── Focal Loss (gamma=2.0, same as proven experiment) ──

# ── SpecAugment (frequency + time masking on LFR features) ──

def apply_specaugment(feats, feat_lens, freq_mask_param=14, time_mask_param=10):
    """Apply SpecAugment on [B, T, 560] mel+LFR features.

    Args:
        freq_mask_param: max # of freq channels to mask (out of 560)
        time_mask_param: max # of time frames to mask
    """
    b, t, f = feats.shape
    # Frequency masking (same mask for all samples in batch)
    f_start = torch.randint(0, max(1, f - freq_mask_param), (1,)).item()
    feats[:, :, f_start:f_start + freq_mask_param] = 0.0
    # Time masking (per-sample, only within valid length)
    for i in range(b):
        t_len = int(feat_lens[i].item())
        if t_len > time_mask_param:
            t_start = torch.randint(0, t_len - time_mask_param, (1,)).item()
            feats[i, t_start:t_start + time_mask_param, :] = 0.0
    return feats


def focal_loss(logits, labels, gamma=2.0, class_weights=None, sample_weights=None):
    ce = F.cross_entropy(logits, labels, reduction='none', weight=class_weights)
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma * ce)
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


# ── AttentionPooling (same as proven experiment) ──

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -float('inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


# ── Instance-dependent Confusion Matrix Generator (V2: full MLP) ──

class MGenerator(nn.Module):
    """Lightweight MLP: pooled(512) → hidden(64) → 3×4×4 logits.

    Each 4×4 block is row-softmaxed to produce P(annotator_label | true_class)
    conditioned on the input audio features.
    """

    def __init__(self, dim=512, hidden=64, n_annotators=3, n_classes=4, init_diag=0.8):
        super().__init__()
        self.n_annotators = n_annotators
        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_annotators * n_classes * n_classes),
        )

        # Initialize final layer bias to produce near-diagonal M at start
        off_diag_logit = torch.logit(torch.tensor((1 - init_diag) / (n_classes - 1)))
        diag_logit = torch.logit(torch.tensor(init_diag))
        bias = torch.full((n_annotators, n_classes, n_classes), off_diag_logit.item())
        for k in range(n_annotators):
            for i in range(n_classes):
                bias[k, i, i] = diag_logit.item()
        self.net[-1].bias.data = bias.view(-1)

    def forward(self, pooled):
        """pooled: [B, dim] → list of 3 M_k: [B, 4, 4]"""
        M_logits = self.net(pooled)  # [B, 48]
        M_logits = M_logits.view(-1, self.n_annotators, self.n_classes, self.n_classes)
        Ms = []
        for k in range(self.n_annotators):
            Ms.append(F.softmax(M_logits[:, k], dim=-1))  # [B, 4, 4]
        return Ms

    def forward_logits(self, pooled):
        """pooled: [B, dim] → [B, n_annotators, n_classes, n_classes] 未 softmax."""
        M_logits = self.net(pooled)
        return M_logits.view(-1, self.n_annotators, self.n_classes, self.n_classes)


# ── DictAdaptor: ASR dict 特征 → CM offset ──

class DictAdaptor(nn.Module):
    """根据 ASR 字典特征生成 per-annotator CM offset。

    如果 ASR 识别质量差（重度患者典型），offset 会调整混淆矩阵
    对标注不确定性的建模。零初始化 → 开始时无偏移。
    """
    def __init__(self, dict_dim=126, proj_dim=64, n_annotators=3, n_classes=4):
        super().__init__()
        self.n_annotators = n_annotators
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(dict_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, n_annotators * n_classes * n_classes),
        )
        # 零初始化 → 开始训练时 offset=0，M_k 完全由语音决定
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, dict_features):
        """dict_features: [B, 126] → [B, n_annotators, n_classes, n_classes]"""
        offset = self.net(dict_features)
        return offset.view(-1, self.n_annotators, self.n_classes, self.n_classes)


# ── LFC-x: Instance-dependent via additive design (Li et al. 2021) ──

class LFCxGenerator(nn.Module):
    """Instance-dependent confusion matrix via additive offset.

    M_k(x) = softmax(W_k^T h + B_k)
      - B_k: per-annotator confusion matrix template (near-diagonal init)
      - W_k: instance impact linear projection (init to 0 per paper)

    Two-stage training procedure (see train.py):
      1. Freeze W_k (instance_impact), train B_k + classifier
      2. Freeze B_k, train W_k + classifier
    """

    def __init__(self, dim=512, n_annotators=3, n_classes=4, init_diag=0.8):
        super().__init__()
        self.n_annotators = n_annotators
        self.n_classes = n_classes

        # B_k: per-annotator confusion matrix logits (near-diagonal init)
        self.confusion_logits = nn.Parameter(torch.zeros(n_annotators, n_classes, n_classes))
        off_diag = (1 - init_diag) / (n_classes - 1)
        for k in range(n_annotators):
            for i in range(n_classes):
                for j in range(n_classes):
                    val = init_diag if i == j else off_diag
                    self.confusion_logits.data[k, i, j] = math.log(val / (1 - val + 1e-10) + 1e-10)

        # W_k: instance impact linear layer (init to 0, per paper §3.3)
        self.instance_impact = nn.Linear(dim, n_annotators * n_classes * n_classes, bias=False)
        nn.init.zeros_(self.instance_impact.weight)

    def forward(self, pooled):
        """pooled: [B, dim] → list of 3 M_k: [B, 4, 4]"""
        impact = self.instance_impact(pooled)  # [B, 48]
        impact = impact.view(-1, self.n_annotators, self.n_classes, self.n_classes)
        Ms = []
        for k in range(self.n_annotators):
            M_logits = impact[:, k] + self.confusion_logits[k].unsqueeze(0)  # [B, 4, 4]
            Ms.append(F.softmax(M_logits, dim=-1))
        return Ms


# ── Fixed (instance-independent) Confusion Matrix ──

class ConfusionMatrix(nn.Module):
    """4×4 可学习混淆矩阵，行 softmax 约束。"""

    def __init__(self, n_classes=4, init_diag=0.8, noise=0.1):
        super().__init__()
        raw_diag = init_diag + noise * torch.randn(n_classes)
        diag = torch.clamp(raw_diag, 0.4, 0.95)
        mat = torch.zeros(n_classes, n_classes)
        for i in range(n_classes):
            remaining = (1 - diag[i].item()) / (n_classes - 1)
            row = torch.full((n_classes,), max(remaining, 0.01))
            row[i] = diag[i]
            mat[i] = row
        mat = mat / mat.sum(dim=-1, keepdim=True)
        self.logits = nn.Parameter(torch.log(mat / (1 - mat + 1e-10) + 1e-10))

    def forward(self):
        return F.softmax(self.logits, dim=-1)


# ── Main Model ──

# Pre-fine-tuned SenseVoice checkpoint (trained on the training split of the
# closed dataset). Not distributed; set SENSEVOICE_CKPT to your local copy.
FINETUNED_CKPT = os.environ.get(
    "SENSEVOICE_CKPT",
    str(Path(__file__).resolve().parent / "pretrained" / "sensevoice_finetuned.pt"),
)


class SenseVoiceMultiAnnotator(nn.Module):
    """端到端多标注者模型（WavFrontend + 全 finetune + AttentionPool + instance-dependent M + CTC）。"""

    # CTC token map for "一二三四五六七八九十"
    VOCAB_SIZE = 11  # blank(0) + 10 digits
    TARGET_SEQ = list(range(1, 11))

    def __init__(
        self,
        n_classes=4,
        n_annotators=3,
        unfreeze_from_ckpt=FINETUNED_CKPT,
        use_ctc=True,
        focal_gamma=2.0,
        variant="v2",
        dict_dim=0,
        num_unfrozen_layers: int = -1,  # -1 = all, N = last N Conformer blocks
    ):
        """SenseVoice multi-annotator model.

        Args:
            variant: one of "v2" (full MLP M + CTC + weighted focal),
                           "v2a" (fixed M + CTC + weighted focal),
                           "crowdlayer" (fixed M + CE only, no CTC),
                           "lfcx" (additive M + CE only, no CTC),
                           "tanno" (fixed M + trace reg, no CTC),
                           "coinnet" (fixed M + outlier vec + vol reg, no CTC),
                           "crowdattention" (cross-attention pseudo-label + CTC).
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.variant = variant
        self.use_specaug = False

        # Baseline variants (crowdlayer, lfcx, tanno, tannobias, coinnet, ddpm): no CTC
        is_baseline = variant in ("crowdlayer", "lfcx", "tanno", "tannobias", "coinnet", "ddpm", "loratanno",
                                   "sgtanno", "dictfusion_v2", "dictfusion_tanno", "dictadapt", "prototanno")
        self.use_ctc = use_ctc and not is_baseline

        # WavFrontend (mel 80 + LFR m=7,n=6)
        self.frontend = WavFrontend(
            fs=16000, window='hamming', n_mels=80,
            frame_length=25, frame_shift=10,
            lfr_m=7, lfr_n=6,
            dither=0.0, snip_edges=True,
        )

        # Encoder (same architecture as proven experiment)
        self.encoder = SenseVoiceEncoderSmall(
            input_size=560, output_size=512, attention_heads=4,
            linear_units=2048, num_blocks=50, tp_blocks=20,
            kernel_size=11, dropout_rate=0.1, positional_dropout_rate=0.1,
            attention_dropout_rate=0.1, input_layer="pe",
            pos_enc_class=SinusoidalPositionEncoder,
            normalize_before=True, selfattention_layer_type="sanm",
        )

        # Load finetuned checkpoint
        if unfreeze_from_ckpt and os.path.isfile(unfreeze_from_ckpt):
            ckpt = torch.load(unfreeze_from_ckpt, map_location='cpu', weights_only=True)
            enc_sd = {k.replace("encoder.", ""): v for k, v in ckpt.items()
                      if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc_sd, strict=False)
            print(f"[init] Loaded finetuned encoder ({len(enc_sd)} keys)", flush=True)

        # Partial freeze: keep only last N Conformer blocks trainable
        for p in self.encoder.parameters():
            p.requires_grad = False
        if num_unfrozen_layers < 0:
            # Full finetuning — unfreeze everything
            for p in self.encoder.parameters():
                p.requires_grad = True
        elif num_unfrozen_layers > 0:
            # Only unfreeze last N Conformer blocks + after_norm
            n_total = len(self.encoder.encoders)
            start = max(0, n_total - num_unfrozen_layers)
            for i in range(start, n_total):
                for p in self.encoder.encoders[i].parameters():
                    p.requires_grad = True
            for p in self.encoder.after_norm.parameters():
                p.requires_grad = True
        self.encoder.train()
        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"[init] Encoder: {n_enc:,} trainable params"
              f" ({num_unfrozen_layers if num_unfrozen_layers >= 0 else 'all'} blocks)",
              flush=True)

        # Attention pooling (load from checkpoint or init fresh)
        self.attention_pool = AttentionPooling(512)
        _attn_sd = {k.replace("attention_pool.", ""): v for k, v in ckpt.items()
                     if k.startswith("attention_pool.")} if 'ckpt' in dir() else {}
        if _attn_sd:
            self.attention_pool.load_state_dict(_attn_sd, strict=False)
            print(f"[init] Loaded attention_pool", flush=True)

        # Dict feature projection (for dict_fusion variant)
        self.dict_dim = dict_dim
        if dict_dim > 0:
            self.dict_proj = nn.Linear(dict_dim, 128)
        encoder_feat_dim = 512 + (128 if dict_dim > 0 else 0)

        # Severity head (load from checkpoint or init fresh)
        self.severity_fc1 = nn.Linear(encoder_feat_dim, 64)
        self.severity_dropout = nn.Dropout(0.3)
        self.severity_fc2 = nn.Linear(64, n_classes)
        if dict_dim == 0 and 'ckpt' in dir():
            for src, dst in [('severity_fc1', self.severity_fc1),
                             ('severity_fc2', self.severity_fc2)]:
                sd = {k.replace(src + ".", ""): v for k, v in ckpt.items()
                      if k.startswith(src + ".")}
                if sd:
                    dst.load_state_dict(sd, strict=False)
                    print(f"[init] Loaded {src}", flush=True)

        # v2_mid: override severity_fc1 to 1024->64, init first 512 cols from checkpoint
        if variant == "v2_mid":
            old_weight = self.severity_fc1.weight.data  # [64, 512]
            old_bias = self.severity_fc1.bias.data  # [64]
            self.severity_fc1 = nn.Linear(1024, 64)
            with torch.no_grad():
                self.severity_fc1.weight[:, :512].copy_(old_weight)  # final-feat part
                self.severity_fc1.weight[:, 512:].normal_(0, 0.01)   # mid-feat part
                self.severity_fc1.bias.copy_(old_bias)
            print(f"[init] severity_fc1 expanded to 1024->64", flush=True)

        # CTC head (auxiliary task, only for non-baseline variants)
        if self.use_ctc:
            self.ctc_head = nn.Linear(512, self.VOCAB_SIZE)
            self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            self.register_buffer('ctc_targets',
                                 torch.tensor([self.TARGET_SEQ], dtype=torch.long),
                                 persistent=False)

        # Confusion matrix generator per variant
        if variant == "v2":
            self.M_generator = MGenerator(
                dim=512, hidden=64,
                n_annotators=n_annotators, n_classes=n_classes,
            )
        elif variant == "v2_mid":
            self.M_generator = MGenerator(
                dim=1024, hidden=64,  # 512 mid + 512 final
                n_annotators=n_annotators, n_classes=n_classes,
            )
            # severity_fc1 override is done after checkpoint load (above)
            # Hook to capture middle-layer encoder features (layer 25 of 50)
            self.mid_features = None
            def _capture_mid(module, input, output):
                self.mid_features = output[0]  # xs_pad [B, T, 512]
            self.encoder.encoders[24].register_forward_hook(_capture_mid)
            self.mid_attention_pool = AttentionPooling(512)
        elif variant == "v2a":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
        elif variant == "crowdlayer":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
        elif variant == "tanno":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
        elif variant == "tannobias":
            self.doctor_biases = nn.Parameter(torch.zeros(n_annotators, n_classes))
        elif variant == "loratanno":
            self.r = 1  # low-rank adaptation rank
            # Shared base: near-diagonal init
            S = torch.zeros(n_classes, n_classes)
            S = S + torch.eye(n_classes) * 2.0  # diag=2 → softmax gives ~0.82 on diag
            self.shared_S = nn.Parameter(S)
            # Per-annotator low-rank factors (zero init → M_k starts ≈ softmax(S))
            self.lora_A = nn.Parameter(torch.randn(n_annotators, n_classes, self.r) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(n_annotators, n_classes, self.r) * 0.01)
        elif variant == "sgtanno":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
        elif variant == "prototanno":
            n_prototypes = 2
            self.prototypes = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_prototypes)
            ])
            self.annotator_logits = nn.Parameter(
                torch.zeros(n_annotators, n_prototypes)
            )
        elif variant == "dictfusion_v2":
            # Instance-dependent CMs on speech features + dict-augmented classifier
            self.M_generator = MGenerator(
                dim=512, hidden=64,
                n_annotators=n_annotators, n_classes=n_classes,
            )
        elif variant == "dictfusion_tanno":
            # Fixed CMs + dict-augmented classifier (Tanno-style)
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
        elif variant == "dictadapt":
            # Dict-Adaptive CMs: speech-based M_generator + dict_offset
            self.M_generator = MGenerator(
                dim=512, hidden=64,
                n_annotators=n_annotators, n_classes=n_classes,
            )
            self.dict_adaptor = DictAdaptor(
                dict_dim=126, proj_dim=64,
                n_annotators=n_annotators, n_classes=n_classes,
            )
            # Override: severify head takes 512-dim speech only (no dict)
            self.severity_fc1 = nn.Linear(512, 64)
            self.severity_dropout = nn.Dropout(0.3)
            self.severity_fc2 = nn.Linear(64, n_classes)
        elif variant == "ddpm":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
            self.consensus_fc = nn.Linear(64, n_classes)  # separate consensus head
            nn.init.xavier_uniform_(self.consensus_fc.weight, gain=0.5)
        elif variant == "coinnet":
            self.confusion_matrices = nn.ModuleList([
                ConfusionMatrix(n_classes) for _ in range(n_annotators)
            ])
            # Per-sample outlier vectors: [N_train, M, K]; placeholder until init_e_params is called
            self.e_params = nn.Parameter(torch.empty(0, n_annotators, n_classes))
        elif variant == "lfcx":
            self.M_generator = LFCxGenerator(
                dim=512, n_annotators=n_annotators, n_classes=n_classes,
            )
        elif variant == "crowdattention":
            pass  # No confusion matrices, no M_generator
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.focal_gamma = focal_gamma

    def init_e_params(self, n_train: int, init_scale: float = 0.01):
        """Initialize per-sample outlier vectors for COINNet.

        Called from train.py after dataloader is created, since n_train
        is only known then.
        """
        if self.variant != "coinnet":
            return
        self.e_params = nn.Parameter(
            torch.randn(n_train, self.n_annotators, self.n_classes) * init_scale
        )

    def _compute_frontend(self, speech, speech_lengths):
        """Raw audio → WavFrontend features (iterates batch items)."""
        batch_size = speech.size(0)
        feats_list, feat_lens_list = [], []
        for i in range(batch_size):
            wav_len = speech_lengths[i]
            wav = speech[i, :wav_len]
            feat, feat_len = self.frontend(
                wav.unsqueeze(0), torch.tensor([wav_len], device=speech.device)
            )
            feats_list.append(feat[0])
            feat_lens_list.append(feat_len[0])
        feat_lens = torch.stack(feat_lens_list)
        from torch.nn.utils.rnn import pad_sequence
        feats_pad = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
        feats_pad = feats_pad.to(device=speech.device)
        feat_lens = feat_lens.to(device=speech.device)
        return feats_pad, feat_lens

    def _encoder_checkpointed(self, feats, feat_lens):
        """Run encoder with gradient checkpointing on the heavy 50-layer Conformer block."""
        maxlen = feats.shape[1]
        from dys_model_sensevoice import sequence_mask
        masks = sequence_mask(feat_lens, maxlen=maxlen, device=feat_lens.device)[:, None, :]
        xs_pad = feats * (self.encoder.output_size() ** 0.5)
        xs_pad = self.encoder.embed(xs_pad)

        # encoders0 (lightweight)
        for layer in self.encoder.encoders0:
            outs = layer(xs_pad, masks)
            xs_pad, masks = outs[0], outs[1]

        # encoders (50-layer Conformer, heavy — checkpointed)
        def run_encoders(xs, m):
            for layer in self.encoder.encoders:
                outs = layer(xs, m)
                xs, m = outs[0], outs[1]
            return xs, m

        xs_pad, masks = checkpoint.checkpoint(run_encoders, xs_pad, masks, use_reentrant=False)

        xs_pad = self.encoder.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1).int()

        for layer in self.encoder.tp_encoders:
            outs = layer(xs_pad, masks)
            xs_pad, masks = outs[0], outs[1]

        xs_pad = self.encoder.tp_norm(xs_pad)
        return xs_pad, olens

    def forward(self, speech, speech_lengths, labels=None, indices=None, dict_features=None):
        """
        Args:
            speech: [B, T_max] raw waveform @ 16kHz
            speech_lengths: [B] length of each utterance in samples
            labels: [B, 3] annotator labels (optional, for loss computation)
            indices: [B] dataset indices (required for coinnet variant)
            dict_features: [B, dict_dim] ASR dictionary features (for dict_fusion variant)
        Returns:
            dict with keys: 'logits', 'p', 'q_list', 'M_list', 'pooled',
              and optionally 'ctc_loss', 'e_list' (coinnet)
        """
        # 1. Frontend
        feats, feat_lens = self._compute_frontend(speech, speech_lengths)

        # 1b. SpecAugment (only during training, controlled via self.training)
        if self.training and getattr(self, 'use_specaug', False):
            feats = apply_specaugment(feats, feat_lens)

        # 2. Encoder (with gradient checkpointing to save memory)
        encoder_out, olens = self._encoder_checkpointed(feats, feat_lens)

        # 3. Attention pooling
        from dys_model_sensevoice import sequence_mask
        mask = sequence_mask(olens, maxlen=encoder_out.size(1))
        pooled = self.attention_pool(encoder_out, mask)
        pooled_original = pooled  # save for M_generator (dictfusion_v2 uses speech-only)

        # 3b. Middle-layer feature fusion (v2_mid variant)
        if self.variant == "v2_mid":
            mid_out = self.mid_features  # [B, T_mid, 512] from hook on encoders[24]
            if mid_out is not None:
                mid_mask = sequence_mask(olens, maxlen=mid_out.size(1))
                mid_pooled = self.mid_attention_pool(mid_out, mid_mask)  # [B, 512]
                pooled = torch.cat([mid_pooled, pooled], dim=1)  # [B, 1024]

        # 3b. Dict feature fusion (for dict_fusion variants)
        if dict_features is not None and self.dict_dim > 0:
            dict_feat = self.dict_proj(dict_features)
            if self.variant != "dictadapt":
                pooled = torch.cat([pooled, dict_feat], dim=1)

        # 4. Severity head
        severity_out = self.severity_dropout(F.relu(self.severity_fc1(pooled)))
        logits = self.severity_fc2(severity_out)
        p = F.softmax(logits, dim=-1)

        # 5. Confusion matrices → annotator distributions
        if self.variant in ("v2a", "crowdlayer", "tanno"):
            M_list = [cm() for cm in self.confusion_matrices]  # list of 3 [4, 4]
            q_list = [p @ M for M in M_list]
        elif self.variant == "sgtanno":
            M_list = [cm() for cm in self.confusion_matrices]
            q_list = [p.detach() @ M for M in M_list]  # stop-gradient: p only trained by MV
        elif self.variant == "loratanno":
            S = self.shared_S  # [4, 4]
            M_list = []
            for k in range(self.n_annotators):
                Delta = self.lora_A[k] @ self.lora_B[k].transpose(-2, -1)  # [4, 4]
                M_k = F.softmax(S + Delta, dim=-1)  # [4, 4]
                M_list.append(M_k)
            q_list = [p @ M for M in M_list]
        elif self.variant == "prototanno":
            prots = [prot() for prot in self.prototypes]  # list of 2 [4, 4]
            alphas = F.softmax(self.annotator_logits, dim=-1)  # [3, 2]
            M_list = []
            for k in range(self.n_annotators):
                M_k = sum(alphas[k, s] * prots[s] for s in range(len(self.prototypes)))
                M_list.append(M_k)
            q_list = [p @ M for M in M_list]
        elif self.variant == "ddpm":
            # Annotator path: existing severity head → p_anno → @M_k → q_k
            p_anno = p  # severity_fc2 output → p
            M_list = [cm() for cm in self.confusion_matrices]
            q_list = [p_anno @ M for M in M_list]
            # Consensus path: separate head → p_soft
            logits_soft = self.consensus_fc(severity_out)
            p_soft = F.softmax(logits_soft, dim=-1)
        elif self.variant == "tannobias":
            M_list = None
            q_list = [F.softmax(torch.log(p + 1e-10) + self.doctor_biases[k], dim=-1)
                      for k in range(self.n_annotators)]
        elif self.variant == "coinnet":
            M_list = [cm() for cm in self.confusion_matrices]  # list of 3 [4, 4]
            q_list = [p @ M for M in M_list]
            # Per-sample outlier vectors with zero-sum constraint
            if indices is not None and self.e_params.shape[0] > 0:
                e_list = []
                e_params = self.e_params.to(indices.device)
                for k in range(self.n_annotators):
                    e_k = e_params[indices, k]  # [B, 4]
                    e_k = e_k - e_k.mean(dim=-1, keepdim=True)  # zero-sum
                    e_list.append(e_k)
            else:
                e_list = [torch.zeros_like(q) for q in q_list]
        elif self.variant == "crowdattention":
            # Cross-attention: pseudo-label = weighted sum of annotator one-hot labels
            # weight_k = p[label_of_annotator_k] — how much p agrees with annotator k
            if labels is not None:
                B = p.size(0)
                one_hot = F.one_hot(labels, self.n_classes).float()  # [B, 3, 4]
                weights = torch.stack([
                    p[torch.arange(B, device=p.device), labels[:, k]]
                    for k in range(self.n_annotators)
                ], dim=1)  # [B, 3]
                weights_norm = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
                pseudo_label = torch.bmm(weights_norm.unsqueeze(1), one_hot).squeeze(1)  # [B, 4]
            else:
                pseudo_label = None
            M_list = []
            q_list = []
        elif self.variant in ("v2", "v2_mid", "lfcx"):
            M_list = self.M_generator(pooled)  # list of 3 [B, 4, 4]
            q_list = []
            for k in range(self.n_annotators):
                # detach p: conf_CE gradient only trains C^k, not p
                q_k = torch.bmm(p.detach().unsqueeze(1), M_list[k]).squeeze(1)  # [B, 4]
                q_list.append(q_k)
        elif self.variant == "dictfusion_v2":
            M_list = self.M_generator(pooled_original)  # speech-only features for CMs
            q_list = []
            for k in range(self.n_annotators):
                q_k = torch.bmm(p.detach().unsqueeze(1), M_list[k]).squeeze(1)
                q_list.append(q_k)
        elif self.variant == "dictfusion_tanno":
            M_list = [cm() for cm in self.confusion_matrices]  # list of 3 [4, 4]
            q_list = [p @ M for M in M_list]
        elif self.variant == "dictadapt":
            # Base CMs from speech features (pre-softmax logits)
            M_base_logits = self.M_generator.forward_logits(pooled_original)  # [B, 3, 4, 4]
            # Dict offset calibration (zero-init → starts as zero offset)
            if dict_features is not None:
                dict_offset = self.dict_adaptor(dict_features)  # [B, 3, 4, 4]
            else:
                dict_offset = 0
            M_list = []
            for k in range(self.n_annotators):
                M_k = F.softmax(M_base_logits[:, k] + dict_offset[:, k], dim=-1)  # [B, 4, 4]
                M_list.append(M_k)
            q_list = []
            for k in range(self.n_annotators):
                q_k = torch.bmm(p.unsqueeze(1), M_list[k]).squeeze(1)
                q_list.append(q_k)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        out = {
            'logits': logits_soft if self.variant == "ddpm" else logits,
            'p': p_soft if self.variant == "ddpm" else p,
            'q_list': q_list,
            'M_list': M_list,  # per-sample M_k for inspection
            'pooled': pooled,
        }

        # DDPM: also expose annotator-path logits/p for loss computation
        if self.variant == "ddpm":
            out['logits_anno'] = logits
            out['p_anno'] = p_anno

        # COINNet: add per-sample outlier vectors to output
        if self.variant == "coinnet":
            out['e_list'] = e_list

        # CrowdAttention: add pseudo-label to output
        if self.variant == "crowdattention":
            out['pseudo_label'] = pseudo_label

        # 6. CTC head
        if self.use_ctc:
            ctc_logits = self.ctc_head(encoder_out)
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
            targets = self.ctc_targets.expand(speech.size(0), -1)
            target_lens = torch.full((speech.size(0),), 10,
                                     dtype=torch.long, device=speech.device)
            ctc_loss = self.ctc_loss_fn(
                ctc_log_probs.permute(1, 0, 2),
                targets, olens.int(), target_lens,
            )
            out['ctc_loss'] = ctc_loss

        return out

    def _compute_reg_loss(self) -> torch.Tensor:
        """(1 - diag(M))^2 regularization for confusion matrices."""
        if not hasattr(self, 'confusion_matrices'):
            return torch.tensor(0.0)
        total = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            diag = torch.diag(M)
            total = total + ((1 - diag) ** 2).mean()
        return total / len(self.confusion_matrices)

    def compute_loss(self, logits, p, q_list, labels,
                     lambda_conf=0.1, lambda_reg=0.0, class_weights=None):
        """
        Loss = weight × focal_loss(logits, consensus)
             + lambda_conf * Σ CE(q_k, y_k)
             + CTC_loss

        Consensus: majority vote (median for 3-different).
        Weight: 3 agree=1.0, 2 agree=0.5, all diff=0.0.
        """
        # ── Consensus with median tie-breaking ──
        counts = torch.zeros(labels.size(0), self.n_classes, device=labels.device)
        for k in range(self.n_annotators):
            counts.scatter_add_(1, labels[:, k:k+1],
                                torch.ones_like(labels[:, k:k+1], dtype=torch.float))

        same_01 = labels[:, 0] == labels[:, 1]
        same_02 = labels[:, 0] == labels[:, 2]
        same_12 = labels[:, 1] == labels[:, 2]
        any_two = same_01 | same_02 | same_12
        all_three = same_01 & same_02 & same_12

        # Agreement weight
        agree_weight = torch.full((labels.size(0),), 0.5, device=logits.device)
        agree_weight[all_three] = 1.0
        agree_weight[~any_two] = 0.0

        # Consensus (default: argmax = majority vote)
        consensus = counts.argmax(dim=-1)
        # 3-way tie → median instead of argmax(0)
        all_diff = ~any_two
        if all_diff.any():
            sorted_labels, _ = labels.sort(dim=-1)
            consensus[all_diff] = sorted_labels[all_diff, 1]

        # ── Loss components ──
        fl = focal_loss(logits, consensus, gamma=self.focal_gamma,
                        class_weights=class_weights, sample_weights=agree_weight)

        conf_ce = 0.0
        for k in range(self.n_annotators):
            conf_ce = conf_ce + F.nll_loss((q_list[k] + 1e-10).log(), labels[:, k])
        conf_ce = conf_ce / self.n_annotators

        reg_loss = self._compute_reg_loss()

        total = fl + lambda_conf * conf_ce + lambda_reg * reg_loss
        return total, fl, conf_ce, reg_loss

    def compute_baseline_loss(self, q_list, labels):
        """Standard CE(q_k, y_k) summed over annotators (for crowdlayer/lfcx)."""
        loss = 0.0
        for k in range(self.n_annotators):
            loss = loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        return loss / self.n_annotators

    def compute_v2tanno_loss(self, q_list, M_list, labels, lambda_trace=0.01):
        """Direction B: per-annotator CE + trace on instance-dependent M_k.

        ℒ = (1/3) Σ_k CE(q_k, y_k) + λ_trace · (1/3) Σ_k tr(M_k)
        """
        # CE loss
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Trace regularization on instance-dependent M_k
        trace_reg = 0.0
        for M in M_list:
            tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
            trace_reg = trace_reg + tr.mean()
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_trace * trace_reg
        return total, ce_loss, trace_reg

    def compute_tanno_loss(self, q_list, labels, lambda_trace=0.01):
        """Tanno et al. (CVPR 2019): CE(q_k, y_k) + trace regularization.

        ℒ = (1/3) Σ_k CE(q_k, y_k) + λ_trace · (1/3) Σ_k tr(M_k)

        Paper §2.2: "encouraging the estimated annotators to be maximally
        unreliable by minimizing the trace of the estimated CMs."
        Minimizing tr(M) forces the classifier p(x) to be accurate, because
        unreliable CMs cannot absorb the classification uncertainty.
        """
        # CE loss (same as baseline)
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Trace regularization: minimize tr(M_k) (push toward max unreliable)
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_trace * trace_reg
        return total, ce_loss, trace_reg

    def compute_loratanno_loss(self, q_list, M_list, labels, lambda_trace=0.01):
        """LoRATanno: CE(q_k, y_k) + λ_trace · tr(M_k).

        Same formula as Tanno, but M_k = softmax(S + A_k @ B_k^T) uses
        shared base + low-rank residual parameterization.
        """
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        trace_reg = 0.0
        for M in M_list:
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_trace * trace_reg
        return total, ce_loss, trace_reg

    def compute_softtanno_loss(self, q_list, p, labels, soft_labels,
                                lambda_soft=0.5, lambda_trace=0.01):
        """SoftTanno: CE(q_k, y_k) + λ_soft · KL(p || soft_label) + λ_trace · tr(M_k).

        Combines Tanno's per-annotator CE (guides confusion matrices to learn
        individual annotator biases) with SoftLabel's KL divergence against the
        consensus soft distribution (guides clean prediction p toward consensus).
        """
        # Per-annotator CE (same as Tanno)
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Soft label KL: -(soft_labels * log(p)).sum(dim=-1)
        p_log = (p + 1e-10).log()
        kl_loss = -(soft_labels * p_log).sum(dim=-1).mean()

        # Trace regularization (same as Tanno)
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_soft * kl_loss + lambda_trace * trace_reg
        return total, ce_loss, kl_loss, trace_reg

    def compute_crtanno_loss(self, q_list, p, labels, lambda_mv=0.1, lambda_trace=0.01):
        """CR-Tanno: CE(q_k, y_k) + λ_mv · CE(p, MV) + λ_trace · tr(M_k).

        MV = majority vote across 3 annotators. When all 3 disagree (rare),
        MV defaults to annotator 0's label.
        """
        # Per-annotator CE (same as Tanno)
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Majority vote — pick mode; on tie (all 3 diff) fallback to annotator 0
        mv_labels, mv_counts = torch.mode(labels, dim=1)
        # If mode count == 1 (all disagree), use annotator 0
        tie_mask = (mv_counts == 1)
        if tie_mask.any():
            mv_labels = mv_labels.clone()
            mv_labels[tie_mask] = labels[tie_mask, 0]

        # CE(p, MV)
        p_log = (p + 1e-10).log()
        mv_ce = F.cross_entropy(p_log, mv_labels)

        # Trace regularization
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_mv * mv_ce + lambda_trace * trace_reg
        return total, ce_loss, mv_ce, trace_reg

    def compute_softtanno_v2_loss(self, p, q_list, soft_labels,
                                   lambda_soft=1.0, lambda_anno=0.5,
                                   lambda_trace=0.01):
        """SoftTanno v2: KL(p||soft) + λ_anno·Σ_k KL(q_k||soft) + λ_trace·tr(M_k).

        All targets are the consensus soft distribution — no per-annotator hard
        labels. q_k = p @ M_k should match the soft label, meaning M_k learns
        how doctor k systematically deviates from the consensus.
        """
        # KL(p || soft_label)
        p_log = (p + 1e-10).log()
        kl_p = -(soft_labels * p_log).sum(dim=-1).mean()

        # Per-annotator KL(q_k || soft_label) — all same target, no contradiction
        kl_q = 0.0
        for k in range(self.n_annotators):
            qk_log = (q_list[k] + 1e-10).log()
            kl_q = kl_q - (soft_labels * qk_log).sum(dim=-1).mean()
        kl_q = kl_q / self.n_annotators

        # Trace regularization
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = lambda_soft * kl_p + lambda_anno * kl_q + lambda_trace * trace_reg
        return total, kl_p, kl_q, trace_reg

    def compute_tanno_blocked_loss(self, p, labels, soft_labels,
                                    lambda_soft=1.0, lambda_anno=0.5,
                                    lambda_trace=0.01):
        """TannoBlocked: KL(p||soft) + λ·CE(detach(p)@M_k, y_k) + λ_trace·tr(M_k).

        The key innovation: p is DETACHED before CE(q_k, y_k), so:
        - KL(p || soft_label) → trains encoder + classifier (clean consensus)
        - CE(q_k, y_k) with p.detach()  → trains ONLY confusion matrices M_k
        - trace → regularizes M_k toward identity

        No contradictory gradients on backbone — best of both worlds.
        """
        # KL(p || soft_label) — trains backbone
        p_log = (p + 1e-10).log()
        kl_loss = -(soft_labels * p_log).sum(dim=-1).mean()

        # Per-annotator CE with DETACHED p — trains only M_k
        ce_loss = 0.0
        for k in range(self.n_annotators):
            M_k = self.confusion_matrices[k]()  # [4, 4], has grad
            q_k = p.detach() @ M_k               # no grad to p
            ce_loss = ce_loss + F.cross_entropy((q_k + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Trace regularization
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = lambda_soft * kl_loss + lambda_anno * ce_loss + lambda_trace * trace_reg
        return total, kl_loss, ce_loss, trace_reg

    def compute_tanno_bias_loss(self, logits, labels, soft_labels,
                                 doctor_biases, lambda_soft=1.0, lambda_anno=0.5):
        """TannoBias: KL(p||soft) + λ·CE(detach(p) + b_k, y_k).

        Instead of 4×4 confusion matrices, each doctor has a 4-D bias vector b_k.
        q_k = softmax(log(p) + b_k).

        - Bias b_k captures "doctor k's tendency" with only 4 params each
        - No trace needed (bias vectors don't have identity interpretation)
        """
        p = F.softmax(logits, dim=-1)

        # KL(p || soft_label) — trains backbone
        p_log = (p + 1e-10).log()
        kl_loss = -(soft_labels * p_log).sum(dim=-1).mean()

        # Per-doctor bias-based CE with detached p — trains only bias vectors
        log_p_detached = torch.log(p.detach() + 1e-10)  # [B, 4]
        ce_loss = 0.0
        for k in range(self.n_annotators):
            q_k = F.softmax(log_p_detached + doctor_biases[k], dim=-1)  # [B, 4]
            ce_loss = ce_loss + F.cross_entropy((q_k + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        total = lambda_soft * kl_loss + lambda_anno * ce_loss
        return total, kl_loss, ce_loss

    def compute_ddpm_loss(self, q_list, p_soft, p_anno, labels, soft_labels,
                          lambda_soft=1.0, lambda_bridge=0.5, lambda_trace=0.01):
        """DDPM: CE(q_k, y_k) + λ_soft·KL(p_soft||soft) + λ_bridge·MSE(p_soft, p_anno) + λ_trace·tr(M_k).

        Decoupled Dual-Path Model:
        - q_k = p_anno @ M_k: per-annotator predictions with confusion matrices
        - p_soft: consensus head prediction, trained toward soft labels
        - MSE bridge: keeps the two heads consistent
        """
        # Per-annotator CE — trains backbone + CMs via p_anno path
        ce_loss = 0.0
        for k in range(self.n_annotators):
            ce_loss = ce_loss + F.cross_entropy((q_list[k] + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # Soft label KL — trains consensus head + backbone
        p_log = (p_soft + 1e-10).log()
        kl_loss = -(soft_labels * p_log).sum(dim=-1).mean()

        # Bridge: MSE between consensus and annotator predictions
        bridge_loss = F.mse_loss(p_soft, p_anno)

        # Trace regularization
        trace_reg = 0.0
        for cm in self.confusion_matrices:
            M = cm()
            tr = torch.diag(M).sum()
            trace_reg = trace_reg + tr
        trace_reg = trace_reg / self.n_annotators

        total = ce_loss + lambda_soft * kl_loss + lambda_bridge * bridge_loss + lambda_trace * trace_reg
        return total, ce_loss, kl_loss, bridge_loss, trace_reg

    def compute_coinnet_loss(self, q_list, e_list, labels,
                              mu1=0.01, mu2=0.01, zeta=1e-10, p=None):
        """COINNet loss (NeurIPS 2024): Coupled CE + outlier sparsity + volume.

        ℒ = ℒ_ce + µ₁·ℒ_outlier + µ₂·ℒ_vol

        ℒ_ce: Coupled CE — (1/3) Σ_k CE(g_k, y_k), g_k = q_k + e_k
        ℒ_outlier: (Σ_m ||e_n^{(m)}||₂² + ζ)^{p/2}, p=0.4 — sum squared norms
                    across annotators first, then power (Eq 15)
        ℒ_vol: -log det(F F^T), F = [f(x_1), ..., f(x_N)] where f(x) is the
               classifier output (K-dim). Encourages max volume of conv{F}.
        """
        # ── Coupled Cross-Entropy ──
        ce_loss = 0.0
        for k in range(self.n_annotators):
            g_k = q_list[k] + e_list[k]  # [B, 4]
            g_k = g_k.clamp(min=1e-10)
            g_k = g_k / g_k.sum(dim=-1, keepdim=True)
            ce_loss = ce_loss + F.cross_entropy((g_k + 1e-10).log(), labels[:, k])
        ce_loss = ce_loss / self.n_annotators

        # ── Outlier sparsity: (Σ_m ||e||₂² + ζ)^{p/2} (Eq 15) ──
        p_val = 0.4
        e_stack = torch.stack(e_list, dim=0)  # [n_annotators, B, 4]
        l2_sq = (e_stack ** 2).sum(dim=-1)  # [n_annotators, B]
        total_l2_sq = l2_sq.sum(dim=0)  # [B] — Σ_m ||e_n^{(m)}||₂²
        outlier_loss = ((total_l2_sq + zeta) ** (p_val / 2)).mean()

        # ── Volume regularization: -log det(F·F^T) (Eq 15) ──
        # F = [f(x_1), ..., f(x_N)] — classifier output (K-dim, 4 in our case)
        vol_loss = torch.tensor(0.0, device=p.device) if p is not None else torch.tensor(0.0)
        if p is not None and p.size(0) > 1:
            K = self.n_classes
            # F is K×B, F·F^T is K×K
            FFT = p.T @ p  # [K, K]
            I = torch.eye(K, device=p.device)
            sign, logdet = torch.linalg.slogdet(FFT + zeta * I)
            vol_loss = -logdet

        total = ce_loss + mu1 * outlier_loss + mu2 * vol_loss
        return total, ce_loss, outlier_loss, vol_loss

    def compute_crowdattention_loss(self, p, pseudo_label):
        """CrowdAttention loss: CE(p, pseudo_label).

        pseudo_label is a soft target distribution from cross-attention weighting.
        Loss = -Σ_j pseudo_label[j] · log(p[j])
        """
        return -(pseudo_label * torch.log(p + 1e-10)).sum(dim=-1).mean()

    def predict(self, speech, speech_lengths):
        """Inference: return hard labels [B,]."""
        out = self.forward(speech, speech_lengths)
        return out['logits'].argmax(dim=-1)

    def get_confusion_matrices(self, pooled=None):
        """Return average M_k (instance-dep) or fixed M_k."""
        if self.variant in ("v2a", "crowdlayer", "tanno", "coinnet", "dictfusion_tanno"):
            return [cm() for cm in self.confusion_matrices]
        if self.variant in ("v2", "lfcx", "dictfusion_v2", "dictadapt"):
            if pooled is None:
                return None
            M_list = self.M_generator(pooled)
            return [M.mean(dim=0) for M in M_list]
        if self.variant == "crowdattention":
            return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
