"""SenseVoiceMultiTask: Pretrained SenseVoice encoder + CTC ASR + severity head.

Architecture:
  Raw audio (16kHz) → WavFrontend (mel 80-dim + LFR m=7,n=6 → 560-dim, ~/6 frame rate)
  → SenseVoiceEncoderSmall (pretrained, partially unfrozen) → [B, T', 512]
  → Head A: Linear(512→11) + CTC loss (recognize the 10 digits)
  → Head B: AttentionPool → FC(512→64) → ReLU → Dropout → FC(64→4) + Focal loss

Motivation: CTC forces the encoder to produce meaningful speech representations
(aligned to known content "一二三四五六七八九十"), regularizing the shared encoder
so the severity head learns from phonetically grounded features.

Known text "一二三四五六七八九十" → CTC vocabulary of 10 chars + blank = 11 tokens.
Target sequence is constant [1,2,...,10] for all samples (everyone counts 1→10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def sequence_mask(lengths, maxlen=None, dtype=torch.bool, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row = torch.arange(maxlen, device=lengths.device)
    mask = row[None, :] < lengths[:, None]
    if dtype == torch.float32:
        mask = mask.float()
    return mask.to(device=device) if device else mask

from funasr.frontends.wav_frontend import WavFrontend
from funasr.models.sense_voice.model import SenseVoiceEncoderSmall


def focal_loss(logits, labels, gamma=2.0, alpha=None):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss
    if alpha is not None:
        alpha_t = alpha[labels]
        focal = alpha_t * focal
    return focal.mean()


class AttentionPooling(nn.Module):
    """Learnable attention-weighted temporal pooling."""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, D], mask: [B, T] bool, True = valid
        scores = self.attn(x).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, -float('inf'))
        weights = F.softmax(scores, dim=-1)  # [B, T]
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return weighted


class SenseVoiceMultiTask(nn.Module):
    """Pretrained SenseVoice encoder + multi-task ASR (CTC) + severity.

    Input: raw waveform [B, T_wav] at 16kHz, speech_lengths [B] in samples.
    """

    # CTC token map for "一二三四五六七八九十" (blank=0)
    CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate('一二三四五六七八九十')}
    VOCAB_SIZE = 11  # blank(0) + 10 digits
    TARGET_SEQ = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]
    TARGET_LEN = 10

    def __init__(self, out_dim=4, unfreeze_last_n=0, audio_dim=64,
                 model_id='FunAudioLLM/SenseVoiceSmall',
                 input_size=560, output_size=512,
                 attention_heads=4, linear_units=2048, num_blocks=50,
                 tp_blocks=20, dropout_rate=0.1, positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0, kernel_size=11, sanm_shfit=0,
                 use_ctc=True, focal_gamma=2.0, **kwargs):
        super().__init__()

        # ---- Frontend: mel + LFR ----
        self.frontend = WavFrontend(
            fs=16000, window='hamming', n_mels=80,
            frame_length=25, frame_shift=10,
            lfr_m=7, lfr_n=6,
            dither=0.0, snip_edges=True,
        )

        # ---- Encoder: pretrained SenseVoiceEncoderSmall ----
        self.encoder = SenseVoiceEncoderSmall(
            input_size=input_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            tp_blocks=tp_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            kernel_size=kernel_size,
            sanm_shfit=sanm_shfit,
        )
        self._load_pretrained_encoder(model_id)

        # ---- Freeze / unfreeze ----
        self.unfreeze_last_n = unfreeze_last_n
        self._apply_freeze(unfreeze_last_n)

        # ---- Head A: CTC ASR (optional) ----
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_head = nn.Linear(output_size, self.VOCAB_SIZE)
            self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            # Pre-compute target sequence (same for every sample)
            self.register_buffer('ctc_targets',
                                 torch.tensor([self.TARGET_SEQ], dtype=torch.long),
                                 persistent=False)

        # ---- Head B: Severity classification ----
        self.attention_pool = AttentionPooling(output_size)
        self.severity_fc1 = nn.Linear(output_size, audio_dim)
        self.severity_dropout = nn.Dropout(0.3)
        self.severity_fc2 = nn.Linear(audio_dim, out_dim)
        self.focal_gamma = focal_gamma
        self.severity_loss_fn = lambda logits, labels: focal_loss(logits, labels, gamma=focal_gamma)

    @staticmethod
    def _find_model_path(model_id):
        """Resolve model_id to local cache path, handling various cache structures."""
        import os
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        # Try FunAudioLLM cache name first
        model_dir_name = 'models--FunAudioLLM--SenseVoiceSmall'
        potential = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(potential):
            # Check if files are in snapshots/<hash> (standard HF cache)
            refs_file = os.path.join(potential, 'refs', 'main')
            if os.path.isfile(refs_file):
                with open(refs_file) as f:
                    commit_hash = f.read().strip()
                snapshot = os.path.join(potential, 'snapshots', commit_hash)
                if os.path.isdir(snapshot):
                    return snapshot
            # Fallback: files directly in model directory root
            if os.path.isfile(os.path.join(potential, 'model.pt')):
                return potential
        # Try as given
        model_dir_name = 'models--' + model_id.replace('/', '--')
        potential = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(potential) and potential != os.path.join(cache_dir, 'models--FunAudioLLM--SenseVoiceSmall'):
            refs_file = os.path.join(potential, 'refs', 'main')
            if os.path.isfile(refs_file):
                with open(refs_file) as f:
                    commit_hash = f.read().strip()
                snapshot = os.path.join(potential, 'snapshots', commit_hash)
                if os.path.isdir(snapshot):
                    return snapshot
            if os.path.isfile(os.path.join(potential, 'model.pt')):
                return potential
        return None

    def _load_pretrained_encoder(self, model_id):
        """Load pretrained encoder weights from model.pt."""
        import os
        model_path = self._find_model_path(model_id)
        if model_path is None:
            print(f'[SenseVoiceMultiTask] Model not found at {model_id}, '
                  f'using random init (will likely underperform)')
            return

        # Try encoder_state.pth first (saved separately in previous session)
        encoder_state_path = os.path.join(model_path, 'encoder_state.pth')
        if os.path.isfile(encoder_state_path):
            state = torch.load(encoder_state_path, map_location='cpu')
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            print(f'[SenseVoiceMultiTask] Loaded encoder from {encoder_state_path}')
            if missing:
                print(f'  Missing keys: {missing}')
            if unexpected:
                print(f'  Unexpected keys: {unexpected}')
            return

        # Fallback: extract encoder from full model.pt
        model_pt_path = os.path.join(model_path, 'model.pt')
        if os.path.isfile(model_pt_path):
            full_state = torch.load(model_pt_path, map_location='cpu')
            # Filter encoder keys
            encoder_state = {}
            for k, v in full_state.items():
                if k.startswith('encoder.'):
                    encoder_state[k[len('encoder.'):]] = v
            if encoder_state:
                missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
                print(f'[SenseVoiceMultiTask] Loaded encoder from {model_pt_path}')
                if missing:
                    print(f'  Missing keys ({len(missing)}): {missing[:5]}...')
                if unexpected:
                    print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
                return

        print(f'[SenseVoiceMultiTask] No weights found at {model_path}')

    def _apply_freeze(self, unfreeze_last_n):
        """Apply freeze/unfreeze to encoder based on unfreeze_last_n.

        Args:
            unfreeze_last_n: 0=freeze all, -1=unfreeze all,
                             N>0=unfreeze last N encoder blocks
        """
        # Freeze all encoder params by default
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        if unfreeze_last_n == -1:
            # Full fine-tuning
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder.train()
            print(f'[SenseVoiceMultiTask] Unfroze ALL encoder parameters')
            return

        if unfreeze_last_n == 0:
            print(f'[SenseVoiceMultiTask] Encoder fully frozen')
            return

        # Collect encoder blocks in order: encoders0[0] + encoders[0..N-2] + tp_encoders[0..tp-1]
        all_blocks = nn.ModuleList()
        all_blocks.extend(self.encoder.encoders0)   # 1 block: 560→512
        all_blocks.extend(self.encoder.encoders)     # 49 blocks
        all_blocks.extend(self.encoder.tp_encoders)  # 20 blocks
        total = len(all_blocks)  # 70

        n = min(unfreeze_last_n, total)
        if n > 0:
            for block in all_blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Also unfreeze the layer norms after main encoders
            for p in self.encoder.after_norm.parameters():
                p.requires_grad = True
            for p in self.encoder.tp_norm.parameters():
                p.requires_grad = True
            print(f'[SenseVoiceMultiTask] Unfroze last {n}/{total} encoder blocks '
                  f'(+ norm layers)')

    def _compute_frontend(self, speech, speech_lengths):
        """Convert raw audio to mel+LFR features.

        Args:
            speech: [B, T_max] raw waveform
            speech_lengths: [B] length of each utterance in samples
        Returns:
            feats: [B, T', 560] LFR features
            feat_lens: [B] length of each feature sequence
        """
        batch_size = speech.size(0)
        feats_list = []
        feat_lens_list = []
        for i in range(batch_size):
            wav_len = speech_lengths[i]
            wav = speech[i, :wav_len]
            # WavFrontend expects [1, T] and returns [T', D]
            mat = self.frontend(wav.unsqueeze(0),
                                torch.tensor([wav_len], device=speech.device))
            feat, feat_len = mat  # [1, T', 560], [1]
            feats_list.append(feat[0])
            feat_lens_list.append(feat_len[0])

        feat_lens = torch.stack(feat_lens_list)  # [B]
        feats_pad = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
        # Frontend returns CPU tensors, move to input device
        feats_pad = feats_pad.to(device=speech.device)
        feat_lens = feat_lens.to(device=speech.device)
        return feats_pad, feat_lens

    def forward(self, speech, video, speech_lengths, label, video_masks=None, **kwargs):
        """Forward pass.

        Args:
            speech: [B, T_max] raw waveform at 16kHz
            speech_lengths: [B] length of each utterance in samples
            label: [B] severity label (0-3)
        Returns:
            dict with 'loss', 'logits', and optional auxiliary losses
        """
        # ---- Frontend ----
        feats, feat_lens = self._compute_frontend(speech, speech_lengths)
        # feats: [B, T', 560], feat_lens: [B]

        # ---- Encoder ----
        with torch.set_grad_enabled(self.unfreeze_last_n != 0):
            encoder_out, olens = self.encoder(feats, feat_lens)
        # encoder_out: [B, T', 512], olens: [B]

        # ---- Head A: CTC ASR (optional) ----
        if self.use_ctc:
            ctc_logits = self.ctc_head(encoder_out)  # [B, T', 11]
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

            targets = self.ctc_targets.expand(speech.size(0), -1)  # [B, 10]
            target_lens = torch.full((speech.size(0),), self.TARGET_LEN,
                                     dtype=torch.long, device=speech.device)

            ctc_loss = self.ctc_loss_fn(
                ctc_log_probs.permute(1, 0, 2),  # [T', B, 11]
                targets,           # [B, 10]
                olens.int(),       # [B]
                target_lens,       # [B]
            )

        # ---- Head B: Severity classification ----
        mask = sequence_mask(olens, maxlen=encoder_out.size(1))  # [B, T']
        pooled = self.attention_pool(encoder_out, mask)  # [B, 512]
        severity_out = self.severity_dropout(F.relu(self.severity_fc1(pooled)))
        severity_logits = self.severity_fc2(severity_out)  # [B, 4]
        severity_loss = self.severity_loss_fn(severity_logits, label)

        total_loss = severity_loss
        if self.use_ctc:
            total_loss = total_loss + ctc_loss

        out = {
            'loss': total_loss,
            'logits': severity_logits.detach(),
            'severity_loss': severity_loss.detach(),
        }
        if self.use_ctc:
            out['ctc_loss'] = ctc_loss.detach()
        return out

    def decode(self, speech, video, speech_lengths, video_masks=None):
        """Inference: return severity probabilities.

        Args:
            speech: [B, T_max] raw waveform
            speech_lengths: [B] in samples
        Returns:
            probabilities: [B, 4] softmax over severity classes
        """
        feats, feat_lens = self._compute_frontend(speech, speech_lengths)

        with torch.set_grad_enabled(self.unfreeze_last_n != 0):
            encoder_out, olens = self.encoder(feats, feat_lens)

        mask = sequence_mask(olens, maxlen=encoder_out.size(1))
        pooled = self.attention_pool(encoder_out, mask)
        severity_out = self.severity_dropout(F.relu(self.severity_fc1(pooled)))
        severity_logits = self.severity_fc2(severity_out)

        return torch.softmax(severity_logits, dim=1)
