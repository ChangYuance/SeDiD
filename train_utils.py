#!/usr/bin/env python3
"""Shared training helpers: device, LR schedule, and the LFC-x stage switch."""
from __future__ import annotations
import math
import torch

from model import SenseVoiceMultiAnnotator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    warmup_ratio: float = 0.2,
) -> torch.optim.lr_scheduler.LambdaLR:
    """CosineAnnealing with linear warmup."""
    warmup_steps = int(n_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, n_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _set_lfcx_stage(model: SenseVoiceMultiAnnotator, stage: int):
    """LFC-x two-stage training (Li et al. 2021, §3.3):
    Stage 1: freeze instance_impact, keep confusion_logits trainable (degenerates to CrowdLayer)
    Stage 2: freeze confusion_logits, keep instance_impact + classifier trainable
    """
    for name, p in model.named_parameters():
        if "M_generator.instance_impact" in name:
            p.requires_grad = (stage == 2)
        elif "M_generator.confusion_logits" in name:
            p.requires_grad = (stage == 1)
        else:
            p.requires_grad = True
