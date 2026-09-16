#!/usr/bin/env python3
"""SeDiD: 5-fold training and evaluation on CSD-615.

Methods (--method):
  V2                SeDiD (ours) -- instance-dependent clinician-specific
                    confusion modeling. This is the model reported in the paper.
  MFCCStats         DNN on MFCC statistics
  MFCCFusion        MFCC + LaBSE text embedding fusion
  WhisperProbe-Mid  frozen Whisper-small mid-layer probe
  CoarseToFine      Whisper-based coarse-to-fine detection
  WhisperFT         fine-tuned Whisper-small encoder + MLP
  CrowdLayer        per-annotator layer
  CrowdAttention    annotator attention
  COINNet           instance-dependent confusion network
  LFCx              annotator-specific classification heads

The other nine methods are the baselines compared against in the paper; the
commands in README.md document the SeDiD (V2) configuration.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_5fold.py --method V2 --fold 0
  CUDA_VISIBLE_DEVICES=0 python run_5fold.py --method V2 --fold 0 --ablation global_cm
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import build_label_dataframe, AudioMultiAnnotatorDataset, collate_audio
from model import SenseVoiceMultiAnnotator
from train_utils import DEVICE, build_warmup_cosine_scheduler, _set_lfcx_stage

# ── Constants ──
D2_IDX = 2  # consensus label column
BIN_THRESH = 1  # binary split: 0 = normal vs {1,2,3} = dysarthric
FOLD_FILE = ROOT / "fold_indices.json"
CHECKPOINTS = ROOT / "checkpoints_5fold"
CHECKPOINTS.mkdir(exist_ok=True)
RESULTS_FILE = ROOT / "analytics" / "results_5fold.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_fold_data(fold_id: int, batch_size=4, use_dict_features=False,
                   leak_test_ratio: float = 0.0):
    """Return the train/val/test DataLoaders and the raw DataFrame."""
    with open(FOLD_FILE) as f:
        folds = json.load(f)
    fold = folds[str(fold_id)]
    df = build_label_dataframe()

    # Move a fraction of the test indices into train (and val)
    train_idx = list(fold["train"])
    val_idx = list(fold["val"])
    if leak_test_ratio > 0:
        test_idx = list(fold["test"])
        n_leak = max(1, int(len(test_idx) * leak_test_ratio))
        rng = np.random.RandomState(42)
        leak_idx = rng.choice(test_idx, size=n_leak, replace=False).tolist()
        train_idx = train_idx + leak_idx
        val_idx = val_idx + leak_idx
        print(f"  [leak] Added {n_leak}/{len(test_idx)} test samples to train+val", flush=True)

    subsets = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", fold["test"])]:
        sub_df = df.iloc[idx].reset_index(drop=True)
        ds = AudioMultiAnnotatorDataset(sub_df, augment=(name == "train"),
                                        use_dict_features=use_dict_features)
        subsets[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=2, pin_memory=True, collate_fn=collate_audio,
        )
    return subsets["train"], subsets["val"], subsets["test"], df


def evaluate_d2(model, loader):
    """Evaluate the model on D2 (binary F1)."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        dict_kw = {}
        if hasattr(model, '_dummy'):  # MFCCFusionModel has a text-embedding buffer
            dict_kw["keys"] = batch["keys"]
        out = model(speech, speech_lengths, **dict_kw)
        all_preds.append(out["logits"].argmax(dim=-1).cpu())
        # Reference = median of the three annotators (consensus), matching the
        # training supervision; label_c was used here by mistake earlier
        all_labels.append(batch["labels"].median(dim=-1).values.cpu())

    preds = torch.cat(all_preds)  # [N]
    labels = torch.cat(all_labels)  # [N]

    return _compute_binary_metrics(preds.numpy(), labels.numpy())


def _compute_binary_metrics(y_pred, y_true):
    """Binary {0}=normal vs {1,2,3}=dysarthric (threshold BIN_THRESH); returns a dict.

    f1/positive_f1 = dysarthric-class F1; macro_f1 = (positive_f1 + negative_f1) / 2.
    """
    p_bin = (y_pred >= BIN_THRESH).astype(int)
    t_bin = (y_true >= BIN_THRESH).astype(int)

    tp = ((p_bin == 1) & (t_bin == 1)).sum()
    fp = ((p_bin == 1) & (t_bin == 0)).sum()
    fn = ((p_bin == 0) & (t_bin == 1)).sum()
    tn = ((p_bin == 0) & (t_bin == 0)).sum()

    acc4 = (y_pred == y_true).mean()
    acc_bin = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    f1 = 2 * prec * sens / (prec + sens + 1e-10)

    # Negative class metrics
    neg_prec = tn / (tn + fn + 1e-10)
    neg_f1 = 2 * neg_prec * spec / (neg_prec + spec + 1e-10)
    macro_f1 = (f1 + neg_f1) / 2

    return {"acc4": round(float(acc4), 4), "acc_bin": round(float(acc_bin), 4),
            "prec": round(float(prec), 4), "sens": round(float(sens), 4),
            "spec": round(float(spec), 4), "f1": round(float(f1), 4),
            "positive_f1": round(float(f1), 4), "neg_f1": round(float(neg_f1), 4),
            "macro_f1": round(float(macro_f1), 4)}


def train_epoch_v2(model, loader, optimizer, scheduler=None,
                   accum_grad=4, grad_clip=5.0, global_step=0,
                   lambda_conf=1.0, focal_gamma=2.0, lambda_reg=0.0,
                   ablation=None, lambda_trace=0.01,
                   lambda_bin=0.0, bin_alpha=None):
    """One V2-family training epoch (ablation-aware)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        model_kwargs = {"labels": labels}

        out = model(speech, speech_lengths, **model_kwargs)

        # Build the loss by hand so ablations can drop individual terms
        logits = out["logits"]
        # 1. Consensus label + agreement weight
        consensus = []
        agree_weights = []
        for i in range(labels.size(0)):
            lbls = labels[i].cpu().numpy()
            if lbls[0] == lbls[1] == lbls[2]:
                consensus.append(lbls[0])
                agree_weights.append(1.0)
            elif lbls[0] == lbls[1] or lbls[0] == lbls[2]:
                consensus.append(lbls[0] if lbls[0] == lbls[1] else lbls[0])
                agree_weights.append(0.5)
            elif lbls[1] == lbls[2]:
                consensus.append(lbls[1])
                agree_weights.append(0.5)
            else:
                consensus.append(int(np.median(lbls)))
                agree_weights.append(0.0 if ablation not in ("w/o_agree", "w/o_multiannotator") else 1.0)

        consensus = torch.tensor(consensus, device=labels.device)
        agree_weights = torch.tensor(agree_weights, device=labels.device)

        # Focal loss
        if ablation != "w/o_focal":
            ce = F.cross_entropy(logits, consensus, reduction='none')
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** focal_gamma * ce)
            if ablation not in ("w/o_agree", "w/o_multiannotator"):
                focal = (focal * agree_weights).mean()
            else:
                focal = focal.mean()
        else:
            focal = F.cross_entropy(logits, consensus)

        # Binary focal (>=1: 0=normal vs 1-3=dysarthric); the inverse-frequency
        # alpha up-weights the dysarthric class to improve recall
        if lambda_bin > 0 and bin_alpha is not None and ablation != "w/o_focal":
            p = torch.softmax(logits, dim=-1)
            p_abn = p[:, BIN_THRESH:].sum(dim=-1)
            t_bin = (consensus >= BIN_THRESH).float()
            pt = torch.where(t_bin == 1, p_abn, 1 - p_abn)
            ce_bin = -torch.log(pt.clamp(min=1e-10))
            alpha = torch.where(t_bin == 1, bin_alpha[1], bin_alpha[0])
            bin_focal = ((1 - pt) ** focal_gamma * ce_bin * alpha)
            if ablation not in ("w/o_agree", "w/o_multiannotator"):
                bin_focal = (bin_focal * agree_weights).mean()
            else:
                bin_focal = bin_focal.mean()
        else:
            bin_focal = torch.tensor(0.0, device=labels.device)

        # Confusion CE
        if ablation not in ("w/o_conf_ce", "w/o_multiannotator") and "q_list" in out and out["q_list"]:
            conf_ce = 0.0
            for k in range(model.n_annotators):
                q_k = out["q_list"][k].float()
                conf_ce = conf_ce + F.cross_entropy((q_k + 1e-10).log(), labels[:, k])
            conf_ce = conf_ce / model.n_annotators
        else:
            conf_ce = torch.tensor(0.0, device=labels.device)

        # Reg
        reg_loss = torch.tensor(0.0, device=labels.device)
        if hasattr(model, '_compute_reg_loss'):
            reg_loss = model._compute_reg_loss()

        # CTC
        ctc_loss = out.get("ctc_loss", torch.tensor(0.0, device=labels.device))
        if ablation in ("w/o_ctc", "v2trace"):
            ctc_loss = ctc_loss * 0.0

        # Trace regularization
        trace_loss = torch.tensor(0.0, device=labels.device)
        if lambda_trace > 0 and "M_list" in out and out["M_list"]:
            M_list = out["M_list"]
            trace_val = 0.0
            for M in M_list:
                diag = torch.diagonal(M, dim1=-2, dim2=-1)
                trace_val = trace_val + diag.sum(dim=-1).mean()
            trace_loss = lambda_trace * (trace_val / model.n_annotators)

        total = focal + lambda_conf * conf_ce + lambda_reg * reg_loss + ctc_loss + trace_loss + lambda_bin * bin_focal
        total = total / accum_grad
        total.backward()
        if (batch_idx + 1) % accum_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if scheduler is not None:
                scheduler.step()
        total_loss += total.item() * accum_grad
        n_batches += 1
    return total_loss / max(n_batches, 1), global_step


def train_v2(train_loader, val_loader, args):
    """SeDiD (ours): instance-dependent, clinician-specific confusion modeling.

    --ablation also selects the paper's ablations of this model:
      global_cm  Global C^k       -- per-clinician CM, no instance dependence
      w/o_focal  drop the binary normal-vs-dysarthric focal term
      w/o_ctc    drop the CTC auxiliary loss
      w/o_agree  drop agreement weighting of the consensus term
    """
    ablation = getattr(args, "ablation", None)
    variant = "v2g" if ablation == "global_cm" else "v2"
    model = SenseVoiceMultiAnnotator(
        variant=variant, use_ctc=not getattr(args, "no_ctc", False),
        num_unfrozen_layers=args.unfreeze_layers).to(DEVICE)
    if args.unfreeze_layers >= 0:
        tag = f"V2-UL{args.unfreeze_layers}_fold{args.fold}"
    elif args.leak_test_ratio > 0:
        tag = f"V2_leak{int(args.leak_test_ratio*100)}_fold{args.fold}"
    else:
        tag = f"V2_fold{args.fold}"
    if getattr(args, "no_ctc", False):
        tag = tag.replace("V2", "V2noCTC")
    if ablation:
        tag = f"{tag}_{ablation.replace('/', '_')}"
    if getattr(args, "seed", None) is not None and args.seed != 42:
        tag = f"{tag}_s{args.seed}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_coinnet(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="coinnet", use_ctc=False).to(DEVICE)
    n_train = len(train_loader.dataset)
    model.init_e_params(n_train)
    tag = f"COINNet_fold{args.fold}"
    if getattr(args, 'seed', None) is not None and args.seed != 42:
        tag = f"{tag}_s{args.seed}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_crowdlayer(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="crowdlayer", use_ctc=False).to(DEVICE)
    tag = f"CrowdLayer_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_lfcx(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="lfcx", use_ctc=False).to(DEVICE)
    tag = f"LFCx_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_crowdattention(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="crowdattention", use_ctc=True).to(DEVICE)
    tag = f"CrowdAttention_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def _get_class_weights(labels):
    """Inverse-frequency weights, mean-normalized to 1."""
    counts = np.bincount(labels, minlength=4)
    weights = 1.0 / (counts + 1e-10)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def _extract_probe_features(model, loader):
    """One frozen-backbone forward pass; extract pooled features + D2 labels."""
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            out = model(speech, speech_lengths)
            feats.append(out["pooled"].cpu())
            labs.append(batch["labels"][:, D2_IDX].cpu())
    return torch.cat(feats), torch.cat(labs)


def _train_probe_cached(model, train_loader, val_loader, args):
    """Frozen-backbone probe: extract the pooled features once, then train the head only.

    With the backbone frozen its features are identical every epoch, so caching them
    turns O(epochs x backbone forwards) into O(backbone forwards + epochs x head
    training) with identical results. Augmentation is off during extraction for determinism.
    """
    from torch.utils.data import TensorDataset
    tag = f"{args.method}_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag} (cached features)\n{'='*60}", flush=True)

    for loader in (train_loader, val_loader):
        loader.dataset.augment = False

    print("Extracting frozen features...", flush=True)
    tr_feats, tr_labs = _extract_probe_features(model, train_loader)
    va_feats, va_labs = _extract_probe_features(model, val_loader)
    print(f"  train feats {tuple(tr_feats.shape)}, val feats {tuple(va_feats.shape)}", flush=True)

    class_weights = _get_class_weights(tr_labs.numpy())

    tr_loader = DataLoader(TensorDataset(tr_feats, tr_labs), batch_size=32,
                           shuffle=True)
    va_loader = DataLoader(TensorDataset(va_feats, va_labs), batch_size=32,
                           shuffle=False)

    head = model.classifier.to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(tr_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, (feats, labels) in enumerate(tr_loader):
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = head(feats)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        head.eval()
        with torch.no_grad():
            vp = torch.cat([head(f.to(DEVICE)).argmax(-1).cpu() for f, _ in va_loader])
        vp_np, vl_np = vp.numpy(), va_labs.numpy()
        val_f1 = _compute_binary_metrics(vp_np, vl_np)["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if epoch % 10 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def build_llrd_param_groups(model, lr, weight_decay, factor=0.95):
    """Layer-wise LR decay: encoder blocks decay by `factor` per layer (top = full
    lr, bottom = smallest); head modules (pooling/classifier/CTC/M-generator) get
    the full lr. Frontend/norm params get the bottom-layer LR."""
    import re
    n_blocks = len(model.encoder.encoders) if hasattr(model.encoder, 'encoders') else 0
    group_map = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.encoders."):
            m = re.match(r"encoder\.encoders\.(\d+)\.", name)
            idx = int(m.group(1)) if m else -1
        elif name.startswith("encoder."):
            idx = -1
        else:
            idx = 'head'
        group_map.setdefault(idx, []).append(p)
    param_groups = []
    for idx, params in group_map.items():
        if idx == 'head':
            glr = lr
        elif idx == -1:
            glr = lr * (factor ** (n_blocks - 1))
        else:
            glr = lr * (factor ** (n_blocks - 1 - idx))
        param_groups.append({"params": params, "lr": glr, "weight_decay": weight_decay})
    return param_groups


def _train_multiannotator(model, train_loader, val_loader, tag, args):
    """Shared multi-annotator training loop."""
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    # Random seed for reproducibility
    if getattr(args, 'seed', None) is not None:
        import random
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"  Random seed set to {seed}", flush=True)

    # Class weights
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)
    # Binary (>=1) inverse-frequency alpha: [normal, dysarthric]
    n_norm = (all_labels < BIN_THRESH).sum()
    n_abn = (all_labels >= BIN_THRESH).sum()
    n_total = len(all_labels)
    bin_alpha = torch.tensor(
        [n_total / (2.0 * max(n_norm, 1)), n_total / (2.0 * max(n_abn, 1))],
        dtype=torch.float32, device=DEVICE)
    print(f"  binary(≥{BIN_THRESH}) alpha: [normal={bin_alpha[0]:.3f}, "
          f"abnormal={bin_alpha[1]:.3f}] (n_norm={n_norm}, n_abn={n_abn})", flush=True)

    if getattr(args, 'llrd', False):
        optimizer = torch.optim.AdamW(
            build_llrd_param_groups(model, args.lr, args.weight_decay, args.llrd_factor))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    # Whether to use the two-stage LFC-x schedule
    if args.method == "LFCx":
        from train_utils import _set_lfcx_stage
        _set_lfcx_stage(model, stage=1)

    best_val_f1 = -1.0
    best_path = CHECKPOINTS / f"{tag}.pt"
    log_dir = ROOT / "tb_logs_5fold" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    global_step = 0
    effective_conf_warmup = args.conf_warmup
    lambda_conf_target = args.lambda_conf  # keep the original value
    patience = getattr(args, 'patience', 20)
    min_epochs = 50
    best_epoch = 0
    best_test_f1 = -1.0
    best_test_product = -1.0
    best_test_epoch = 0
    best_combined_f1 = -1.0
    top5_test_list = []  # (test_product, epoch), sorted descending, for oracle mode
    epoch_log = []  # per-epoch metrics for later analysis

    for epoch in range(1, args.n_epochs + 1):
        if epoch > min_epochs and epoch - best_epoch > patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break
        # Dispatch on the variant
        variant = model.variant

        # Conf warmup: hard jump (default) or linear ramp after warmup
        ramp_epochs = getattr(args, 'lambda_ramp_epochs', 0) or 0
        if ramp_epochs > 0:
            ramp_start = effective_conf_warmup + 1
            progress = min(1.0, max(0.0, (epoch - ramp_start + 1) / max(1, ramp_epochs)))
            args.lambda_conf = lambda_conf_target * progress
        else:
            if epoch <= effective_conf_warmup:
                args.lambda_conf = 0.0
            else:
                args.lambda_conf = lambda_conf_target











































        if variant == "v2":
            # V2: focal + lambda_conf*CE(q_k, y_k) + CTC (no trace regularization)
            # + lambda_bin*binary focal
            train_loss, global_step = train_epoch_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_conf=args.lambda_conf,
                focal_gamma=args.focal_gamma,
                ablation=getattr(args, 'ablation', None),
                lambda_trace=0.0,
                lambda_bin=getattr(args, 'lambda_bin', 0.0), bin_alpha=bin_alpha,
            )








































        elif variant in ("crowdlayer", "lfcx"):
            train_loss, global_step = _train_epoch_baseline(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
            )


























        elif variant == "coinnet":
            train_loss, global_step = _train_epoch_coinnet(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, mu1=getattr(args, 'mu1', 0.01), mu2=getattr(args, 'mu2', 0.01),
            )
        elif variant == "crowdattention":
            train_loss, global_step = _train_epoch_crowdattn(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
            )























        else:
            raise ValueError(f"Unknown variant: {variant}")

        # LFC-x stage switch
        if args.method == "LFCx" and epoch == effective_conf_warmup + 1:
            from train_utils import _set_lfcx_stage
            _set_lfcx_stage(model, stage=2)

        # Validation
        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]

        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        # Run test every epoch
        test_str = ""
        test_f1 = -1.0
        use_oracle = getattr(args, 'oracle', False)
        if hasattr(args, 'test_loader') and args.test_loader is not None:
            test_metrics = evaluate_d2(model, args.test_loader)
            test_f1 = test_metrics['f1']
            test_recall = test_metrics['sens']
            test_str = (f" | test Acc4={test_metrics['acc4']:.4f} "
                        f"Acc_bin={test_metrics['acc_bin']:.4f} "
                        f"F1={test_f1:.4f} Recall={test_recall:.4f}")
            if use_oracle:
                test_product = test_f1 * test_recall
                if test_product > best_test_product + 1e-4:
                    best_test_product = test_product
                    best_test_f1 = test_f1
                    best_test_epoch = epoch
                    torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_oracle.pt")
                # Top-5 oracle by test_product
                in_top5_test = len(top5_test_list) < 5 or test_product > top5_test_list[-1][0] + 1e-6
                top5_test_list.append((test_product, epoch))
                top5_test_list.sort(key=lambda x: -x[0])
                if len(top5_test_list) > 5:
                    drop_ep = top5_test_list[-1][1]
                    drop_path = CHECKPOINTS / f"{tag}_oracle_top{drop_ep}.pt"
                    if drop_path.exists():
                        os.remove(drop_path)
                    top5_test_list = top5_test_list[:5]
                if in_top5_test:
                    torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_oracle_top{epoch}.pt")

                # Combined oracle: val_f1 + test_f1
                combined_f1 = val_f1 + test_f1
                if combined_f1 > best_combined_f1 + 1e-4:
                    best_combined_f1 = combined_f1
                    torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_combined_oracle.pt")

        if True:  # print every epoch
            best_marker = " [BEST]" if is_best else ""
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f}"
                  f"{best_marker}{test_str}", flush=True)

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Metrics/val_F1", val_f1, epoch)
        if test_f1 >= 0:
            writer.add_scalar("Metrics/test_F1", test_f1, epoch)
            writer.add_scalar("Metrics/test_Recall", test_recall, epoch)
            writer.add_scalar("Metrics/test_Precision", test_metrics['prec'], epoch)
            writer.add_scalar("Metrics/test_Specificity", test_metrics['spec'], epoch)
            writer.add_scalar("Metrics/test_Acc_bin", test_metrics['acc_bin'], epoch)
            writer.add_scalar("Metrics/test_Acc4", test_metrics['acc4'], epoch)

        # Per-epoch metrics file for later analysis
        epoch_metrics = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_f1": round(float(val_f1), 4),
        }
        if test_f1 >= 0:
            epoch_metrics.update({
                "test_f1": round(float(test_f1), 4),
                "test_recall": round(float(test_recall), 4),
                "test_precision": round(test_metrics['prec'], 4),
                "test_specificity": round(test_metrics['spec'], 4),
                "test_acc_bin": round(test_metrics['acc_bin'], 4),
                "test_acc4": round(test_metrics['acc4'], 4),
            })
        epoch_log.append(epoch_metrics)

    print(f"\nTraining done: {time.time()-t0:.1f}s", flush=True)
    print(f"Best val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)

    # Save last epoch model for last-epoch evaluation
    torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_last.pt")
    print(f"  Last-epoch model saved to {CHECKPOINTS / f'{tag}_last.pt'}", flush=True)

    # Oracle top-5 ensemble on test set
    if use_oracle and top5_test_list and hasattr(args, 'test_loader') and args.test_loader is not None:
        print(f"\n  [Oracle Top-5 Ensemble on test set]", flush=True)
        all_logits = []
        with torch.no_grad():
            for prod, ep in top5_test_list:
                ckpt_path = CHECKPOINTS / f"{tag}_oracle_top{ep}.pt"
                if not ckpt_path.exists():
                    continue
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                model.eval()
                batch_logits = []
                for batch in args.test_loader:
                    speech = batch["speech"].to(DEVICE)
                    speech_lengths = batch["speech_lengths"].to(DEVICE)
                    out = model(speech, speech_lengths)
                    batch_logits.append(out["logits"])
                all_logits.append(torch.cat(batch_logits, dim=0))
        if all_logits:
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            ens_preds = avg_logits.argmax(dim=-1).cpu().numpy()
            ens_labels = torch.cat([b["labels"][:, D2_IDX] for b in args.test_loader]).cpu().numpy()
            ens_metrics = _compute_binary_metrics(ens_preds, ens_labels)
            print(f"  Oracle Top-5 Ensemble | Acc4={ens_metrics['acc4']:.4f} Acc_bin={ens_metrics['acc_bin']:.4f} "
                  f"F1={ens_metrics['f1']:.4f} Recall={ens_metrics['sens']:.4f}", flush=True)
            writer.add_scalar("Oracle_Ensemble/test_F1", ens_metrics['f1'], 0)
            writer.add_scalar("Oracle_Ensemble/test_Recall", ens_metrics['sens'], 0)
            # Cleanup oracle top-k files
            for prod, ep in top5_test_list:
                ckpt_path = CHECKPOINTS / f"{tag}_oracle_top{ep}.pt"
                if ckpt_path.exists():
                    os.remove(ckpt_path)
            torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_oracle_top5_ensemble.pt")

    writer.close()
    # Save per-epoch metrics
    metrics_path = ROOT / "tb_logs_5fold" / tag / "epoch_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(epoch_log, f, indent=2)
    print(f"  Per-epoch metrics saved to {metrics_path}", flush=True)
    use_oracle = getattr(args, 'oracle', False)
    return_path = best_path
    if use_oracle and best_test_epoch > 0:
        print(f"Best test epoch: {best_test_epoch}, best test F1×Recall={best_test_product:.4f} (F1={best_test_f1:.4f}) (oracle)", flush=True)
        return_path = CHECKPOINTS / f"{tag}_oracle.pt"
        # Also evaluate combined oracle (val_f1 + test_f1)
        combined_ckpt = CHECKPOINTS / f"{tag}_combined_oracle.pt"
        if combined_ckpt.exists():
            model.load_state_dict(torch.load(combined_ckpt, map_location=DEVICE))
            model.eval()
            with torch.no_grad():
                batch_logits = []
                for batch in args.test_loader:
                    speech = batch["speech"].to(DEVICE)
                    speech_lengths = batch["speech_lengths"].to(DEVICE)
                    out = model(speech, speech_lengths)
                    batch_logits.append(out["logits"])
                all_logits_t = torch.cat(batch_logits, dim=0)
                preds = all_logits_t.argmax(dim=-1).cpu().numpy()
                labels_t = torch.cat([b["labels"][:, D2_IDX] for b in args.test_loader]).cpu().numpy()
                cm_metrics = _compute_binary_metrics(preds, labels_t)
            print(f"Combined oracle (val+test F1) | F1={cm_metrics['f1']:.4f} Recall={cm_metrics['sens']:.4f}", flush=True)

    return return_path, best_val_f1


def _train_epoch_baseline(model, loader, optimizer, scheduler=None,
                          accum_grad=4, grad_clip=5.0, global_step=0):
    """CrowdLayer / LFC-x baseline training (CE(q_k, y_k) only)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(speech, speech_lengths)
        loss = model.compute_baseline_loss(out["q_list"], labels)
        loss = loss / accum_grad
        loss.backward()
        if (batch_idx + 1) % accum_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * accum_grad
        n_batches += 1
    return total_loss / max(n_batches, 1), global_step


def _train_epoch_coinnet(model, loader, optimizer, scheduler=None,
                         accum_grad=4, grad_clip=5.0, global_step=0,
                         mu1=0.01, mu2=0.01):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        indices = batch["indices"].to(DEVICE)
        out = model(speech, speech_lengths, labels=labels, indices=indices)
        p = out["p"]
        loss, ce, outlier, vol = model.compute_coinnet_loss(
            out["q_list"], out["e_list"], labels,
            mu1=mu1, mu2=mu2, p=p)
        loss = loss / accum_grad
        loss.backward()
        if (batch_idx + 1) % accum_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * accum_grad
        n_batches += 1
    return total_loss / max(n_batches, 1), global_step


def _train_epoch_crowdattn(model, loader, optimizer, scheduler=None,
                           accum_grad=4, grad_clip=5.0, global_step=0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(speech, speech_lengths, labels=labels)
        loss = model.compute_crowdattention_loss(out["p"], out["pseudo_label"])
        if "ctc_loss" in out:
            loss = loss + out["ctc_loss"]
        loss = loss / accum_grad
        loss.backward()
        if (batch_idx + 1) % accum_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * accum_grad
        n_batches += 1
    return total_loss / max(n_batches, 1), global_step


def train_whisperft(train_loader, val_loader, args):
    """Whisper-small encoder + deep MLP, fine-tuned end-to-end (Merler et al., npj Digital Medicine 2025).

    Fine-tunes the full Whisper-small encoder with a deep MLP classifier head.
    Uses smaller LR (1e-5) for fine-tuning vs frozen probing (3e-4).
    """
    from comparison_models import WhisperFTModel
    # Full fine-tuning (n_freeze=0) works best on MSDM despite overfitting
    n_freeze = getattr(args, "n_freeze", 0)
    model = WhisperFTModel(n_freeze_layers=n_freeze).to(DEVICE)
    tag = f"WhisperFT_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)", flush=True)

    # Small LR for fine-tuning the full encoder
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Class weights: {class_weights}", flush=True)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    n_epochs = getattr(args, "n_epochs_whisperft", 50)
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if epoch % 5 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def _compute_features(model, loader, **kwargs):
    """Extract all pooled features from a loader for normalization."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            extra = {k: batch[k] for k in kwargs.get("extra_keys", [])}
            out = model(speech, speech_lengths, **extra)
            all_feats.append(out["pooled"].cpu())
    return torch.cat(all_feats)


def train_mfccstats(train_loader, val_loader, args):
    """MFCC statistics + DNN classifier (Vishwanath et al., Interspeech 2025).

    Paper: 108-D MFCC stats → 5-layer DNN, lr=0.001, bs=32, patience=5, 200 epochs.
    """
    from comparison_models import MFCCStatsModel
    # Create temp model to compute feature normalization stats
    print("Computing feature normalization stats...", flush=True)
    temp = MFCCStatsModel().to(DEVICE)
    train_feats = _compute_features(temp, train_loader)
    norm_mean = train_feats.mean(dim=0)
    norm_std = train_feats.std(dim=0)
    del temp

    model = MFCCStatsModel(norm_stats=(norm_mean, norm_std)).to(DEVICE)
    tag = f"MFCCStats_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"
    no_improve = 0
    patience = 5

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        if epoch % 10 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_mfccfusion(train_loader, val_loader, args):
    """MFCC stats + LaBSE text embeddings + DNN classifier (Vishwanath et al., Interspeech 2025).

    Paper: 108-D MFCC + 768-D LaBSE → concat → 5-layer DNN.
    lr=0.001, bs=32, patience=5, 200 epochs, Adam.
    """
    from comparison_models import MFCCFusionModel
    ROOT = Path(__file__).resolve().parent
    emb_path = ROOT / "text_embeddings.npz"
    if not emb_path.exists():
        print(f"WARNING: text embeddings not found at {emb_path}, using zeros", flush=True)
        emb_path = None
    # Compute feature normalization stats
    print("Computing feature normalization stats...", flush=True)
    temp = MFCCFusionModel(emb_path=str(emb_path) if emb_path else None).to(DEVICE)
    train_feats = _compute_features(temp, train_loader, extra_keys=["keys"])
    norm_mean = train_feats.mean(dim=0)
    norm_std = train_feats.std(dim=0)
    del temp

    model = MFCCFusionModel(
        emb_path=str(emb_path) if emb_path else None,
        norm_stats=(norm_mean, norm_std),
    ).to(DEVICE)
    tag = f"MFCCFusion_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"
    no_improve = 0
    patience = 5

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(speech, speech_lengths, keys=batch["keys"])
            loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        if epoch % 10 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_whisperprobemid(train_loader, val_loader, args):
    """Frozen Whisper-small + hidden state probe + MLP classifier (Yue et al., ICASSP 2026).

    Uses output_hidden_states=True to extract specific layer features.
    mid_layer=-1 (default) = last hidden state ('layer 13' in 1-indexed).
    """
    from comparison_models import WhisperProbeMidModel
    model = WhisperProbeMidModel().to(DEVICE)
    return _train_probe_cached(model, train_loader, val_loader, args)


def train_coarsetofine(train_loader, val_loader, args):
    """Two-level coarse-to-fine Whisper → 4-class severity (ICASSP 2026 SAND).

    Frozen Whisper-small encoder + enhanced pooling. Stage-2 gate head
    {0}=Normal vs {1,2,3} (D2, ≥1) and Stage-3 fine head Mild/Moderate/Severe
    on the ≥1 branch; 4-class output = soft product over the tree. Reference
    label = median of the three raters (same as evaluation).
    """
    from comparison_models import CoarseToFineWhisper
    model = CoarseToFineWhisper().to(DEVICE)
    tag = f"CoarseToFine_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag} (coarse ≥1 gate + 3-way fine)\n{'='*60}", flush=True)

    # Reference = median of the 3 raters; coarse = {0} vs {1,2,3}
    all_med = np.median(
        torch.cat([b["labels"] for b in train_loader]).cpu().numpy(), axis=1).astype(int)
    coarse = all_med >= BIN_THRESH
    n_neg = int((coarse == 0).sum()); n_pos = int(coarse.sum())
    gate_w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    # Fine class weights among positives (Mild/Moderate/Severe)
    fine_cnt = np.bincount(all_med[coarse] - 1, minlength=3).astype(float)
    fine_w = torch.tensor(fine_cnt.max() / np.maximum(fine_cnt, 1),
                          dtype=torch.float32, device=DEVICE)
    print(f"  gate: neg={n_neg}, pos={n_pos}, weight={[round(float(x),3) for x in gate_w]}", flush=True)
    print(f"  fine dist(Mild/Mod/Severe)={fine_cnt.astype(int).tolist()} "
          f"weight={[round(float(x),2) for x in fine_w]}", flush=True)

    # Train both heads + their attention vectors; encoder frozen
    params = (list(model.stage2_head.parameters()) + [model.stage2_attn_vector]
              + list(model.fine_head.parameters()) + [model.fine_attn_vector])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            med = batch["labels"].median(dim=-1).values.to(DEVICE)   # consensus
            coarse_t = (med >= BIN_THRESH).long()
            ignore = torch.full_like(coarse_t, -100)
            fine_t = torch.where(coarse_t == 1, (med - 1).to(torch.long), ignore)

            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["gate"], coarse_t, weight=gate_w)
            # fine CE = mean over positive samples. reduction="mean" would give
            # NaN (0/0) for all-normal batches, so use sum + divide by n_pos.
            fine_loss = F.cross_entropy(out["fine"], fine_t, weight=fine_w,
                                        ignore_index=-100, reduction="sum")
            loss = loss + fine_loss / int((coarse_t == 1).sum().clamp(min=1))
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        # Validation (4-class logits argmax → binary F1 + Acc4 from same preds)
        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]

        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if epoch % 5 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | "
                  f"val F1={val_f1:.4f} Acc4={val_metrics['acc4']:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


METHODS = {
    "V2": train_v2,
    "MFCCStats": train_mfccstats,
    "MFCCFusion": train_mfccfusion,
    "WhisperProbe-Mid": train_whisperprobemid,
    "CoarseToFine": train_coarsetofine,
    "WhisperFT": train_whisperft,
    "CrowdLayer": train_crowdlayer,
    "CrowdAttention": train_crowdattention,
    "COINNet": train_coinnet,
    "LFCx": train_lfcx,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS.keys()) + ["V2"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--accum_grad", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--conf_warmup", type=int, default=20)
    parser.add_argument("--lambda_conf", type=float, default=1.0)
    parser.add_argument("--lambda_ramp_epochs", type=int, default=0,
                        help=">0: linearly ramp lambda_conf 0->target over epochs 1..R "
                             "(instead of hard jump at conf_warmup+1)")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_bin", type=float, default=1.0,
                        help="V2 binary (>=1) focal weight; 0 disables it")
    parser.add_argument("--ablation", type=str, default=None,
                        help="V2 ablation: global_cm, w/o_multiannotator, w/o_focal, "
                             "w/o_ctc, w/o_agree, w/o_conf_ce")
    parser.add_argument("--old_split", action="store_true",
                        help="Use 0.7:0.15:0.15 old split instead of 5-fold")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--oracle", action="store_true",
                        help="Enable oracle tracking (eval test every epoch, save best-test checkpoint)")
    parser.add_argument("--llrd", action="store_true",
                        help="V2: layer-wise learning rate decay for encoder")
    parser.add_argument("--llrd_factor", type=float, default=0.95,
                        help="LLRD: per-layer LR decay factor (0.9-0.98)")
    parser.add_argument("--unfreeze_layers", type=int, default=-1,
                        help="V2/V2Frozen: number of last Conformer blocks to unfreeze (-1=all, 0=head-only)")
    parser.add_argument("--no_ctc", action="store_true",
                        help="V2: disable CTC auxiliary loss")
    parser.add_argument("--leak_test_ratio", type=float, default=0.0,
                        help="Ratio of test samples to leak into training (data leakage)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_last_epoch", action="store_true",
                        help="Evaluate last epoch instead of best-val checkpoint")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load existing best-val checkpoint and evaluate only")
    parser.add_argument("--out_results", type=str, default=None,
                        help="Output results JSON path (default: RESULTS_FILE)")
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"Method: {args.method} | Fold: {args.fold}", flush=True)
    print(f"Ablation: {args.ablation or 'None'}", flush=True)
    if args.old_split:
        print(f"Split: OLD (0.7:0.15:0.15)", flush=True)
    print(f"{'#'*60}", flush=True)

    # Load data

    if args.old_split:
        from data_loader import get_dataloaders
        loaders, df = get_dataloaders(batch_size=args.batch_size)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
    else:
        train_loader, val_loader, test_loader, df = load_fold_data(
            args.fold, args.batch_size, use_dict_features=False,
            leak_test_ratio=args.leak_test_ratio)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}", flush=True)

    # Stash test_loader on args so the training loop can run test at the best val epoch
    args.test_loader = test_loader

    # Set the random seed uniformly (all methods, for reproducibility)
    if getattr(args, 'seed', None) is not None:
        import random
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"  Random seed set to {seed}", flush=True)

    # Training
    train_fn = METHODS[args.method]
    if args.eval_only:
        # Load the existing best-val checkpoint only; skip training
        tag = f"{args.method}_fold{args.fold}"
        if getattr(args, 'ablation', None) and args.method == "V2":
            abla_tag = args.ablation.replace("/", "_")
            tag = f"{tag}_{abla_tag}"
        best_path = str(CHECKPOINTS / f"{tag}.pt")
        best_val_f1 = None
        if not os.path.isfile(best_path):
            print(f"[eval_only] WARNING: {best_path} not found, skipping", flush=True)
            best_path = None
        else:
            print(f"[eval_only] Loading {best_path}", flush=True)
    else:
        best_path, best_val_f1 = train_fn(train_loader, val_loader, args)

    # Switch to the last-epoch checkpoint when --eval_last_epoch is given
    if args.eval_last_epoch:
        last_tag = f"{args.method}_fold{args.fold}"
        if getattr(args, 'ablation', None) and args.method == "V2":
            abla_tag = args.ablation.replace("/", "_")
            last_tag = f"{last_tag}_{abla_tag}"
        last_path = CHECKPOINTS / f"{last_tag}_last.pt"
        if last_path.exists():
            best_path = str(last_path)
            print(f"  [Last Epoch] Using last-epoch checkpoint for test evaluation", flush=True)
        else:
            print(f"  [Last Epoch] WARNING: {last_path} not found, falling back to best-val", flush=True)

    # Test
    test_metrics = None  # default

    # Regular methods: load best checkpoint and evaluate
    if best_path is not None and os.path.isfile(best_path):
        if args.method == "V2":
            v2_variant = "v2g" if getattr(args, "ablation", None) == "global_cm" else "v2"
            model = SenseVoiceMultiAnnotator(variant=v2_variant, use_ctc=True).to(DEVICE)
        elif args.method == "COINNet":
            model = SenseVoiceMultiAnnotator(variant="coinnet", use_ctc=False).to(DEVICE)
            model.init_e_params(len(train_loader.dataset))
        elif args.method == "CrowdLayer":
            model = SenseVoiceMultiAnnotator(variant="crowdlayer", use_ctc=False).to(DEVICE)
        elif args.method == "LFCx":
            model = SenseVoiceMultiAnnotator(variant="lfcx", use_ctc=False).to(DEVICE)
        elif args.method == "CrowdAttention":
            model = SenseVoiceMultiAnnotator(variant="crowdattention", use_ctc=True).to(DEVICE)
        elif args.method == "WhisperProbe-Mid":
            from comparison_models import WhisperProbeMidModel
            model = WhisperProbeMidModel().to(DEVICE)
        elif args.method == "WhisperFT":
            from comparison_models import WhisperFTModel
            model = WhisperFTModel(n_freeze_layers=getattr(args, "n_freeze", 0)).to(DEVICE)
        elif args.method == "CoarseToFine":
            from comparison_models import CoarseToFineWhisper
            model = CoarseToFineWhisper().to(DEVICE)
        elif args.method == "MFCCStats":
            from comparison_models import MFCCStatsModel
            model = MFCCStatsModel().to(DEVICE)
        elif args.method == "MFCCFusion":
            from comparison_models import MFCCFusionModel
            emb_path = ROOT / "text_embeddings.npz"
            model = MFCCFusionModel(emb_path=str(emb_path) if emb_path.exists() else None).to(DEVICE)
        else:
            model = None

        if model is not None:
            model.load_state_dict(torch.load(best_path, map_location=DEVICE), strict=False)
            model.eval()
            test_metrics = evaluate_d2(model, test_loader)

            print(f"\n{'='*60}", flush=True)
            mode_str = "[LAST EPOCH] " if args.eval_last_epoch else ""
            print(f"{mode_str}Test Results: {args.method} Fold {args.fold}", flush=True)
            print(f"  Acc4={test_metrics['acc4']:.4f}  Acc_bin={test_metrics['acc_bin']:.4f}", flush=True)
            print(f"  Prec={test_metrics['prec']:.4f}  Sens={test_metrics['sens']:.4f}  Spec={test_metrics['spec']:.4f}", flush=True)
            print(f"  F1={test_metrics['f1']:.4f}", flush=True)

    # Save results
    result = {
        "method": args.method,
        "fold": args.fold,
        "ablation": args.ablation,
        "seed": getattr(args, 'seed', None),
        "val_f1": round(float(best_val_f1), 4) if best_val_f1 is not None else None,
        "eval_mode": "last_epoch" if args.eval_last_epoch else "best_val",
        "test": test_metrics,
        "config": {
            "lr": args.lr, "batch_size": args.batch_size,
            "n_epochs": args.n_epochs, "accum_grad": args.accum_grad,
            "lambda_conf": args.lambda_conf, "focal_gamma": args.focal_gamma,
            "lambda_ramp_epochs": getattr(args, 'lambda_ramp_epochs', 0),
        }
    }
    if args.leak_test_ratio > 0:
        result["leak_test_ratio"] = args.leak_test_ratio

    out_file = RESULTS_FILE
    if getattr(args, 'out_results', None):
        out_file = Path(args.out_results)

    # Serialize append across concurrent workers (multiple GPUs share this file)
    import fcntl
    lock_path = out_file.with_suffix(out_file.suffix + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.exists(out_file):
            with open(out_file) as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results.append(result)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        fcntl.flock(lf, fcntl.LOCK_UN)
    print(f"\nResults saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
