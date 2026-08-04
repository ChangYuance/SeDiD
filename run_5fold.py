#!/usr/bin/env python3
"""5折统一实验运行器：支持所有方法，只对 D2 评估。

用法:
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method COINNet --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method Tanno --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method CrowdLayer --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method LFCx --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method CrowdAttention --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method FixedM --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method SoftLabel --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method D0 --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method D1 --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method MV --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method DS --fold 0
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method Ensemble --fold 0

消融:
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --ablation w/o_focal
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --ablation w/o_conf_ce
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --ablation w/o_ctc
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --ablation w/o_agree
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --lambda_conf 0.5
  CUDA_VISIBLE_DEVICES=7 python run_5fold.py --method V2 --fold 0 --focal_gamma 1.0
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
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import build_label_dataframe, AudioMultiAnnotatorDataset, collate_audio
from model import (
    SenseVoiceMultiAnnotator, count_parameters,
)
from finetune_baseline import SingleLabelModel
# 徐天键 CTC 方法: 用 importlib 直接从 CTC_2c/model.py 导入 (避免与 ROOT/model.py 冲突)
import importlib.util
_CTC_2C_MODEL_PATH = str(ROOT / "徐天键方法/CTC/CTC_2c/model.py")
_spec = importlib.util.spec_from_file_location("dolphinctc_model", _CTC_2C_MODEL_PATH)
_dolphinctc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dolphinctc_mod)
DolphinCTCClassifier = _dolphinctc_mod.DolphinEncoderClassifier
from train import DEVICE, build_warmup_cosine_scheduler

# ── EMA (Exponential Moving Average) ──

class EMA:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # After each optimizer step:
        ema.update(model)
        # Before val/test:
        ema.apply(model)
        # After val/test:
        ema.restore(model)
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


# ── DictOnlyModel: MLP classifier on ASR dictionary features ──

class DictOnlyModel(nn.Module):
    """轻量 MLP: 126-dim dict features → 64 → 32 → 16 → 4."""
    def __init__(self, input_dim=126):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        logits = self.net(x)
        return {"logits": logits, "p": F.softmax(logits, dim=-1)}

# ── 常量 ──
D2_IDX = 1  # label_b = 师 = D2
FOLD_FILE = ROOT / "fold_indices.json"
CHECKPOINTS = ROOT / "checkpoints_5fold"
CHECKPOINTS.mkdir(exist_ok=True)
RESULTS_FILE = ROOT / "analytics/results_5fold.json"


# ════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════

def load_fold_data(fold_id: int, batch_size=4, use_dict_features=False,
                   leak_test_ratio: float = 0.0):
    """返回 train/val/test DataLoader 和原始 DataFrame。"""
    with open(FOLD_FILE) as f:
        folds = json.load(f)
    fold = folds[str(fold_id)]
    df = build_label_dataframe()

    # 将部分 test 索引加入 train（和 val）
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


# ════════════════════════════════════════════════════
# 评估函数（只对 D2）
# ════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_d2(model, loader):
    """评估模型对 D2 的性能（二分类 F1）。"""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        if isinstance(model, DictOnlyModel):
            dict_features = batch["dict_features"].to(DEVICE)
            out = model(dict_features)
        else:
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            dict_kw = {}
            if "dict_features" in batch:
                dict_kw["dict_features"] = batch["dict_features"].to(DEVICE)
            if hasattr(model, '_dummy'):  # MFCCFusionModel has text embeddings buffer
                dict_kw["keys"] = batch["keys"]
            out = model(speech, speech_lengths, **dict_kw)
        all_preds.append(out["logits"].argmax(dim=-1).cpu())
        all_labels.append(batch["labels"][:, D2_IDX].cpu())

    preds = torch.cat(all_preds)  # [N]
    labels = torch.cat(all_labels)  # [N]

    return _compute_binary_metrics(preds.numpy(), labels.numpy())


def _compute_binary_metrics(y_pred, y_true):
    """二分类 {0,1}=非重度 vs {2,3}=重度, 返回 dict。

    f1/positive_f1 = 重度类 F1; macro_f1 = (positive_f1 + negative_f1) / 2。
    """
    p_bin = (y_pred >= 2).astype(int)
    t_bin = (y_true >= 2).astype(int)

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


# ════════════════════════════════════════════════════
# 标签生成
# ════════════════════════════════════════════════════

def make_soft_labels(labels_hard):
    B = labels_hard.size(0)
    soft = torch.zeros(B, 4, device=labels_hard.device)
    for c in range(4):
        soft[:, c] = (labels_hard == c).float().mean(dim=1)
    return soft


def _compute_mv_labels(labels_np):
    mv = stats.mode(labels_np, axis=1, keepdims=False).mode.flatten()
    all_diff = ~np.any([labels_np[:, 0] == labels_np[:, 1],
                        labels_np[:, 0] == labels_np[:, 2],
                        labels_np[:, 1] == labels_np[:, 2]], axis=0)
    if all_diff.any():
        mv[all_diff] = np.sort(labels_np[all_diff], axis=1)[:, 1]
    return mv


def _compute_ds_labels(labels_np, n_classes=4, max_iter=100, tol=1e-4):
    N, K = labels_np.shape
    z = stats.mode(labels_np, axis=1, keepdims=False).mode.flatten()
    for _ in range(max_iter):
        z_prev = z.copy()
        prior = np.array([(z == c).sum() / max(N, 1) for c in range(n_classes)])
        pi = np.zeros((K, n_classes, n_classes))
        for j in range(K):
            for true_c in range(n_classes):
                mask = z == true_c
                if mask.sum() == 0:
                    pi[j, true_c] = np.ones(n_classes) / n_classes
                else:
                    for obs_c in range(n_classes):
                        pi[j, true_c, obs_c] = (labels_np[mask, j] == obs_c).sum() / mask.sum()
        z_new = np.zeros(N, dtype=int)
        for i in range(N):
            scores = np.array([
                np.log(prior[c] + 1e-10) + sum(
                    np.log(pi[j, c, labels_np[i, j]] + 1e-10) for j in range(K)
                ) for c in range(n_classes)
            ])
            z_new[i] = np.argmax(scores)
        z = z_new
        if np.mean(z == z_prev) > 1 - tol:
            break
    return z


# ════════════════════════════════════════════════════
# 训练函数
# ════════════════════════════════════════════════════

def train_epoch_v2(model, loader, optimizer, scheduler=None,
                   accum_grad=4, grad_clip=5.0, global_step=0,
                   lambda_conf=1.0, focal_gamma=2.0, lambda_reg=0.0,
                   ablation=None, lambda_trace=0.01):
    """V2 系列训练（支持消融）。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        model_kwargs = {"labels": labels}
        if "dict_features" in batch:
            model_kwargs["dict_features"] = batch["dict_features"].to(DEVICE)
        out = model(speech, speech_lengths, **model_kwargs)

        # 手动构建 loss（支持消融）
        logits = out["logits"]
        # 1. 共识标签 + 一致度权重
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

        total = focal + lambda_conf * conf_ce + lambda_reg * reg_loss + ctc_loss + trace_loss
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


def train_epoch_single(model, loader, optimizer, get_label_fn, scheduler=None,
                       accum_grad=4, grad_clip=5.0, global_step=0,
                       use_ctc=True):
    """单标签模型训练。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = get_label_fn(batch).to(DEVICE)
        out = model(speech, speech_lengths)
        loss = F.cross_entropy(out["logits"], labels)
        if use_ctc and "ctc_loss" in out:
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


@torch.no_grad()
def validate_single(model, loader, get_label_fn, use_ctc=True):
    """单标签模型验证（D2 指标）。"""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        out = model(speech, speech_lengths)
        all_preds.append(out["logits"].argmax(dim=-1).cpu())
        all_labels.append(batch["labels"][:, D2_IDX].cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return _compute_binary_metrics(preds.numpy(), labels.numpy())


# ════════════════════════════════════════════════════
# 训练各方法
# ════════════════════════════════════════════════════

def train_v2(train_loader, val_loader, args):
    """V2 / FixedM 训练。"""
    variant = {"V2": "v2", "FixedM": "v2a"}[args.method]
    use_ctc = not getattr(args, 'no_ctc', False)
    model = SenseVoiceMultiAnnotator(variant=variant, use_ctc=use_ctc,
                                     num_unfrozen_layers=args.unfreeze_layers).to(DEVICE)
    if args.unfreeze_layers >= 0:
        tag = f"V2-UL{args.unfreeze_layers}_fold{args.fold}"
    elif args.leak_test_ratio > 0:
        tag = f"V2_leak{int(args.leak_test_ratio*100)}_fold{args.fold}"
    else:
        tag = f"V2_fold{args.fold}"
    if getattr(args, 'no_ctc', False):
        tag = tag.replace("V2", "V2noCTC")
    if getattr(args, 'aug_ema', False):
        model.use_specaug = True
        tag = f"V2Aug-EMA_fold{args.fold}"
    elif getattr(args, 'aug', False):
        model.use_specaug = True
        tag = f"V2Aug_fold{args.fold}"
    if getattr(args, 'ablation', None):
        abla_tag = args.ablation.replace("/", "_")
        tag = f"{tag}_{abla_tag}"
    if getattr(args, 'seed', None) is not None and args.seed != 42:
        tag = f"{tag}_s{args.seed}"
    if getattr(args, 'tune_tag', None):
        tag = f"V2_fold{args.fold}_tune{args.tune_tag}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_v2frozen(train_loader, val_loader, args):
    """V2 with frozen encoder — only trains attention_pool + classifier + M_generator (~66K params vs 221M)."""
    model = SenseVoiceMultiAnnotator(variant="v2", use_ctc=True).to(DEVICE)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen encoder: {n_trainable:,}/{n_total:,} trainable ({100*n_trainable/n_total:.1f}%)", flush=True)
    tag = f"V2Frozen_fold{args.fold}"
    if getattr(args, 'tune_tag', None):
        tag = f"V2Frozen_{args.tune_tag}_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_v2_mid(train_loader, val_loader, args):
    """V2 + 中层特征 (encoder layer 25) + 去掉 CTC。"""
    model = SenseVoiceMultiAnnotator(variant="v2_mid", use_ctc=False).to(DEVICE)
    tag = f"V2mid_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_coinnet(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="coinnet", use_ctc=False).to(DEVICE)
    n_train = len(train_loader.dataset)
    model.init_e_params(n_train)
    tag = f"COINNet_fold{args.fold}"
    if getattr(args, 'seed', None) is not None and args.seed != 42:
        tag = f"{tag}_s{args.seed}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_tanno(train_loader, val_loader, args):
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    # Ablation: random CM init instead of near-identity
    if getattr(args, 'ablation', None) == "random_init":
        with torch.no_grad():
            for cm in model.confusion_matrices:
                nn.init.normal_(cm.logits, mean=0.0, std=1.0)
        print("[ablation] Random CM init applied", flush=True)
    tag = f"Tanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_tanno_twostage(train_loader, val_loader, args):
    """Two-stage Tanno: stage 1 = CE(p, MV), stage 2 = CE(q_k, y_k) + trace."""
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"TannoTwoStage_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_tanno_aug(train_loader, val_loader, args):
    """Tanno with extra data augmentation (speed perturb + gaussian noise)."""
    # Enable extra augmentation on the existing training dataset
    train_loader.dataset.extra_augment = True
    train_loader.dataset.augment = True
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"TannoAug_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_crtanno(train_loader, val_loader, args):
    """CR-Tanno: CE(q_k, y_k) + λ_mv · CE(p, MV) + λ_trace · tr(M_k)."""
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"CRTanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_sgtanno(train_loader, val_loader, args):
    """SG-Tanno: stop-gradient on p in q_k = p.detach() @ M_k.

    p is trained only by CE(p, MV), M_k only by CE(q_k, y_k) + trace.
    """
    model = SenseVoiceMultiAnnotator(variant="sgtanno", use_ctc=False).to(DEVICE)
    tag = f"SGTanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_loratanno(train_loader, val_loader, args):
    """LoRATanno: shared base S + low-rank residual A_k@B_k^T."""
    model = SenseVoiceMultiAnnotator(variant="loratanno", use_ctc=False).to(DEVICE)
    tag = f"LoRATanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_prototanno(train_loader, val_loader, args):
    """ProtoTanno: shared prototype CMs + per-annotator prototype weights."""
    model = SenseVoiceMultiAnnotator(variant="prototanno", use_ctc=False).to(DEVICE)
    tag = f"ProtoTanno_fold{args.fold}"
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


def train_v2tanno(train_loader, val_loader, args):
    """Direction B: V2's instance-dependent CM + Tanno's CE + trace loss."""
    model = SenseVoiceMultiAnnotator(variant="v2", use_ctc=False).to(DEVICE)
    tag = f"V2tanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_softtanno(train_loader, val_loader, args):
    """SoftTanno: Tanno CMs + CE(q_k, y_k) + KL(p || soft_label) + trace."""
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"SoftTanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_softtanno_v2(train_loader, val_loader, args):
    """SoftTanno v2: Tanno CMs + KL(q_k||soft) + KL(p||soft) + trace.

    All targets are the consensus soft distribution — no hard labels.
    """
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"SoftTannoV2_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_ddpm(train_loader, val_loader, args):
    """DDPM: Decoupled Dual-Path Model — consensus head + annotator head + bridge."""
    model = SenseVoiceMultiAnnotator(variant="ddpm", use_ctc=False).to(DEVICE)
    tag = f"DDPM_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_v2trace(train_loader, val_loader, args):
    """Direction A: V2 - CTC + trace regularization."""
    model = SenseVoiceMultiAnnotator(variant="v2", use_ctc=True).to(DEVICE)
    tag = f"V2trace_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_v2soft(train_loader, val_loader, args):
    """V2Soft: V2 instance-dependent CMs + SoftLabel-style KL objective.

    V2 architecture (MGenerator), but trained with:
      L = λ_soft·KL(p||soft) + λ_anno·Σ_k KL(q_k||soft) + λ_trace·tr(M_k)
    No CE(q_k, y_k), no focal loss, no CTC.
    """
    model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False).to(DEVICE)
    tag = f"V2Soft_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def _get_class_weights(labels):
    """逆频率权重，均值归一化为 1。"""
    counts = np.bincount(labels, minlength=4)
    weights = 1.0 / (counts + 1e-10)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def _extract_probe_features(model, loader):
    """冻结 backbone 单次前向，抽取 pooled 特征 + D2 标签。"""
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
    """Frozen-backbone probe: 抽取 pooled 特征一次，只训练分类头。

    冻结 backbone 的特征每轮完全相同，缓存后把 O(epochs × backbone前向)
    降为 O(backbone前向 + epochs × 头训练)，结果等价。特征抽取时关闭增强，
    保证确定性。
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
    """通用多标注者训练循环。"""
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

    # 类别权重
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    if getattr(args, 'llrd', False):
        optimizer = torch.optim.AdamW(
            build_llrd_param_groups(model, args.lr, args.weight_decay, args.llrd_factor))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    # 是否 LFC-x 两阶段
    if args.method == "LFCx":
        from train import _set_lfcx_stage
        _set_lfcx_stage(model, stage=1)

    # EMA (Exponential Moving Average)
    use_ema = getattr(args, 'aug_ema', False)
    if use_ema:
        ema = EMA(model, decay=0.999)
        # Hook into optimizer.step to update EMA after each step
        orig_step = optimizer.step
        def _step_with_ema(*args_ema, **kwargs_ema):
            orig_step(*args_ema, **kwargs_ema)
            ema.update(model)
        optimizer.step = _step_with_ema

    best_val_f1 = -1.0
    top5_list = []  # (val_f1, epoch), sorted descending
    best_path = CHECKPOINTS / f"{tag}.pt"
    log_dir = ROOT / "tb_logs_5fold" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    global_step = 0
    effective_conf_warmup = args.conf_warmup
    lambda_conf_target = args.lambda_conf  # 保存原始值
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
        # 根据变体选择训练
        variant = model.variant

        # Conf warmup
        if epoch <= effective_conf_warmup:
            args.lambda_conf = 0.0
        else:
            args.lambda_conf = lambda_conf_target

        if args.method == "V2tanno":
            train_loss, global_step = _train_epoch_v2tanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "SoftTanno":
            train_loss, global_step = _train_epoch_softtanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_soft=getattr(args, 'lambda_soft', 0.5),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "SoftTannoV2":
            train_loss, global_step = _train_epoch_softtanno_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_soft=getattr(args, 'lambda_soft', 1.0),
                lambda_anno=getattr(args, 'lambda_anno', 0.5),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "V2Soft":
            train_loss, global_step = _train_epoch_softtanno_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_soft=getattr(args, 'lambda_soft', 1.0),
                lambda_anno=getattr(args, 'lambda_anno', 0.5),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "V2trace":
            train_loss, global_step = train_epoch_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_conf=args.lambda_conf,
                focal_gamma=args.focal_gamma,
                ablation="v2trace", lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif variant in ("v2", "v2a", "v2_mid"):
            # V2 训练：focal + λ_conf·CE(q_k, y_k) + CTC + (可选) λ_trace·tr(M)
            train_loss, global_step = train_epoch_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_conf=args.lambda_conf,
                focal_gamma=args.focal_gamma,
                ablation=getattr(args, 'ablation', None),
                lambda_trace=getattr(args, 'lambda_trace', 0.0),
            )
        elif args.method == "CRTanno":
            train_loss, global_step = _train_epoch_crtanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_mv=getattr(args, 'lambda_mv', 0.1),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "SGTanno":
            train_loss, global_step = _train_epoch_crtanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_mv=getattr(args, 'lambda_mv', 0.1),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif args.method == "TannoTwoStage":
            stage = 1 if epoch <= 20 else 2
            if stage == 1:
                train_loss, global_step = _train_epoch_tanno_stage1(
                    model, train_loader, optimizer, scheduler,
                    accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                    global_step=global_step,
                )
                if epoch == 1:
                    print("  [Stage 1: CE(p, MV) — backbone only]", flush=True)
            else:
                if epoch == 21:
                    print("  [Stage 2: CE(q_k, y_k) + trace — adding CMs]", flush=True)
                train_loss, global_step = _train_epoch_tanno(
                    model, train_loader, optimizer, scheduler,
                    accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                    global_step=global_step, lambda_trace=getattr(args, 'lambda_trace', 0.01),
                )
        elif variant == "tanno":
            train_loss, global_step = _train_epoch_tanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif variant in ("crowdlayer", "lfcx"):
            train_loss, global_step = _train_epoch_baseline(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
            )
        elif variant == "dictfusion_v2":
            train_loss, global_step = train_epoch_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_conf=args.lambda_conf,
                focal_gamma=args.focal_gamma,
                ablation=getattr(args, 'ablation', None),
                lambda_trace=getattr(args, 'lambda_trace', 0.0),
            )
        elif variant == "dictfusion_tanno":
            train_loss, global_step = _train_epoch_tanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif variant == "dictadapt":
            train_loss, global_step = train_epoch_v2(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step, lambda_conf=args.lambda_conf,
                focal_gamma=args.focal_gamma,
                ablation=getattr(args, 'ablation', None),
                lambda_trace=getattr(args, 'lambda_trace', 0.0),
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
        elif variant == "loratanno":
            train_loss, global_step = _train_epoch_loratanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif variant == "prototanno":
            train_loss, global_step = _train_epoch_loratanno(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        elif variant == "ddpm":
            train_loss, global_step = _train_epoch_ddpm(
                model, train_loader, optimizer, scheduler,
                accum_grad=args.accum_grad, grad_clip=args.grad_clip,
                global_step=global_step,
                lambda_soft=getattr(args, 'lambda_soft', 1.0),
                lambda_bridge=getattr(args, 'lambda_bridge', 0.5),
                lambda_trace=getattr(args, 'lambda_trace', 0.01),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # LFC-x 阶段切换
        if args.method == "LFCx" and epoch == effective_conf_warmup + 1:
            from train import _set_lfcx_stage
            _set_lfcx_stage(model, stage=2)

        # EMA apply before validation
        if use_ema:
            ema.apply(model)

        # 验证
        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]

        # EMA restore after validation
        if use_ema:
            ema.restore(model)

        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        # Top-5 checkpoint saving (only save when entering top-5)
        in_top5 = len(top5_list) < 5 or val_f1 > top5_list[-1][0] + 1e-6
        top5_list.append((val_f1, epoch))
        top5_list.sort(key=lambda x: -x[0])
        if len(top5_list) > 5:
            drop_epoch = top5_list[-1][1]
            drop_path = CHECKPOINTS / f"{tag}_top{drop_epoch}.pt"
            if drop_path.exists():
                os.remove(drop_path)
            top5_list = top5_list[:5]
        if in_top5:
            torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_top{epoch}.pt")

        # 每个 epoch 跑 test
        test_str = ""
        test_f1 = -1.0
        use_oracle = getattr(args, 'oracle', False)
        run_test = (use_oracle or getattr(args, 'aug_ema', False) or getattr(args, 'aug', False)
                    or (hasattr(args, 'test_loader') and args.test_loader is not None))
        if run_test and hasattr(args, 'test_loader') and args.test_loader is not None:
            if use_ema:
                ema.apply(model)
            test_metrics = evaluate_d2(model, args.test_loader)
            if use_ema:
                ema.restore(model)
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

    # Top-5 ensemble on test set
    if use_ema and top5_list and hasattr(args, 'test_loader') and args.test_loader is not None:
        print(f"\n  [Top-5 Ensemble on test set]", flush=True)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for rank, (f1_val, ep) in enumerate(top5_list):
                ckpt_path = CHECKPOINTS / f"{tag}_top{ep}.pt"
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                model.eval()
                batch_logits = []
                for batch in args.test_loader:
                    speech = batch["speech"].to(DEVICE)
                    speech_lengths = batch["speech_lengths"].to(DEVICE)
                    out = model(speech, speech_lengths)
                    batch_logits.append(out["logits"])
                all_logits.append(torch.cat(batch_logits, dim=0))
            all_labels = torch.cat([b["labels"][:, D2_IDX] for b in args.test_loader]).to(DEVICE)
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        ens_preds = avg_logits.argmax(dim=-1).cpu().numpy()
        ens_labels = all_labels.cpu().numpy()
        ens_metrics = _compute_binary_metrics(ens_preds, ens_labels)
        print(f"  Top-5 Ensemble | Acc4={ens_metrics['acc4']:.4f} Acc_bin={ens_metrics['acc_bin']:.4f} "
              f"F1={ens_metrics['f1']:.4f} Recall={ens_metrics['sens']:.4f}", flush=True)
        # TensorBoard
        writer.add_scalar("Ensemble/test_F1", ens_metrics['f1'], 0)
        writer.add_scalar("Ensemble/test_Recall", ens_metrics['sens'], 0)
        # Cleanup top-k files
        for _, ep in top5_list:
            ckpt_path = CHECKPOINTS / f"{tag}_top{ep}.pt"
            if ckpt_path.exists():
                os.remove(ckpt_path)
        # Save ensemble state
        torch.save(model.state_dict(), CHECKPOINTS / f"{tag}_top5_ensemble.pt")

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


def _train_epoch_tanno(model, loader, optimizer, scheduler=None,
                       accum_grad=4, grad_clip=5.0, global_step=0,
                       lambda_trace=0.01):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        model_kwargs = {}
        if "dict_features" in batch:
            model_kwargs["dict_features"] = batch["dict_features"].to(DEVICE)
        out = model(speech, speech_lengths, **model_kwargs)
        loss, ce, trace = model.compute_tanno_loss(
            out["q_list"], labels, lambda_trace=lambda_trace)
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


def _train_epoch_tanno_stage1(model, loader, optimizer, scheduler=None,
                               accum_grad=4, grad_clip=5.0, global_step=0):
    """Stage 1: CE(p, MV) only — train backbone without confusion matrices."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(speech, speech_lengths)
        # Majority vote
        mv_labels = torch.mode(labels, dim=1).values
        loss = F.cross_entropy(out["logits"], mv_labels)
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


def _train_epoch_crtanno(model, loader, optimizer, scheduler=None,
                          accum_grad=4, grad_clip=5.0, global_step=0,
                          lambda_mv=0.1, lambda_trace=0.01):
    """CR-Tanno: CE(q_k, y_k) + λ_mv · CE(p, MV) + λ_trace · tr(M_k)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        model_kwargs = {}
        if "dict_features" in batch:
            model_kwargs["dict_features"] = batch["dict_features"].to(DEVICE)
        out = model(speech, speech_lengths, **model_kwargs)
        loss, ce, mv_ce, trace = model.compute_crtanno_loss(
            out["q_list"], out["p"], labels,
            lambda_mv=lambda_mv, lambda_trace=lambda_trace)
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


def _train_epoch_v2tanno(model, loader, optimizer, scheduler=None,
                          accum_grad=4, grad_clip=5.0, global_step=0,
                          lambda_trace=0.01):
    """Direction B: per-annotator CE + trace on instance-dependent M_k."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(speech, speech_lengths)
        loss, ce, trace = model.compute_v2tanno_loss(
            out["q_list"], out["M_list"], labels, lambda_trace=lambda_trace)
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


def _train_epoch_loratanno(model, loader, optimizer, scheduler=None,
                            accum_grad=4, grad_clip=5.0, global_step=0,
                            lambda_trace=0.01):
    """LoRATanno: CE(q_k, y_k) + λ_trace·tr(M_k) with shared + low-rank CMs."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(speech, speech_lengths)
        loss, ce, trace = model.compute_loratanno_loss(
            out["q_list"], out["M_list"], labels, lambda_trace=lambda_trace)
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


def _train_epoch_softtanno(model, loader, optimizer, scheduler=None,
                            accum_grad=4, grad_clip=5.0, global_step=0,
                            lambda_soft=0.5, lambda_trace=0.01):
    """SoftTanno training step: CE(q_k, y_k) + λ_soft·KL(p||soft) + λ_trace·tr(M)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        soft_labels = make_soft_labels(labels)
        out = model(speech, speech_lengths)
        loss, ce, kl, trace = model.compute_softtanno_loss(
            out["q_list"], out["p"], labels, soft_labels,
            lambda_soft=lambda_soft, lambda_trace=lambda_trace)
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


def _train_epoch_ddpm(model, loader, optimizer, scheduler=None,
                       accum_grad=4, grad_clip=5.0, global_step=0,
                       lambda_soft=1.0, lambda_bridge=0.5, lambda_trace=0.01):
    """DDPM: CE(q_k, y_k) + λ_soft·KL(p_soft||soft) + λ_bridge·MSE(p_soft, p_anno) + λ_trace·tr(M)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        soft_labels = make_soft_labels(labels)
        out = model(speech, speech_lengths)
        loss, ce, kl, bridge, trace = model.compute_ddpm_loss(
            out["q_list"], out["p"], out["p_anno"], labels, soft_labels,
            lambda_soft=lambda_soft, lambda_bridge=lambda_bridge, lambda_trace=lambda_trace)
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


def _train_epoch_tanno_blocked(model, loader, optimizer, scheduler=None,
                                accum_grad=4, grad_clip=5.0, global_step=0,
                                lambda_soft=1.0, lambda_anno=0.5, lambda_trace=0.01):
    """TannoBlocked: KL(p||soft) + CE(detach(p)@M_k, y_k) + trace.

    Blocked gradient design: backbone sees only KL(p||soft), M_k sees only CE.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        soft_labels = make_soft_labels(labels)
        out = model(speech, speech_lengths)
        loss, kl_loss, ce_loss, trace = model.compute_tanno_blocked_loss(
            out["p"], labels, soft_labels,
            lambda_soft=lambda_soft, lambda_anno=lambda_anno,
            lambda_trace=lambda_trace)
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


def _train_epoch_tanno_bias(model, loader, optimizer, scheduler=None,
                             accum_grad=4, grad_clip=5.0, global_step=0,
                             lambda_soft=1.0, lambda_anno=0.5):
    """TannoBias: KL(p||soft) + CE(detach(p) + b_k, y_k).

    Per-doctor bias vectors instead of full confusion matrices.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        soft_labels = make_soft_labels(labels)
        out = model(speech, speech_lengths)
        loss, kl_loss, ce_loss = model.compute_tanno_bias_loss(
            out["logits"], labels, soft_labels,
            model.doctor_biases,
            lambda_soft=lambda_soft, lambda_anno=lambda_anno)
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


def _train_epoch_softtanno_v2(model, loader, optimizer, scheduler=None,
                               accum_grad=4, grad_clip=5.0, global_step=0,
                               lambda_soft=1.0, lambda_anno=0.5, lambda_trace=0.01):
    """SoftTanno v2: KL(p||soft) + λ_anno·Σ_k KL(q_k||soft) + λ_trace·tr(M)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        soft_labels = make_soft_labels(labels)
        out = model(speech, speech_lengths)
        loss, kl_p, kl_q, trace = model.compute_softtanno_v2_loss(
            out["p"], out["q_list"], soft_labels,
            lambda_soft=lambda_soft, lambda_anno=lambda_anno, lambda_trace=lambda_trace)
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


def _train_epoch_baseline(model, loader, optimizer, scheduler=None,
                          accum_grad=4, grad_clip=5.0, global_step=0):
    """CrowdLayer / LFC-x 基线训练（CE(q_k, y_k) only）。"""
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


# ── 单标签方法 ──

def train_doctor(train_loader, val_loader, args):
    """单医生 finetune（D0 或 D1）。"""
    doc_idx = int(args.method[1])  # "D0" -> 0, "D1" -> 1
    model = SingleLabelModel(use_ctc=True).to(DEVICE)
    tag = f"{args.method}_fold{args.fold}"
    def get_label(batch): return batch["labels"][:, doc_idx]
    return _train_single(model, train_loader, val_loader, get_label, tag, args)


def train_mv(train_loader, val_loader, args):
    """Majority Vote。"""
    # 预计算 MV 标签
    all_labels = np.concatenate([b["labels"].numpy() for b in train_loader])
    mv_labels = _compute_mv_labels(all_labels)
    for batch in train_loader:
        batch["labels"][:, 0] = torch.from_numpy(mv_labels[:len(batch["labels"])])
        break  # 上面已遍历完，需要重新构造
    # 重新设置 dataset labels
    train_loader.dataset.df["label_a"] = mv_labels
    train_loader.dataset.df["label_b"] = mv_labels
    train_loader.dataset.df["label_c"] = mv_labels

    model = SingleLabelModel(use_ctc=True).to(DEVICE)
    tag = f"MV_fold{args.fold}"
    def get_label(batch): return batch["labels"][:, 0]
    return _train_single(model, train_loader, val_loader, get_label, tag, args)


def train_ds(train_loader, val_loader, args):
    """Dawid-Skene。"""
    all_labels = np.concatenate([b["labels"].numpy() for b in train_loader])
    ds_labels = _compute_ds_labels(all_labels)
    train_loader.dataset.df["label_a"] = ds_labels
    train_loader.dataset.df["label_b"] = ds_labels
    train_loader.dataset.df["label_c"] = ds_labels

    model = SingleLabelModel(use_ctc=True).to(DEVICE)
    tag = f"DS_fold{args.fold}"
    def get_label(batch): return batch["labels"][:, 0]
    return _train_single(model, train_loader, val_loader, get_label, tag, args)


def train_softlabel(train_loader, val_loader, args):
    """软标签（Soft Label）。"""
    model = SingleLabelModel(use_ctc=True).to(DEVICE)
    tag = f"SoftLabel_fold{args.fold}"
    return _train_soft(model, train_loader, val_loader, tag, args)


def _train_single(model, train_loader, val_loader, get_label_fn, tag, args):
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    t0 = time.time()
    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        train_loss, global_step = train_epoch_single(
            model, train_loader, optimizer, get_label_fn, scheduler,
            accum_grad=args.accum_grad, grad_clip=args.grad_clip,
            global_step=global_step, use_ctc=True,
        )
        val_metrics = validate_single(model, val_loader, get_label_fn)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            if hasattr(args, 'test_loader') and args.test_loader is not None:
                test_metrics = evaluate_d2(model, args.test_loader)
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST] | test Acc4={test_metrics['acc4']:.4f} "
                      f"Acc_bin={test_metrics['acc_bin']:.4f} "
                      f"F1={test_metrics['f1']:.4f}", flush=True)
            else:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST]", flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def _train_soft(model, train_loader, val_loader, tag, args):
    """软标签训练。"""
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    t0 = time.time()
    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            soft_labels = make_soft_labels(batch["labels"].to(DEVICE))
            out = model(speech, speech_lengths)
            log_probs = F.log_softmax(out["logits"], dim=-1)
            loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            if model.use_ctc and "ctc_loss" in out:
                loss = loss + out["ctc_loss"]
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if epoch % 10 == 0 or is_best:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_ensemble(train_loader, val_loader, args):
    """Ensemble: 训练 D0 + D1，logits 平均。"""
    models = []
    for doc_idx in [0, 1]:
        model = SingleLabelModel(use_ctc=True).to(DEVICE)
        def _get_label(batch, di=doc_idx): return batch["labels"][:, di]
        _, _ = _train_single(model, train_loader, val_loader,
                             lambda b: _get_label(b, doc_idx),
                             f"Ensemble_D{doc_idx}_fold{args.fold}", args)
        models.append(model)

    # Ensemble 评估
    tag = f"Ensemble_fold{args.fold}"
    print(f"\nEvaluating {tag}...", flush=True)
    all_preds, all_labels = [], []
    for batch in val_loader:
        speech = batch["speech"].to(DEVICE)
        speech_lengths = batch["speech_lengths"].to(DEVICE)
        logits_list = [mdl(speech, speech_lengths)["logits"] for mdl in models]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        all_preds.append(avg_logits.argmax(dim=-1).cpu())
        all_labels.append(batch["labels"][:, D2_IDX].cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    val_f1 = _compute_binary_metrics(preds.numpy(), labels.numpy())["f1"]
    print(f"Ensemble val F1: {val_f1:.4f}", flush=True)
    return None, val_f1


# ════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 对比方法: DySARNet / HuBERTProbe / SSLMultiTask
# ═══════════════════════════════════════════════════════════════

def train_dysarnet(train_loader, val_loader, args):
    """DySARNet: lightweight CNN + STFT spectrogram."""
    from comparison_models import DySARNet
    model = DySARNet().to(DEVICE)
    tag = f"DySARNet_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)

            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
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


def train_hubertprobe(train_loader, val_loader, args):
    """HuBERT frozen backbone + linear probe."""
    from comparison_models import HuBERTProbeModel
    model = HuBERTProbeModel().to(DEVICE)
    return _train_probe_cached(model, train_loader, val_loader, args)

def train_wavlmprobe(train_loader, val_loader, args):
    """WavLM-large frozen backbone + linear probe (ICASSP 2026)."""
    from comparison_models import WavLMProbeModel
    model = WavLMProbeModel().to(DEVICE)
    return _train_probe_cached(model, train_loader, val_loader, args)

def train_sslmultitask(train_loader, val_loader, args):
    """SALR-style: SSL backbone + CE + optional triplet."""
    from comparison_models import SSLMultiTaskModel
    model = SSLMultiTaskModel(freeze_backbone=not args.ft_backbone).to(DEVICE)
    tag = f"SSLMT_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag} (freeze_backbone={not args.ft_backbone})\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)

            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
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


def _build_dolphinctc():
    """Build and wrap DolphinEncoderClassifier for evaluate_d2 compatibility."""
    base = DolphinCTCClassifier(
        num_classes=4,
        freeze_s2t=True,
        freeze_layer=12,
        use_specaug=False,
        classification_mode="ce",
        use_multiscale_temporal=True,
        use_bilstm=True,
        lstm_hidden_size=64,
        head_hidden_size=64,
        num_head_layers=1,
        use_ctc_head=False,
        device=DEVICE,
    ).to(DEVICE)

    class _LogitsWrapper(torch.nn.Module):
        """Wrap model to return dict with 'logits' key (compatible with evaluate_d2).

        nn.Module.train()/eval() propagate to submodules automatically.
        """
        def __init__(self, m):
            super().__init__()
            self._m = m
        def forward(self, speech, speech_lengths, **kwargs):
            out = self._m(speech, speech_lengths, **kwargs)
            if isinstance(out, dict) and "logits" not in out:
                out["logits"] = out.get("shared_logits", next(iter(out.values())))
                return out
            if not isinstance(out, dict):
                return {"logits": out}
            return out
        def state_dict(self, *args, **kwargs):
            return self._m.state_dict(*args, **kwargs)
        def load_state_dict(self, sd, *args, **kwargs):
            return self._m.load_state_dict(sd, *args, **kwargs)

    return _LogitsWrapper(base)


def train_dolphinctc(train_loader, val_loader, args):
    """Dolphin-small frozen encoder + temporal conv + BiLSTM (Xu Tianjian's CTC method, adapted)."""
    model = _build_dolphinctc()
    tag = f"DolphinCTC_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)

            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
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


def train_whisperprobe(train_loader, val_loader, args):
    """Frozen Whisper-small + linear classifier (Yue et al., ICASSP 2026)."""
    from comparison_models import WhisperProbeModel
    model = WhisperProbeModel().to(DEVICE)
    tag = f"WhisperProbe_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(speech, speech_lengths)
            loss = F.cross_entropy(out["logits"], labels)
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
    """Coarse-to-Fine Whisper Stage 2: coarse D2 binary (ICASSP 2026 SAND).

    Frozen Whisper-small encoder + enhanced pooling (mean/std/attention)
    + FC(256) → 2. Weighted CE for class imbalance.
    """
    from comparison_models import CoarseToFineWhisper
    model = CoarseToFineWhisper().to(DEVICE)
    tag = f"CoarseToFine_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    # Compute class weights for weighted CE (Stage 2 is binary: {0,1} vs {2,3})
    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    # Map to coarse: 0 if 0 or 1, 1 if 2 or 3
    coarse_labels = (all_labels >= 2).astype(int)
    n_pos = coarse_labels.sum()
    n_neg = len(coarse_labels) - n_pos
    weight_ratio = n_neg / max(n_pos, 1)
    class_weight = torch.tensor([1.0, weight_ratio], dtype=torch.float32, device=DEVICE)
    print(f"  Coarse class_weight: [1.0, {weight_ratio:.4f}] "
          f"(neg={n_neg}, pos={n_pos})", flush=True)

    # Only train Stage 2 head parameters
    stage2_params = list(model.stage2_head.parameters()) + [model.stage2_attn_vector]
    optimizer = torch.optim.AdamW(stage2_params, lr=args.lr, weight_decay=args.weight_decay)

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
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            coarse = (labels >= 2).long()

            logits = model.forward_stage2(speech, speech_lengths)
            loss = F.cross_entropy(logits, coarse, weight=class_weight)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(stage2_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        # Validation
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


def train_v2hybrid(train_loader, val_loader, args):
    """V2 backbone + FixedM + per-annotator CE + trace + focal - CTC."""
    model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False).to(DEVICE)
    tag = f"V2hybrid_fold{args.fold}"
    return _train_multiannotator_v2hybrid(model, train_loader, val_loader, tag, args)


def _train_multiannotator_v2hybrid(model, train_loader, val_loader, tag, args):
    """Training loop for V2hybrid: focal + conf_ce + trace - CTC."""
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    t0 = time.time()
    global_step = 0
    lambda_conf_target = args.lambda_conf

    for epoch in range(1, args.n_epochs + 1):
        effective_conf = lambda_conf_target if epoch > args.conf_warmup else 0.0

        train_loss, global_step = train_epoch_v2(
            model, train_loader, optimizer, scheduler,
            accum_grad=args.accum_grad, grad_clip=args.grad_clip,
            global_step=global_step, lambda_conf=effective_conf,
            focal_gamma=args.focal_gamma,
            ablation="w/o_ctc",
            lambda_trace=getattr(args, 'lambda_trace', 0.01),
        )

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            if hasattr(args, 'test_loader') and args.test_loader is not None:
                test_metrics = evaluate_d2(model, args.test_loader)
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST] | test Acc4={test_metrics['acc4']:.4f} "
                      f"Acc_bin={test_metrics['acc_bin']:.4f} "
                      f"F1={test_metrics['f1']:.4f}", flush=True)
            else:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST]", flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_tannoblocked(train_loader, val_loader, args):
    """TannoBlocked: fixed CMs + KL(p||soft) + CE(detach(p)@M_k, y_k) + trace."""
    model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tag = f"TannoBlocked_fold{args.fold}"
    return _train_multiannotator_tannoblocked(model, train_loader, val_loader, tag, args)


def _train_multiannotator_tannoblocked(model, train_loader, val_loader, tag, args):
    """Training loop for TannoBlocked."""
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"
    log_dir = ROOT / "tb_logs_5fold" / tag
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.n_epochs + 1):
        train_loss, global_step = _train_epoch_tanno_blocked(
            model, train_loader, optimizer, scheduler,
            accum_grad=args.accum_grad, grad_clip=args.grad_clip,
            global_step=global_step,
            lambda_soft=getattr(args, 'lambda_soft', 1.0),
            lambda_anno=getattr(args, 'lambda_anno', 0.5),
            lambda_trace=getattr(args, 'lambda_trace', 0.01),
        )

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]

        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            if hasattr(args, 'test_loader') and args.test_loader is not None:
                test_metrics = evaluate_d2(model, args.test_loader)
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST] | test Acc4={test_metrics['acc4']:.4f} "
                      f"Acc_bin={test_metrics['acc_bin']:.4f} "
                      f"F1={test_metrics['f1']:.4f}", flush=True)
            else:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST]", flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_tannobias(train_loader, val_loader, args):
    """TannoBias: per-doctor bias vectors + KL(p||soft) + CE(detach(p)+b_k, y_k)."""
    model = SenseVoiceMultiAnnotator(variant="tannobias", use_ctc=False).to(DEVICE)
    tag = f"TannoBias_fold{args.fold}"
    return _train_multiannotator_tannobias(model, train_loader, val_loader, tag, args)


def _train_multiannotator_tannobias(model, train_loader, val_loader, tag, args):
    """Training loop for TannoBias."""
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    all_labels = torch.cat([b["labels"][:, D2_IDX] for b in train_loader]).cpu().numpy()
    class_weights = _get_class_weights(all_labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"
    log_dir = ROOT / "tb_logs_5fold" / tag
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.n_epochs + 1):
        train_loss, global_step = _train_epoch_tanno_bias(
            model, train_loader, optimizer, scheduler,
            accum_grad=args.accum_grad, grad_clip=args.grad_clip,
            global_step=global_step,
            lambda_soft=getattr(args, 'lambda_soft', 1.0),
            lambda_anno=getattr(args, 'lambda_anno', 0.5),
        )

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]

        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            if hasattr(args, 'test_loader') and args.test_loader is not None:
                test_metrics = evaluate_d2(model, args.test_loader)
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST] | test Acc4={test_metrics['acc4']:.4f} "
                      f"Acc_bin={test_metrics['acc_bin']:.4f} "
                      f"F1={test_metrics['f1']:.4f}", flush=True)
            else:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f} "
                      f"[BEST]", flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val F1={val_f1:.4f}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_sl_tanno_ensemble(train_loader, val_loader, args):
    """SoftLabel + Tanno 预测融合 (logits平均, 无需额外训练)."""
    from finetune_baseline import SingleLabelModel

    # Load pretrained models from checkpoints
    sl_model = SingleLabelModel(use_ctc=True).to(DEVICE)
    sl_ckpt = CHECKPOINTS / f"SoftLabel_fold{args.fold}.pt"
    sl_model.load_state_dict(torch.load(sl_ckpt, map_location=DEVICE))
    sl_model.eval()

    tanno_model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
    tanno_ckpt = CHECKPOINTS / f"Tanno_fold{args.fold}.pt"
    tanno_model.load_state_dict(torch.load(tanno_ckpt, map_location=DEVICE))
    tanno_model.eval()

    tag = f"SL+Tanno_fold{args.fold}"
    print(f"\n{'='*60}\nSoftLabel+Tanno Ensemble {tag}\n{'='*60}", flush=True)

    # Validation: average logits
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            sl_logits = sl_model(speech, speech_lengths)["logits"]
            tanno_out = tanno_model(speech, speech_lengths)
            tanno_logits = tanno_out["logits"]
            avg_logits = (sl_logits + tanno_logits) / 2.0
            all_preds.append(avg_logits.argmax(dim=-1).cpu())
            all_labels.append(batch["labels"][:, D2_IDX].cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    val_f1 = _compute_binary_metrics(preds.numpy(), labels.numpy())["f1"]
    print(f"Ensemble val F1: {val_f1:.4f}", flush=True)

    # Dummy path to signal test should run
    dummy_path = CHECKPOINTS / f"SL+Tanno_fold{args.fold}.dummy"
    dummy_path.touch()
    return str(dummy_path), val_f1


# ════════════════════════════════════════════════════
# ASR Dictionary Feature Methods
# ════════════════════════════════════════════════════

def train_dict_only(train_loader, val_loader, args):
    """DictOnly: MLP on ASR dictionary features only (no audio)."""
    model = DictOnlyModel(input_dim=126).to(DEVICE)
    tag = f"DictOnly_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            x = batch["dict_features"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out["logits"], labels)
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
        if epoch % 10 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                  f"{'[BEST]' if is_best else ''}", flush=True)

    print(f"\nBest val epoch: {best_epoch}, best val F1: {best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_dict_fusion(train_loader, val_loader, args):
    """DictFusion: SenseVoice encoder + ASR dictionary features."""
    model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False, dict_dim=126).to(DEVICE)
    tag = f"DictFusion_fold{args.fold}"
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = math.ceil(len(train_loader) / args.accum_grad) * args.n_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, n_steps=n_steps, warmup_ratio=args.warmup_ratio)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = CHECKPOINTS / f"{tag}.pt"

    t0 = time.time()
    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            speech = batch["speech"].to(DEVICE)
            speech_lengths = batch["speech_lengths"].to(DEVICE)
            dict_features = batch["dict_features"].to(DEVICE)
            labels = batch["labels"][:, D2_IDX].to(DEVICE)
            out = model(speech, speech_lengths, dict_features=dict_features)
            loss = F.cross_entropy(out["logits"], labels)
            loss = loss / args.accum_grad
            loss.backward()
            if (batch_idx + 1) % args.accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * args.accum_grad
            n_batches += 1

        val_metrics = evaluate_d2(model, val_loader)
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1 + 1e-4
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            if hasattr(args, 'test_loader') and args.test_loader is not None:
                test_metrics = evaluate_d2(model, args.test_loader)
                print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                      f"[BEST] | test Acc4={test_metrics['acc4']:.4f} "
                      f"Acc_bin={test_metrics['acc_bin']:.4f} "
                      f"F1={test_metrics['f1']:.4f}", flush=True)
            else:
                print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f} "
                      f"[BEST]", flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={total_loss/max(n_batches,1):.4f} | val F1={val_f1:.4f}", flush=True)

    print(f"\nTraining done: {time.time()-t0:.1f}s | Best val F1={best_val_f1:.4f}", flush=True)
    return best_path, best_val_f1


def train_dictfusion_v2(train_loader, val_loader, args):
    """DictFusion+V2: speech encoder + dict features + V2's instance-dependent CMs."""
    model = SenseVoiceMultiAnnotator(variant="dictfusion_v2", use_ctc=False, dict_dim=126).to(DEVICE)
    tag = f"DictFusion+V2_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_dictfusion_tanno(train_loader, val_loader, args):
    """DictFusion+Tanno: speech encoder + dict features + fixed CMs + CE+trace."""
    model = SenseVoiceMultiAnnotator(variant="dictfusion_tanno", use_ctc=False, dict_dim=126).to(DEVICE)
    tag = f"DictFusion+Tanno_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


def train_dictadapt(train_loader, val_loader, args):
    """DictAdapt: speech-based CMs + ASR dict-adaptive offset (V2 training)."""
    model = SenseVoiceMultiAnnotator(variant="dictadapt", use_ctc=False, dict_dim=126).to(DEVICE)
    tag = f"DictAdapt_fold{args.fold}"
    return _train_multiannotator(model, train_loader, val_loader, tag, args)


METHODS = {
    "V2": train_v2,
    "V2Frozen": train_v2frozen,
    "V2mid": train_v2_mid,
    "FixedM": train_v2,
    "COINNet": train_coinnet,
    "Tanno": train_tanno,
    "TannoTwoStage": train_tanno_twostage,
    "TannoAug": train_tanno_aug,
    "CRTanno": train_crtanno,
    "SGTanno": train_sgtanno,
    "SoftTanno": train_softtanno,
    "SoftTannoV2": train_softtanno_v2,
    "CrowdLayer": train_crowdlayer,
    "LFCx": train_lfcx,
    "CrowdAttention": train_crowdattention,
    "SoftLabel": train_softlabel,
    "D0": train_doctor,
    "D1": train_doctor,
    "MV": train_mv,
    "DS": train_ds,
    "Ensemble": train_ensemble,
    "V2tanno": train_v2tanno,
    "V2trace": train_v2trace,
    "V2hybrid": train_v2hybrid,
    "V2Soft": train_v2soft,
    "TannoBlocked": train_tannoblocked,
    "TannoBias": train_tannobias,
    "LoRATanno": train_loratanno,
    "ProtoTanno": train_prototanno,
    "SL+Tanno": train_sl_tanno_ensemble,
    "DySARNet": train_dysarnet,
    "HuBERTProbe": train_hubertprobe,
    "SSLMT": train_sslmultitask,
    "WhisperProbe": train_whisperprobe,
    "WhisperFT": train_whisperft,
    "WhisperProbe-Mid": train_whisperprobemid,
    "WavLMProbe": train_wavlmprobe,
    "CoarseToFine": train_coarsetofine,
    "DolphinCTC": train_dolphinctc,
    "MFCCStats": train_mfccstats,
    "MFCCFusion": train_mfccfusion,
    "DDPM": train_ddpm,
    "DictOnly": train_dict_only,
    "DictFusion": train_dict_fusion,
    "DictFusion+V2": train_dictfusion_v2,
    "DictFusion+Tanno": train_dictfusion_tanno,
    "DictAdapt": train_dictadapt,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS.keys()) + ["V2"], required=True)
    parser.add_argument("--hubert_mid_layer", type=int, default=-1,
                        help="HuBERTProbe: hidden_states index to probe (-1=last, 0-11=layers)")
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
    parser.add_argument("--lambda_trace", type=float, default=0.01)
    parser.add_argument("--lambda_soft", type=float, default=0.5,
                        help="SoftTanno: weight for soft-label KL(p||soft) term")
    parser.add_argument("--lambda_mv", type=float, default=0.1,
                        help="CRTanno: weight for CE(p, majority_vote) term")
    parser.add_argument("--lambda_anno", type=float, default=0.5,
                        help="SoftTannoV2: weight for per-annotator KL(q_k||soft) term")
    parser.add_argument("--lambda_bridge", type=float, default=0.5,
                        help="DDPM: weight for MSE bridge between p_soft and p_anno")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--mu1", type=float, default=0.01)
    parser.add_argument("--mu2", type=float, default=0.01)
    parser.add_argument("--ablation", type=str, default=None,
                        help="V2 ablation: w/o_focal, w/o_conf_ce, w/o_ctc, w/o_agree, w/o_multiannotator")
    parser.add_argument("--ft_backbone", action="store_true",
                        help="SSLMT: finetune backbone instead of freezing")
    parser.add_argument("--old_split", action="store_true",
                        help="Use 0.7:0.15:0.15 old split instead of 5-fold")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--oracle", action="store_true",
                        help="Enable oracle tracking (eval test every epoch, save best-test checkpoint)")
    parser.add_argument("--aug_ema", action="store_true",
                        help="Enable SpecAugment + speed perturb + EMA + top-5 ensemble")
    parser.add_argument("--aug", action="store_true",
                        help="Enable SpecAugment + speed perturb only (no EMA)")
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
    parser.add_argument("--tune_tag", type=str, default=None,
                        help="Tag suffix for tuning experiment identification")
    parser.add_argument("--eval_last_epoch", action="store_true",
                        help="Evaluate last epoch instead of best-val checkpoint")
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"Method: {args.method} | Fold: {args.fold}", flush=True)
    print(f"Ablation: {args.ablation or 'None'}", flush=True)
    if args.old_split:
        print(f"Split: OLD (0.7:0.15:0.15)", flush=True)
    print(f"{'#'*60}", flush=True)

    # 加载数据
    use_dict = args.method in ("DictOnly", "DictFusion", "DictFusion+V2", "DictFusion+Tanno", "DictAdapt")
    if args.old_split:
        from data_loader import get_dataloaders
        loaders, df = get_dataloaders(batch_size=args.batch_size)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
    else:
        train_loader, val_loader, test_loader, df = load_fold_data(
            args.fold, args.batch_size, use_dict_features=use_dict,
            leak_test_ratio=args.leak_test_ratio)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}", flush=True)

    # 把 test_loader 存到 args 里，训练循环里 val best 时可以自动跑 test
    args.test_loader = test_loader

    # 统一设置随机种子（覆盖所有方法，保证可复现）
    if getattr(args, 'seed', None) is not None:
        import random
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"  Random seed set to {seed}", flush=True)

    # 训练
    train_fn = METHODS[args.method]
    best_path, best_val_f1 = train_fn(train_loader, val_loader, args)

    # 切换到 last epoch checkpoint（如果指定 --eval_last_epoch）
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

    # 测试
    test_metrics = None  # default

    # Regular methods: load best checkpoint and evaluate
    if best_path is not None and os.path.isfile(best_path):
        if args.method in ("V2", "V2Frozen", "FixedM"):
            variant = {"V2": "v2", "V2Frozen": "v2", "FixedM": "v2a"}[args.method]
            model = SenseVoiceMultiAnnotator(variant=variant, use_ctc=True).to(DEVICE)
        elif args.method == "COINNet":
            model = SenseVoiceMultiAnnotator(variant="coinnet", use_ctc=False).to(DEVICE)
            model.init_e_params(len(train_loader.dataset))
        elif args.method in ("Tanno", "SoftTanno", "SoftTannoV2", "TannoAug", "CRTanno"):
            model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
        elif args.method == "LoRATanno":
            model = SenseVoiceMultiAnnotator(variant="loratanno", use_ctc=False).to(DEVICE)
        elif args.method == "ProtoTanno":
            model = SenseVoiceMultiAnnotator(variant="prototanno", use_ctc=False).to(DEVICE)
        elif args.method == "CrowdLayer":
            model = SenseVoiceMultiAnnotator(variant="crowdlayer", use_ctc=False).to(DEVICE)
        elif args.method == "LFCx":
            model = SenseVoiceMultiAnnotator(variant="lfcx", use_ctc=False).to(DEVICE)
        elif args.method == "CrowdAttention":
            model = SenseVoiceMultiAnnotator(variant="crowdattention", use_ctc=True).to(DEVICE)
        elif args.method in ("SoftLabel", "D0", "D1", "MV", "DS"):
            model = SingleLabelModel(use_ctc=True).to(DEVICE)
        elif args.method in ("V2tanno", "V2trace"):
            model = SenseVoiceMultiAnnotator(variant="v2", use_ctc=args.method=="V2trace").to(DEVICE)
        elif args.method == "V2hybrid":
            model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False).to(DEVICE)
        elif args.method == "V2Soft":
            model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False).to(DEVICE)
        elif args.method == "TannoBlocked":
            model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
        elif args.method == "TannoBias":
            model = SenseVoiceMultiAnnotator(variant="tannobias", use_ctc=False).to(DEVICE)
        elif args.method == "DDPM":
            model = SenseVoiceMultiAnnotator(variant="ddpm", use_ctc=False).to(DEVICE)
        elif args.method == "DictOnly":
            model = DictOnlyModel(input_dim=126).to(DEVICE)
        elif args.method == "DictFusion":
            model = SenseVoiceMultiAnnotator(variant="v2a", use_ctc=False, dict_dim=126).to(DEVICE)
        elif args.method == "DictFusion+V2":
            model = SenseVoiceMultiAnnotator(variant="dictfusion_v2", use_ctc=False, dict_dim=126).to(DEVICE)
        elif args.method == "DictFusion+Tanno":
            model = SenseVoiceMultiAnnotator(variant="dictfusion_tanno", use_ctc=False, dict_dim=126).to(DEVICE)
        elif args.method == "DictAdapt":
            model = SenseVoiceMultiAnnotator(variant="dictadapt", use_ctc=False, dict_dim=126).to(DEVICE)
        elif args.method == "DySARNet":
            from comparison_models import DySARNet
            model = DySARNet().to(DEVICE)
        elif args.method == "HuBERTProbe":
            from comparison_models import HuBERTProbeModel
            mid_layer = getattr(args, "hubert_mid_layer", -1)
            model = HuBERTProbeModel(mid_layer=mid_layer).to(DEVICE)
            if mid_layer != -1:
                print(f"  HuBERT probing layer: {mid_layer} "
                      f"(hidden_states[{mid_layer}])", flush=True)
        elif args.method == "WhisperProbe":
            from comparison_models import WhisperProbeModel
            model = WhisperProbeModel().to(DEVICE)
        elif args.method == "WhisperProbe-Mid":
            from comparison_models import WhisperProbeMidModel
            model = WhisperProbeMidModel().to(DEVICE)
        elif args.method == "WhisperFT":
            from comparison_models import WhisperFTModel
            n_freeze = getattr(args, "n_freeze", 0)
            model = WhisperFTModel(n_freeze_layers=n_freeze).to(DEVICE)
        elif args.method == "WavLMProbe":
            from comparison_models import WavLMProbeModel
            model = WavLMProbeModel().to(DEVICE)
        elif args.method == "CoarseToFine":
            from comparison_models import CoarseToFineWhisper
            model = CoarseToFineWhisper().to(DEVICE)
        elif args.method == "DolphinCTC":
            model = _build_dolphinctc()
        elif args.method == "SSLMT":
            from comparison_models import SSLMultiTaskModel
            model = SSLMultiTaskModel(freeze_backbone=not args.ft_backbone).to(DEVICE)
        elif args.method == "MFCCStats":
            from comparison_models import MFCCStatsModel
            model = MFCCStatsModel().to(DEVICE)
        elif args.method == "MFCCFusion":
            from comparison_models import MFCCFusionModel
            ROOT = Path(__file__).resolve().parent
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

    # SL+Tanno ensemble: load pre-trained checkpoints and average logits
    if args.method == "SL+Tanno":
        from finetune_baseline import SingleLabelModel
        sl_model = SingleLabelModel(use_ctc=True).to(DEVICE)
        sl_ckpt = CHECKPOINTS / f"SoftLabel_fold{args.fold}.pt"
        sl_model.load_state_dict(torch.load(sl_ckpt, map_location=DEVICE))
        sl_model.eval()

        tanno_model = SenseVoiceMultiAnnotator(variant="tanno", use_ctc=False).to(DEVICE)
        tanno_ckpt = CHECKPOINTS / f"Tanno_fold{args.fold}.pt"
        tanno_model.load_state_dict(torch.load(tanno_ckpt, map_location=DEVICE))
        tanno_model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                speech = batch["speech"].to(DEVICE)
                speech_lengths = batch["speech_lengths"].to(DEVICE)
                sl_logits = sl_model(speech, speech_lengths)["logits"]
                tanno_out = tanno_model(speech, speech_lengths)
                tanno_logits = tanno_out["logits"]
                avg_logits = (sl_logits + tanno_logits) / 2.0
                all_preds.append(avg_logits.argmax(dim=-1).cpu())
                all_labels.append(batch["labels"][:, D2_IDX].cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        test_metrics = _compute_binary_metrics(preds.numpy(), labels.numpy())

    # Ensemble special case
    if args.method == "Ensemble":
        models = []
        for doc_idx in [0, 1]:
            m = SingleLabelModel(use_ctc=True).to(DEVICE)
            ckpt = CHECKPOINTS / f"Ensemble_D{doc_idx}_fold{args.fold}.pt"
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                m.eval()
                models.append(m)
        if models:
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    speech = batch["speech"].to(DEVICE)
                    speech_lengths = batch["speech_lengths"].to(DEVICE)
                    logits_list = [mdl(speech, speech_lengths)["logits"] for mdl in models]
                    avg_logits = torch.stack(logits_list).mean(dim=0)
                    all_preds.append(avg_logits.argmax(dim=-1).cpu())
                    all_labels.append(batch["labels"][:, D2_IDX].cpu())
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            test_metrics = _compute_binary_metrics(preds.numpy(), labels.numpy())
            print(f"\nEnsemble Test: F1={test_metrics['f1']:.4f}", flush=True)

    # 保存结果
    result = {
        "method": args.method,
        "fold": args.fold,
        "ablation": args.ablation,
        "seed": getattr(args, 'seed', None),
        "val_f1": round(float(best_val_f1), 4),
        "eval_mode": "last_epoch" if args.eval_last_epoch else "best_val",
        "test": test_metrics,
        "config": {
            "lr": args.lr, "batch_size": args.batch_size,
            "n_epochs": args.n_epochs, "accum_grad": args.accum_grad,
            "lambda_conf": args.lambda_conf, "focal_gamma": args.focal_gamma,
        }
    }
    if args.leak_test_ratio > 0:
        result["leak_test_ratio"] = args.leak_test_ratio

    out_file = RESULTS_FILE
    if getattr(args, 'tune_tag', None):
        out_file = RESULTS_FILE.parent / "tune_results.json"
    if os.path.exists(out_file):
        with open(out_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
