"""One training run: fine-tune a pretrained classifier on the frozen split, with exact resume.

Protocol (identical for every model; see configs/base.yaml)
  AdamW (no decay on bias/norm), per-step linear warm-up then cosine to min_lr, mixed precision, label
  smoothing, class-weighted cross-entropy (weights from TRAIN counts), gradient clipping, geometric-only
  augmentation, max 100 epochs, early stopping (patience 10) on VALIDATION macro-F1.
  Checkpoint choice uses validation only. The test set is evaluated once per run, after training, from the
  best-validation weights (enforced by CheckpointManager.mark_test_evaluated). eval.test=false skips it
  completely (used by the learning-rate check).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .checkpoint import CheckpointManager, build_state, restore_state
from .config import config_hash, get_by_path, make_run_id, run_dir_for
from .data import LeafDataset, class_weights, image_root, load_split, make_loader
from .metrics import compute_metrics
from .runlog import make_record
from .utils import count_params, human_time, set_seed


# ----------------------------------------------------------------------------- model / optimiser / loss
class TinyCNN(nn.Module):
    """Debug-only model (CPU smoke tests). Never used for results."""
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(32, k)

    def forward(self, x):
        return self.head(self.f(x))


def build_model(cfg: dict):
    name, k = cfg["model"]["name"], cfg["data"]["num_classes"]
    if name == "tiny_cnn":
        return TinyCNN(k), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    import timm
    tag = cfg["model"].get("tag")
    full = f"{name}.{tag}" if tag else name
    model = timm.create_model(full, pretrained=bool(cfg["model"].get("pretrained", True)), num_classes=k)
    dc = timm.data.resolve_data_config({}, model=model)
    return model, tuple(dc["mean"]), tuple(dc["std"])


def build_optimizer(model, cfg: dict):
    lr, wd = float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"])
    try:
        from timm.optim import create_optimizer_v2
        return create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=wd)
    except ImportError:                                   # no timm (smoke tests): same idea by hand
        dec = [p for n, p in model.named_parameters() if p.ndim > 1]
        nodec = [p for n, p in model.named_parameters() if p.ndim <= 1]
        return torch.optim.AdamW([{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}], lr=lr)


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.w, self.gamma, self.ls = weight, gamma, label_smoothing

    def forward(self, logits, y):
        ce_w = F.cross_entropy(logits, y, weight=self.w, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** self.gamma * ce_w).mean()


def build_loss(cfg: dict, weights: torch.Tensor, device):
    ls, kind = float(cfg["train"]["label_smoothing"]), cfg["train"]["loss"]
    w = weights.to(device)
    if kind == "ce_class_weighted":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if kind == "focal":
        return FocalLoss(w, label_smoothing=ls)
    raise ValueError(f"unknown loss {kind}")


def lr_at(step, steps_per_epoch, base_lr, warmup_epochs, horizon_epochs, min_lr):
    """Per-step schedule: linear warm-up, cosine to min_lr over `horizon_epochs`, then constant min_lr."""
    warm, total = warmup_epochs * steps_per_epoch, horizon_epochs * steps_per_epoch
    if step < warm:
        return base_lr * (step + 1) / max(warm, 1)
    if step >= total:
        return min_lr
    p = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


# ----------------------------------------------------------------------------- loops
def _prep(x, mean_t, std_t, cuda):
    x = x.to("cuda" if cuda else "cpu", non_blocking=True).float().div_(255.0)
    x = (x - mean_t) / std_t
    return x.contiguous(memory_format=torch.channels_last) if cuda else x


@torch.no_grad()
def predict(model, loader, mean_t, std_t, cuda, amp):
    model.eval()
    L, Y, I = [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and cuda):
            out = model(_prep(x, mean_t, std_t, cuda))
        L.append(out.float().cpu()); Y.append(y); I.append(i)
    return torch.cat(L).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def _save_preds(path, ids, y, logits):
    from .metrics import softmax
    p = softmax(logits.astype(np.float64))
    df = pd.DataFrame({"image_id": ids, "y_true": y, "y_pred": p.argmax(1)})
    for c in range(p.shape[1]):
        df[f"p{c}"] = p[:, c].astype(np.float32)
    df.to_csv(path, index=False)


def train_one(cfg: dict, repo_root: str = ".", budget=None, log=print) -> dict:
    run_id, chash, run_dir = make_run_id(cfg), config_hash(cfg), run_dir_for(cfg)
    mgr = CheckpointManager(run_dir, run_id, chash, search_dirs=cfg["paths"].get("persist_dirs", []))
    if mgr.is_done():
        return {"run_id": run_id, "status": "already_done"}

    seed = int(get_by_path(cfg, "run.seed", 0))
    set_seed(seed, bool(cfg["train"].get("deterministic", False)))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    amp = bool(cfg["train"]["amp"]) and cuda
    K, size = int(cfg["data"]["num_classes"]), int(cfg["data"]["image_size"])
    tr = cfg["train"]
    bs, nw = int(tr["batch_size"]), int(tr["num_workers"])

    splits, root = load_split(cfg), image_root(cfg)
    dcfg = cfg["data"]
    mk = dict(size=size, input_mask=dcfg.get("input_mask", "none"), mask_corner=int(dcfg.get("mask_corner", 32)))
    ds_train = LeafDataset(splits["train"], root, True, aug=tr.get("aug"), **mk)
    ds_val = LeafDataset(splits["val"], root, False, **mk)
    val_loader = make_loader(ds_val, bs * 2, False, nw, seed, cuda)
    healthy_idx = list(dcfg["class_names"]).index(dcfg["healthy_class"])

    model, mean, std = build_model(cfg)
    model = model.to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    opt = build_optimizer(model, cfg)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception:                                      # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = build_loss(cfg, class_weights(splits["train"], K, float(tr["class_weight_power"])), device)

    steps_per_epoch = math.ceil(len(ds_train) / bs)
    cap, patience = int(tr["epochs"]), int(tr["early_stopping"]["patience"])
    min_delta = float(tr["early_stopping"]["min_delta"])
    start, best, best_epoch, bad, history = 0, -1.0, -1, 0, []
    state = mgr.load_last()
    if state is not None:
        c = restore_state(state, model=model, optimizer=opt, scaler=scaler)
        start, best, best_epoch, bad, history = (c["start_epoch"], c["best_metric"], c["best_epoch"],
                                                 c["bad_epochs"], c["history"])
        log(f"[{run_id}] resumed after epoch {start} (best val F1 {best:.4f} @ {best_epoch + 1})")
    log(f"[{run_id}] device={device} params={count_params(model) / 1e6:.2f}M train={len(ds_train)} "
        f"val={len(ds_val)} steps/epoch={steps_per_epoch}")

    finished = bad >= patience or start >= cap
    epoch = start - 1
    for epoch in range(start, cap if not finished else start):
        t0 = time.time()
        loader = make_loader(ds_train, bs, True, nw, seed * 1000 + epoch, cuda)   # epoch-level shuffle seed
        model.train()
        run_loss, n_seen = 0.0, 0
        for step, (x, y, _) in enumerate(loader):
            lr = lr_at(epoch * steps_per_epoch + step, steps_per_epoch, float(tr["lr"]), int(tr["warmup_epochs"]),
                       int(tr["horizon_epochs"]), float(tr["min_lr"]))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = _prep(x, mean_t, std_t, cuda), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            if float(tr["grad_clip"]) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach()) * len(y)
            n_seen += len(y)
        train_sec = time.time() - t0

        lv, yv, _ = predict(model, val_loader, mean_t, std_t, cuda, amp)
        pv = lv.argmax(1)
        val_f1 = float(f1_score(yv, pv, average="macro", labels=list(range(K)), zero_division=0))
        val_acc = float(accuracy_score(yv, pv))
        val_loss = float(F.cross_entropy(torch.from_numpy(lv), torch.from_numpy(yv)))
        ep_sec = time.time() - t0
        history.append({"epoch": epoch + 1, "lr_end": lr, "train_loss": run_loss / max(n_seen, 1),
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
                        "train_seconds": train_sec, "epoch_seconds": ep_sec})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = val_f1 > best + min_delta
        if improved:
            best, best_epoch, bad = val_f1, epoch, 0
            mgr.save_best({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1})
        else:
            bad += 1
        mgr.save_last(build_state(model=model, optimizer=opt, scaler=scaler, epoch=epoch, best_metric=best,
                                  best_epoch=best_epoch, bad_epochs=bad, history=history))
        log(f"[{run_id}] ep {epoch + 1:3d}/{cap} lr {lr:.2e} train_loss {run_loss / max(n_seen, 1):.4f} "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_F1 {val_f1:.4f} "
            f"best {best:.4f}@{best_epoch + 1} {'*' if improved else ''} ({ep_sec:.0f}s, "
            f"{n_seen / max(train_sec, 1e-9):.0f} img/s)")
        if bad >= patience:
            log(f"[{run_id}] early stopping (no improvement for {patience} epochs)")
            break
        if budget is not None and epoch + 1 < cap and budget.should_stop(ep_sec):
            log(f"[{run_id}] time budget reached: checkpoint saved, stopping (resume in the next session)")
            return {"run_id": run_id, "status": "interrupted", "epochs_done": epoch + 1}

    # ------------------------------------------------------------------ finalise from best-VALIDATION weights
    best_state = mgr.load_best()
    model.load_state_dict(best_state["model"])
    lv, yv, iv = predict(model, val_loader, mean_t, std_t, cuda, amp)
    m_val = compute_metrics(yv, lv, healthy_idx, K)
    np.save(run_dir / "val_logits.npy", lv)
    _save_preds(run_dir / "val_predictions.csv", iv, yv, lv)
    (run_dir / "metrics_val.json").write_text(json.dumps(m_val, indent=2))
    mgr.mark_done(best, best_epoch, extra={"epochs_run": len(history)})

    m_test = None
    if cfg.get("eval", {}).get("test", True) and not mgr.test_already_evaluated():
        ds_test = LeafDataset(splits["test"], root, False, **mk)
        lt, yt, it = predict(model, make_loader(ds_test, bs * 2, False, nw, seed, cuda), mean_t, std_t, cuda, amp)
        m_test = compute_metrics(yt, lt, healthy_idx, K)
        np.save(run_dir / "test_logits.npy", lt)
        _save_preds(run_dir / "test_predictions.csv", it, yt, lt)
        (run_dir / "metrics_test.json").write_text(json.dumps(m_test, indent=2))
        mgr.mark_test_evaluated({k: m_test[k] for k in ("macro_f1", "accuracy", "balanced_accuracy")})

    rec = make_record(
        cfg, "done", repo_root, lr=float(tr["lr"]), input_mask=dcfg.get("input_mask", "none"),
        best_epoch=best_epoch + 1, epochs_run=len(history), best_val_macro_f1=best,
        val_accuracy=m_val["accuracy"], val_balanced_accuracy=m_val["balanced_accuracy"],
        test_macro_f1=None if m_test is None else m_test["macro_f1"],
        test_accuracy=None if m_test is None else m_test["accuracy"],
        test_balanced_accuracy=None if m_test is None else m_test["balanced_accuracy"],
        test_auc_ovr=None if m_test is None else m_test["roc_auc_ovr_macro"],
        test_binary_accuracy=None if m_test is None else m_test["binary_accuracy"],
        test_binary_f1_diseased=None if m_test is None else m_test["binary_f1_diseased"],
        n_params_m=round(count_params(model) / 1e6, 3),
        model_mb=round(Path(mgr.best_path).stat().st_size / 1e6, 1),
        train_seconds=round(sum(h["epoch_seconds"] for h in history), 1))
    (run_dir / "final_record.json").write_text(json.dumps(rec, indent=2, default=str))
    try:
        mgr.last_path.unlink()          # resume state no longer needed; saves disk space
    except FileNotFoundError:
        pass
    log(f"[{run_id}] DONE in {human_time(rec['train_seconds'])}: best val F1 {best:.4f} @ epoch {best_epoch + 1}"
        + ("" if m_test is None else f", test macro-F1 {m_test['macro_f1']:.4f}"))
    del model
    if cuda:
        torch.cuda.empty_cache()
    return rec
