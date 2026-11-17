"""Deliverable 3, step B2: near-duplicate groups -> group-aware 70/15/15 split -> leakage report + EDA.

Inputs (from steps A and B1, in results/d3): manifest_dedup.csv, embeddings.npy, summary.json.

What it does
  1. Rebuilds each image's k nearest neighbours from embeddings.npy (exact cosine search).
  2. DIAGNOSTICS for several cosine thresholds T: connect every pair with cosine >= T, take connected
     components as "groups", and report how many groups, how big the largest is (chaining danger),
     and how many groups mix classes. Threshold is chosen from these numbers + your visual montage
     review, BEFORE any model is trained.
  3. Builds the final group-aware split at --t-cos (default 0.90) with StratifiedGroupKFold
     (20 folds -> 14 train / 3 val / 3 test = 70/15/15), seed from configs/base.yaml.
  4. RESIDUAL-LEAKAGE check without training any model: for every val/test image, the maximum cosine
     similarity to any TRAIN image, compared with an ordinary random stratified split.
     (Fractions >= T are ~0 for the group split BY CONSTRUCTION; the informative columns are the lower
     ones, which show same-leaf pairs the threshold may still miss.)
  5. EDA: class counts per split, group-size histogram, sample grid per class.

Run:  python scripts/d3_03_groups_and_split.py            (CPU is fine; GPU used if available)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tomatobench.config import load_config  # noqa: E402
from tomatobench.env import default_roots  # noqa: E402

CLASS_ORDER = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot",
               "spider_mites", "target_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy"]


def knn_pairs(E, k, device, chunk=1024):
    """All unique (a, b, cos) pairs among each image's top-k neighbours."""
    Et = torch.from_numpy(E.astype(np.float32)).to(device)
    n = len(E)
    a_all, b_all, c_all = [], [], []
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        s = Et[i:j] @ Et.T
        s[torch.arange(j - i), torch.arange(i, j)] = -2.0
        v, ix = s.topk(min(k, n - 1), dim=1)
        rows = torch.arange(i, j, device=device)[:, None].expand_as(ix)
        a_all.append(rows.reshape(-1).cpu().numpy())
        b_all.append(ix.reshape(-1).cpu().numpy())
        c_all.append(v.reshape(-1).cpu().numpy())
    a, b, c = np.concatenate(a_all), np.concatenate(b_all), np.concatenate(c_all)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    P = pd.DataFrame({"a": lo, "b": hi, "cos": c}).drop_duplicates(["a", "b"]).reset_index(drop=True)
    return P


def components(n, P, thr):
    sel = P[P.cos >= thr]
    g = csr_matrix((np.ones(len(sel)), (sel.a.values, sel.b.values)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def group_stats(groups, labels):
    sizes = pd.Series(groups).value_counts()
    mixed = pd.DataFrame({"g": groups, "y": labels}).groupby("g").y.nunique()
    return {"n_groups": int(len(sizes)), "n_singletons": int((sizes == 1).sum()),
            "pct_images_in_multi_groups": round(100 * sizes[sizes > 1].sum() / len(groups), 1),
            "largest": int(sizes.iloc[0]), "top5": [int(x) for x in sizes.iloc[:5]],
            "p99_size": float(np.percentile(sizes, 99)), "groups_mixing_classes": int((mixed > 1).sum())}


def max_sim_to_train(E, query_idx, train_idx, device, chunk=2048):
    Tr = torch.from_numpy(E[train_idx].astype(np.float32)).to(device)
    out = []
    for i in range(0, len(query_idx), chunk):
        q = torch.from_numpy(E[query_idx[i:i + chunk]].astype(np.float32)).to(device)
        out.append((q @ Tr.T).max(1).values.cpu().numpy())
    return np.concatenate(out)


def leak_row(name, ms, t):
    return (f"{name:28s} mean={ms.mean():.3f}  p90={np.percentile(ms, 90):.3f}  "
            f">={t:.2f}: {100 * (ms >= t).mean():5.1f}%  >=0.88: {100 * (ms >= 0.88).mean():5.1f}%  "
            f">=0.86: {100 * (ms >= 0.86).mean():5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-dir", default=None)
    ap.add_argument("--t-cos", type=float, default=0.90, help="threshold used for the FINAL groups")
    ap.add_argument("--diag", type=float, nargs="+", default=[0.97, 0.95, 0.92, 0.90, 0.88, 0.86])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--out-name", default="split_v1")
    args = ap.parse_args()

    cfg = load_config(REPO / "configs", repo_root=REPO)
    sp = cfg["protocol"]["split"]
    seed = int(sp["seed"])
    n_tr, n_va, n_te = [round(sp[k] * args.folds) for k in ("train", "val", "test")]
    assert n_tr + n_va + n_te == args.folds, "split ratios must be multiples of 1/folds"

    roots = default_roots(REPO)
    d3 = Path(args.d3_dir) if args.d3_dir else Path(roots["work_root"]) / "results" / "d3"
    color_root = Path(json.loads((d3 / "summary.json").read_text())["color_root"])
    df = pd.read_csv(d3 / "manifest_dedup.csv")
    E = np.load(d3 / "embeddings.npy").astype(np.float32)
    assert len(df) == len(E), "manifest_dedup.csv and embeddings.npy are out of sync; rerun step B1"
    n, y = len(df), df.label.values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} images, device={device}, seed={seed}, folds -> train/val/test = {n_tr}/{n_va}/{n_te}")

    P = knn_pairs(E, args.k, device)
    print(f"{len(P)} unique neighbour pairs (top-{args.k})")

    # ---- 2. threshold diagnostics
    print("\n=== GROUP DIAGNOSTICS (connected components at each cosine threshold) ===")
    diag = {}
    for t in sorted(args.diag, reverse=True):
        st = group_stats(components(n, P, t), y)
        diag[str(t)] = st
        print(f"T={t:.2f}: groups={st['n_groups']:6d}  singletons={st['n_singletons']:6d}  "
              f"images in multi-image groups={st['pct_images_in_multi_groups']:5.1f}%  "
              f"largest={st['largest']:5d}  top5={st['top5']}  p99={st['p99_size']:.0f}  "
              f"groups mixing classes={st['groups_mixing_classes']}")

    # ---- 3. final groups + split
    groups = components(n, P, args.t_cos)
    df["group_id"] = groups
    df["group_size"] = df.group_id.map(pd.Series(groups).value_counts())
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    fold = np.full(n, -1)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(n), y, groups)):
        fold[te] = f
    order = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)   # fold index -> split
    df["split"] = order[fold]

    # sanity: no group spans two splits
    span = df.groupby("group_id").split.nunique()
    assert (span == 1).all(), "BUG: a group is split across train/val/test"
    P_t = P[P.cos >= args.t_cos]
    cross = (df.split.values[P_t.a] != df.split.values[P_t.b]).sum()
    print(f"\nFINAL T={args.t_cos}: groups={df.group_id.nunique()}, edges crossing splits: {cross} (must be 0)")

    tab = pd.crosstab(df.class_name, df.split).reindex(CLASS_ORDER)[["train", "val", "test"]].copy()
    tab["total"] = tab.sum(1)
    tab[["train%", "val%", "test%"]] = (tab[["train", "val", "test"]].div(tab.total, axis=0) * 100).round(1)
    print("\n=== class counts per split ===")
    print(tab.to_string())
    print(f"\nsplit sizes: {df.split.value_counts().to_dict()}")

    # ---- 4. residual leakage vs a random stratified split
    idx = np.arange(n)
    tr_r, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va_r, te_r = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    tr_g, va_g, te_g = (idx[df.split.values == s] for s in ("train", "val", "test"))
    print("\n=== RESIDUAL LEAKAGE: max cosine of each val/test image to ANY train image ===")
    print("(no model trained. Higher = more near-copies of test images sit in train.)")
    res = {}
    for name, q, t in [("random split  - val", va_r, tr_r), ("random split  - test", te_r, tr_r),
                       ("GROUP split   - val", va_g, tr_g), ("GROUP split   - test", te_g, tr_g)]:
        ms = max_sim_to_train(E, q, t, device)
        res[name.strip()] = {"mean": float(ms.mean()), "p90": float(np.percentile(ms, 90)),
                             "pct_ge_T": float(100 * (ms >= args.t_cos).mean()),
                             "pct_ge_0.88": float(100 * (ms >= 0.88).mean())}
        print(leak_row(name, ms, args.t_cos))

    # ---- save
    out_csv = d3 / f"{args.out_name}.csv"
    cols = ["image_id", "rel_path", "filename", "class_name", "label", "binary_label",
            "group_id", "group_size", "split"]
    df[cols].to_csv(out_csv, index=False)
    repo_splits = Path(cfg["paths"]["splits_dir"])
    repo_splits.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(repo_splits / f"{args.out_name}.csv", index=False)
    (d3 / f"{args.out_name}_summary.json").write_text(json.dumps(
        {"t_cos": args.t_cos, "seed": seed, "folds": [n_tr, n_va, n_te], "n_images": n,
         "diagnostics": diag, "residual_leakage": res, "class_by_split": tab.to_dict()}, indent=2))
    print(f"\nSaved {out_csv}\n  and a copy in {repo_splits}  (commit this file to GitHub!)")

    # ---- 5. EDA figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        tab[["train", "val", "test"]].plot(kind="bar", stacked=True, ax=ax[0])
        ax[0].set_title("Images per class and split"); ax[0].set_ylabel("images")
        sizes = df.groupby("group_id").size()
        ax[1].hist(sizes, bins=np.arange(1, min(sizes.max(), 60) + 2) - 0.5, log=True)
        ax[1].set_title(f"Group size (T={args.t_cos}); largest = {sizes.max()}")
        ax[1].set_xlabel("images per group"); ax[1].set_ylabel("groups (log)")
        plt.tight_layout(); plt.savefig(d3 / "eda_counts_groups.png", dpi=120); plt.close()
        rng = np.random.default_rng(seed)
        fig, ax = plt.subplots(len(CLASS_ORDER), 6, figsize=(12, 2 * len(CLASS_ORDER)))
        for r, c in enumerate(CLASS_ORDER):
            rows = df[df.class_name == c]
            for k, i in enumerate(rng.choice(len(rows), 6, replace=False)):
                ax[r, k].imshow(Image.open(color_root / rows.rel_path.iloc[i]).convert("RGB"))
                ax[r, k].axis("off")
            ax[r, 0].set_title(c, fontsize=8, loc="left")
        plt.tight_layout(); plt.savefig(d3 / "eda_samples.png", dpi=80); plt.close()
        print("Saved eda_counts_groups.png, eda_samples.png")
    except Exception as e:  # noqa: BLE001 - plots are optional
        print("EDA plots skipped:", repr(e))


if __name__ == "__main__":
    main()
