"""Deliverable 3, step B2 (v2): near-duplicate groups -> group-aware 70/15/15 split -> purge -> leakage report.

Inputs (from steps A and B1, in results/d3): manifest_dedup.csv, embeddings.npy, summary.json.

WHY v2: v1 (single-linkage connected components at cosine >= 0.90) produced ONE chained group of 5,830
images mixing all 10 classes, which wrecked the 70/15/15 split. v2 fixes this in a principled way that was
fixed BEFORE any model is trained:

  1. Edges: image pairs with cosine >= T0 (default 0.90; visual review: same leaf).
  2. Connected components = groups. Any group larger than --max-group (default 100 images, about 0.55% of
     the data; the largest "ordinary" groups seen were ~30) is a chaining artefact. It is re-split by raising
     the threshold in steps of 0.005 inside that group only, until all pieces are <= max-group.
     Edges cut this way are counted and reported ("severed edges").
  3. Group-aware split with StratifiedGroupKFold. 40 folds, assigned 22/9/9 to train/val/test BEFORE the purge
     (55/22.5/22.5 %). Reason: the first v2 run (20 folds, 14/3/3) showed the purge moves about a third of the
     val/test images to train, ending at 79.7/9.9/10.3 %; the pre-purge shares were raised so that the FINAL
     split lands near the nominal 70/15/15. This choice uses only data structure, no model result.
  4. PURGE (exact, over ALL image pairs, not just top-k): any val/test image whose cosine to some train image
     is >= T0 (and any val image whose cosine to a test image is >= T0) is moved to train. Repeats until no
     such pair remains. So the final split has NO val/test image with a >= T0 near-copy in train, by
     construction and verified. Val/test end up slightly smaller than 15%; the exact numbers are printed.
  5. Residual-leakage report vs an ordinary random stratified split, plus EDA figures.

Limits (state in the paper): similarity is ResNet50-feature cosine; same-leaf pairs below T0 (e.g. rotated,
flipped, cropped) can remain -> measured/audited separately, reported as a limitation.

Run:  python scripts/d3_03_groups_and_split.py
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


# ----------------------------------------------------------------------------- neighbours / graph
def knn_edges(E, k, device, chunk=1024):
    """Unique (a < b, cosine) pairs among each image's top-k neighbours, as numpy arrays."""
    Et = torch.from_numpy(E.astype(np.float32)).to(device)
    n = len(E)
    A, B, C = [], [], []
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        s = Et[i:j] @ Et.T
        s[torch.arange(j - i), torch.arange(i, j)] = -2.0
        v, ix = s.topk(min(k, n - 1), dim=1)
        rows = torch.arange(i, j, device=device)[:, None].expand_as(ix)
        A.append(rows.reshape(-1).cpu().numpy()); B.append(ix.reshape(-1).cpu().numpy())
        C.append(v.reshape(-1).cpu().numpy())
    a, b, c = np.concatenate(A), np.concatenate(B), np.concatenate(C)
    df = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b), "c": c}).drop_duplicates(["a", "b"])
    return df.a.values, df.b.values, df.c.values.astype(np.float32)


def components(n, a, b, c, thr):
    m = c >= thr
    g = csr_matrix((np.ones(m.sum()), (a[m], b[m])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def refine_groups(n, a, b, c, t0, cap, step=0.005, tmax=1.0):
    """Components at t0; every component larger than `cap` is re-split at higher thresholds."""
    lab = components(n, a, b, c, t0)
    final = lab.copy()
    thr_final = np.full(n, t0, dtype=np.float32)
    next_id = int(lab.max()) + 1
    sizes = np.bincount(lab)
    stack = [(np.where(lab == g)[0], t0) for g in np.where(sizes > cap)[0]]
    n_forced = 0
    while stack:
        members, thr = stack.pop()
        thr2 = round(thr + step, 6)
        if thr2 > tmax + 1e-9:                       # cannot split further (e.g. identical features)
            final[members] = next_id; next_id += 1; thr_final[members] = thr; n_forced += len(members)
            continue
        pos = -np.ones(n, dtype=np.int64)
        pos[members] = np.arange(len(members))
        m = (pos[a] >= 0) & (pos[b] >= 0) & (c >= thr2)
        sub = csr_matrix((np.ones(m.sum()), (pos[a[m]], pos[b[m]])), shape=(len(members),) * 2)
        k, cl = connected_components(sub, directed=False)
        for g in range(k):
            idx = members[cl == g]
            if len(idx) > cap:
                stack.append((idx, thr2))
            else:
                final[idx] = next_id; next_id += 1; thr_final[idx] = thr2
    return final, thr_final, n_forced


def group_stats(groups, labels):
    sizes = pd.Series(groups).value_counts()
    mixed = pd.DataFrame({"g": groups, "y": labels}).groupby("g").y.nunique()
    return {"n_groups": int(len(sizes)), "n_singletons": int((sizes == 1).sum()),
            "pct_images_in_multi_groups": round(100 * sizes[sizes > 1].sum() / len(groups), 1),
            "largest": int(sizes.iloc[0]), "top5": [int(x) for x in sizes.iloc[:5]],
            "p99_size": float(np.percentile(sizes, 99)), "groups_mixing_classes": int((mixed > 1).sum())}


# ----------------------------------------------------------------------------- exact similarity
def max_sim(E, query_idx, ref_idx, device, chunk=2048):
    """For each query image: max cosine to ANY image in ref_idx (exact, all pairs)."""
    if len(query_idx) == 0 or len(ref_idx) == 0:
        return np.zeros(len(query_idx), dtype=np.float32)
    R = torch.from_numpy(E[ref_idx].astype(np.float32)).to(device)
    out = []
    for i in range(0, len(query_idx), chunk):
        q = torch.from_numpy(E[query_idx[i:i + chunk]].astype(np.float32)).to(device)
        out.append((q @ R.T).max(1).values.cpu().numpy())
    return np.concatenate(out)


def purge(E, split, t0, device, max_iter=30):
    """Move val/test images that have a >= t0 near-copy in train (or val images near a test image) to train."""
    split = split.copy()
    moved = np.zeros(len(split), dtype=bool)
    for it in range(max_iter):
        tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
        bad_va = (max_sim(E, va, tr, device) >= t0) | (max_sim(E, va, te, device) >= t0)
        bad_te = max_sim(E, te, tr, device) >= t0
        mv = np.concatenate([va[bad_va], te[bad_te]])
        print(f"  purge round {it + 1}: moving {len(mv)} images to train "
              f"(val {int(bad_va.sum())}, test {int(bad_te.sum())})")
        if len(mv) == 0:
            break
        split[mv] = "train"
        moved[mv] = True
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    assert (max_sim(E, va, tr, device) < t0).all() and (max_sim(E, te, tr, device) < t0).all()
    assert (max_sim(E, va, te, device) < t0).all()
    return split, moved


def leak_row(name, ms, t):
    return (f"{name:26s} mean={ms.mean():.3f}  p90={np.percentile(ms, 90):.3f}  "
            f">={t:.2f}: {100 * (ms >= t).mean():5.1f}%  >=0.88: {100 * (ms >= 0.88).mean():5.1f}%  "
            f">=0.86: {100 * (ms >= 0.86).mean():5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-dir", default=None)
    ap.add_argument("--t-cos", type=float, default=0.90, help="T0: cosine >= T0 means 'same leaf'")
    ap.add_argument("--max-group", type=int, default=100)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--no-purge", action="store_true")
    ap.add_argument("--diag", type=float, nargs="+", default=[0.97, 0.95, 0.92, 0.90, 0.88])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=40)
    ap.add_argument("--pre-folds", type=int, nargs=3, default=[22, 9, 9],
                    help="folds given to train/val/test before the purge (must sum to --folds)")
    ap.add_argument("--out-name", default="split_v2")
    args = ap.parse_args()

    cfg = load_config(REPO / "configs", repo_root=REPO)
    sp = cfg["protocol"]["split"]
    seed = int(sp["seed"])
    n_tr, n_va, n_te = args.pre_folds
    assert n_tr + n_va + n_te == args.folds, "--pre-folds must sum to --folds"

    roots = default_roots(REPO)
    d3 = Path(args.d3_dir) if args.d3_dir else Path(roots["work_root"]) / "results" / "d3"
    color_root = Path(json.loads((d3 / "summary.json").read_text())["color_root"])
    df = pd.read_csv(d3 / "manifest_dedup.csv")
    E = np.load(d3 / "embeddings.npy").astype(np.float32)
    assert len(df) == len(E), "manifest_dedup.csv and embeddings.npy out of sync; rerun step B1"
    n, y = len(df), df.label.values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} images, device={device}, seed={seed}, T0={args.t_cos}, max-group={args.max_group}")

    a, b, c = knn_edges(E, args.k, device)
    print(f"{len(a)} unique neighbour pairs (top-{args.k}); pairs with cosine >= T0: {(c >= args.t_cos).sum()}")

    print("\n=== PLAIN connected components (single linkage) at several thresholds ===")
    diag = {}
    for t in sorted(args.diag, reverse=True):
        st = group_stats(components(n, a, b, c, t), y)
        diag[f"plain_{t}"] = st
        print(f"T={t:.2f}: groups={st['n_groups']:6d} largest={st['largest']:5d} top5={st['top5']} "
              f"mixing classes={st['groups_mixing_classes']}")

    # ---- groups with size cap
    groups, thr_final, n_forced = refine_groups(n, a, b, c, args.t_cos, args.max_group, args.step)
    st = group_stats(groups, y)
    diag["refined"] = st
    e0 = c >= args.t_cos
    severed = int(((groups[a] != groups[b]) & e0).sum())
    print(f"\n=== REFINED groups (cap {args.max_group}) ===")
    print(f"groups={st['n_groups']} largest={st['largest']} top5={st['top5']} p99={st['p99_size']:.0f} "
          f"mixing classes={st['groups_mixing_classes']}  images in re-split groups="
          f"{int((thr_final > args.t_cos).sum())}  "
          f"forced (unsplittable)={n_forced}")
    print(f"edges >= T0: {int(e0.sum())}; severed by the refinement: {severed} "
          f"({100 * severed / max(e0.sum(), 1):.1f}%)  -> handled by the purge step below")
    df["group_id"] = groups
    df["group_size"] = df.group_id.map(pd.Series(groups).value_counts())
    df["group_thr"] = thr_final

    # ---- split
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    fold = np.full(n, -1)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(n), y, groups)):
        fold[te] = f
    order = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
    split0 = order[fold]
    t0_tab = pd.Series(split0).value_counts()
    print(f"\nsplit BEFORE purge: {t0_tab.to_dict()}  "
          f"({100 * t0_tab['train'] / n:.1f}/{100 * t0_tab['val'] / n:.1f}/{100 * t0_tab['test'] / n:.1f} %)")

    moved = np.zeros(n, dtype=bool)
    split = split0
    if not args.no_purge:
        print("\n=== PURGE (exact cosine over all pairs) ===")
        split, moved = purge(E, split0, args.t_cos, device)
    df["split"] = split
    df["moved_by_purge"] = moved
    span = df.groupby("group_id").split.nunique()
    n_span = int((span > 1).sum())
    print(f"\ngroups spanning >1 split after purge: {n_span} (only possible for groups whose members were moved)")

    tab = pd.crosstab(df.class_name, df.split).reindex(CLASS_ORDER)[["train", "val", "test"]].copy()
    tab["total"] = tab.sum(1)
    tab[["train%", "val%", "test%"]] = (tab[["train", "val", "test"]].div(tab.total, axis=0) * 100).round(1)
    print("\n=== class counts per split (FINAL) ===")
    print(tab.to_string())
    vc = df.split.value_counts()
    print(f"\nFINAL split sizes: {vc.to_dict()}  ({100 * vc['train'] / n:.1f}/{100 * vc['val'] / n:.1f}/"
          f"{100 * vc['test'] / n:.1f} %); moved by purge: {int(moved.sum())}")

    # ---- residual leakage vs random split
    idx = np.arange(n)
    tr_r, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va_r, te_r = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    tr_g, va_g, te_g = (idx[df.split.values == s] for s in ("train", "val", "test"))
    msim = np.full(n, np.nan, dtype=np.float32)          # per-image similarity to train (val/test only)
    msim[va_g] = max_sim(E, va_g, tr_g, device)
    msim[te_g] = max_sim(E, te_g, tr_g, device)
    df["max_sim_train"] = msim
    print("\n=== LEAKAGE: max cosine of each val/test image to ANY train image (no model trained) ===")
    res = {}
    for name, q, t in [("random split - val", va_r, tr_r), ("random split - test", te_r, tr_r),
                       ("GROUP split  - val", va_g, tr_g), ("GROUP split  - test", te_g, tr_g)]:
        ms = max_sim(E, q, t, device)
        res[name] = {"mean": float(ms.mean()), "p90": float(np.percentile(ms, 90)),
                     "pct_ge_T0": float(100 * (ms >= args.t_cos).mean()),
                     "pct_ge_0.88": float(100 * (ms >= 0.88).mean())}
        print(leak_row(name, ms, args.t_cos))

    # ---- save
    cols = ["image_id", "rel_path", "filename", "class_name", "label", "binary_label",
            "group_id", "group_size", "group_thr", "split", "moved_by_purge", "max_sim_train"]
    out_csv = d3 / f"{args.out_name}.csv"
    df[cols].to_csv(out_csv, index=False)
    repo_splits = Path(cfg["paths"]["splits_dir"]); repo_splits.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(repo_splits / f"{args.out_name}.csv", index=False)
    (d3 / f"{args.out_name}_summary.json").write_text(json.dumps(
        {"t0": args.t_cos, "max_group": args.max_group, "step": args.step, "seed": seed,
         "folds": [n_tr, n_va, n_te], "n_images": n, "severed_edges": severed,
         "n_moved_by_purge": int(moved.sum()), "diagnostics": diag, "leakage": res,
         "class_by_split": tab.to_dict()}, indent=2))
    print(f"\nSaved {out_csv}\n  and a copy in {repo_splits}  (commit this file to GitHub!)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        tab[["train", "val", "test"]].plot(kind="bar", stacked=True, ax=ax[0])
        ax[0].set_title("Images per class and split"); ax[0].set_ylabel("images")
        sizes = df.groupby("group_id").size()
        ax[1].hist(sizes, bins=np.arange(1, min(sizes.max(), 100) + 2) - 0.5, log=True)
        ax[1].set_title(f"Group size (T0={args.t_cos}, cap {args.max_group}); largest = {sizes.max()}")
        ax[1].set_xlabel("images per group"); ax[1].set_ylabel("groups (log)")
        plt.tight_layout(); plt.savefig(d3 / "eda_counts_groups.png", dpi=120); plt.close()
        rng = np.random.default_rng(seed)
        fig, ax = plt.subplots(len(CLASS_ORDER), 6, figsize=(12, 2 * len(CLASS_ORDER)))
        for r, cname in enumerate(CLASS_ORDER):
            rows = df[df.class_name == cname]
            for k, i in enumerate(rng.choice(len(rows), 6, replace=False)):
                ax[r, k].imshow(Image.open(color_root / rows.rel_path.iloc[i]).convert("RGB"))
                ax[r, k].axis("off")
            ax[r, 0].set_title(cname, fontsize=8, loc="left")
        plt.tight_layout(); plt.savefig(d3 / "eda_samples.png", dpi=80); plt.close()
        print("Saved eda_counts_groups.png, eda_samples.png")
    except Exception as e:  # noqa: BLE001
        print("EDA plots skipped:", repr(e))


if __name__ == "__main__":
    main()
