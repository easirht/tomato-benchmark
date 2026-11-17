"""Deliverable 3, step B1: find near-duplicate image pairs and make montages for visual review.

Why: PlantVillage has several photos of the SAME leaf. If such photos land in both train and test,
accuracy is inflated. Here we only FIND candidate pairs and let you LOOK at them, so the
similarity threshold is chosen by eye BEFORE any model result exists (no tuning on test results).

Steps
  1. Read results/d3/manifest.csv (from step A). Keep ONE file per exact-duplicate (MD5) group and
     write manifest_dedup.csv + exact_dup_dropped.csv.
  2. Embed every kept image with an ImageNet CNN (default: timm resnet50, global-pool features,
     L2-normalised). Cosine similarity of two embeddings is high for the same leaf/shot.
  3. For each image find its k nearest neighbours (exact search, chunked matrix product).
  4. Add a perceptual-hash (pHash, 64 bit) Hamming distance for each pair as a second opinion.
  5. Write pairs_cos090.csv (all unique pairs with cosine >= 0.90) and montage_*.png: random
     pairs from several cosine bins, shown side by side, for you to judge by eye.

Turn the GPU ON for this script (a few minutes). On CPU use --embedder tiny only for debugging.

    python scripts/d3_02_near_duplicates.py
"""
import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tomatobench.env import default_roots  # noqa: E402

BINS = [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.92, 0.95), (0.90, 0.92)]


# ----------------------------------------------------------------------------- embeddings
def embed_timm(paths, model_name, device, batch_size=128, workers=2):
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return tf(Image.open(paths[i]).convert("RGB"))

    dl = torch.utils.data.DataLoader(DS(), batch_size=batch_size, num_workers=workers, shuffle=False)
    out = []
    with torch.no_grad():
        for n, x in enumerate(dl):
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                f = model(x.to(device))
            out.append(f.float().cpu())
            if n % 20 == 0:
                print(f"  embedded {min((n + 1) * batch_size, len(paths))}/{len(paths)}", flush=True)
    E = torch.cat(out).numpy()
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)


def embed_tiny(paths, size=16):
    """Debug-only embedder: tiny grayscale thumbnail. NOT good enough for the real analysis."""
    feats = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32).ravel()
        feats.append(a - a.mean())
    E = np.stack(feats)
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)


# ----------------------------------------------------------------------------- neighbours
def knn_cosine(E, k, device, chunk=1024):
    Et = torch.from_numpy(E.astype(np.float32)).to(device)
    N = len(E)
    sims = np.zeros((N, k), np.float32)
    idxs = np.zeros((N, k), np.int64)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        s = Et[i:j] @ Et.T
        s[torch.arange(j - i), torch.arange(i, j)] = -2.0      # ignore self-match
        v, ix = s.topk(k, dim=1)
        sims[i:j], idxs[i:j] = v.cpu().numpy(), ix.cpu().numpy()
    return sims, idxs


def phash_int(path):
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("L"))
    return int(str(h), 16)


# ----------------------------------------------------------------------------- montage
def make_montage(pairs, color_root, df, out_png, cols=3, tile=256):
    """pairs: list of (i, j, cos, ham). One tile = image A | image B + caption."""
    cap_h, pad = 18, 6
    rows = math.ceil(len(pairs) / cols)
    W, H = cols * (2 * tile + pad), rows * (tile + cap_h + pad)
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    for n, (i, j, c, ham) in enumerate(pairs):
        x0, y0 = (n % cols) * (2 * tile + pad), (n // cols) * (tile + cap_h + pad)
        for k, idx in enumerate((i, j)):
            im = Image.open(Path(color_root) / df.rel_path.iloc[idx]).convert("RGB").resize((tile, tile))
            canvas.paste(im, (x0 + k * tile, y0 + cap_h))
        same = "same class" if df.label.iloc[i] == df.label.iloc[j] else "DIFFERENT class"
        d.text((x0 + 2, y0 + 3), f"#{n + 1} cos={c:.3f} phash={ham} {same}", fill=(255, 255, 0))
    canvas.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-dir", default=None, help="step A output dir (default <work_root>/results/d3)")
    ap.add_argument("--embedder", choices=["timm", "tiny"], default="timm")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pairs-per-bin", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    roots = default_roots(Path(__file__).resolve().parents[1])
    d3 = Path(args.d3_dir) if args.d3_dir else Path(roots["work_root"]) / "results" / "d3"
    color_root = Path(json.loads((d3 / "summary.json").read_text())["color_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"color root: {color_root}\nd3 dir: {d3}\ndevice: {device}  embedder: {args.embedder}")

    # ---- 1. exact duplicates: keep the first file of each MD5 group
    m = pd.read_csv(d3 / "manifest.csv")
    m = m[m["error"].isna() | (m["error"] == "")].sort_values("image_id")
    kept = m.drop_duplicates("md5", keep="first")
    dropped = m[~m.image_id.isin(kept.image_id)]
    df = kept.reset_index(drop=True)
    df.to_csv(d3 / "manifest_dedup.csv", index=False)
    dropped[["image_id", "rel_path", "class_name", "md5"]].to_csv(d3 / "exact_dup_dropped.csv", index=False)
    print(f"\nexact duplicates: dropped {len(dropped)} files; {len(df)} images remain")

    paths = [str(color_root / p) for p in df.rel_path]

    # ---- 2. embeddings
    if args.embedder == "timm":
        E = embed_timm(paths, args.model, device)
    else:
        E = embed_tiny(paths)
    np.save(d3 / "embeddings.npy", E.astype(np.float16))

    # ---- 3. nearest neighbours
    sims, idxs = knn_cosine(E, min(args.k, len(df) - 1), device)

    # ---- 4. pHash for the images that appear in some candidate pair
    print("computing pHash ...")
    with ThreadPoolExecutor(8) as ex:
        ph = np.array(list(ex.map(phash_int, paths)), dtype=np.uint64)

    rows = []
    for i in range(len(df)):
        for r in range(sims.shape[1]):
            j = int(idxs[i, r])
            if sims[i, r] >= 0.90:
                a, b = (i, j) if i < j else (j, i)     # unique pair, order-independent
                rows.append((a, b, float(sims[i, r])))
    pairs = pd.DataFrame(rows, columns=["a", "b", "cos"]).drop_duplicates(["a", "b"])
    if len(pairs):
        pairs["phash_dist"] = [bin(int(ph[a]) ^ int(ph[b])).count("1") for a, b in zip(pairs.a, pairs.b)]
        pairs["id_a"], pairs["id_b"] = df.image_id.values[pairs.a], df.image_id.values[pairs.b]
        pairs["class_a"], pairs["class_b"] = df.class_name.values[pairs.a], df.class_name.values[pairs.b]
        pairs["same_class"] = pairs.class_a == pairs.class_b
    pairs.to_csv(d3 / "pairs_cos090.csv", index=False)

    # ---- 5. report + montages
    top1 = sims[:, 0]
    print("\n=== nearest-neighbour cosine of every image (top-1) ===")
    for q in (50, 75, 90, 95, 99):
        print(f"  {q}th percentile: {np.percentile(top1, q):.3f}")
    print("\n=== unique pairs by cosine bin ===")
    summary = {"n_images": len(df), "n_exact_dropped": int(len(dropped)), "bins": []}
    rng = np.random.default_rng(args.seed)
    for lo, hi in BINS:
        sel = pairs[(pairs.cos >= lo) & (pairs.cos < hi)] if len(pairs) else pairs
        n_cross = int((~sel.same_class).sum()) if len(sel) else 0
        print(f"  cos in [{lo:.2f}, {min(hi, 1.0):.2f}): {len(sel):6d} pairs   cross-class: {n_cross}")
        summary["bins"].append({"lo": lo, "hi": min(hi, 1.0), "n_pairs": int(len(sel)), "n_cross_class": n_cross})
        if len(sel):
            pick = sel.sample(min(args.pairs_per_bin, len(sel)), random_state=int(rng.integers(1e9)))
            tiles = [(int(r.a), int(r.b), float(r.cos), int(r.phash_dist)) for r in pick.itertuples()]
            make_montage(tiles, color_root, df, d3 / f"montage_cos_{lo:.2f}_{min(hi, 1.0):.2f}.png")
    if len(pairs):
        print(f"\npHash Hamming <= 4 among candidate pairs: {(pairs.phash_dist <= 4).sum()}  "
              f"(<= 8: {(pairs.phash_dist <= 8).sum()})")
        cross = pairs[(~pairs.same_class) & (pairs.cos >= 0.95)]
        print(f"cross-class pairs with cos >= 0.95: {len(cross)} (possible label conflicts)")
        cross.to_csv(d3 / "cross_class_pairs_cos095.csv", index=False)
    (d3 / "near_dup_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved pairs_cos090.csv, montage_*.png, manifest_dedup.csv in {d3}")


if __name__ == "__main__":
    main()
