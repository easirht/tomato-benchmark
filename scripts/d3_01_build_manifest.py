"""Deliverable 3, step A: build the image manifest and verify counts.

What it does (CPU only, turn GPU OFF):
  1. Finds the PlantVillage 'color' folder (or use --root).
  2. Maps the 10 Tomato___* folders to our canonical class names (fails loudly if a folder
     name is not recognised, instead of silently guessing).
  3. Reads EVERY image fully: records width/height/mode/bytes, and flags unreadable files.
  4. Computes an MD5 of each file -> finds EXACT duplicate files (same bytes).
  5. Writes results/d3/manifest.csv, class_counts.csv, exact_duplicates.csv, bad_images.csv,
     summary.json and prints a readable report.

Nothing here touches labels/splits yet. Near-duplicate detection and the group-aware split come
in step B, after you have looked at this report.

Usage (Kaggle):  python scripts/d3_01_build_manifest.py
         custom: python scripts/d3_01_build_manifest.py --root /path/to/color --out /some/dir
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tomatobench.env import default_roots  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# keyword (found in the lower-cased folder name) -> canonical class name. Each keyword must match
# exactly one folder. Order of CLASS_ORDER defines label ids 0..9 (same as configs/base.yaml).
KEYWORDS = {
    "bacterial": "bacterial_spot",
    "early_blight": "early_blight",
    "late_blight": "late_blight",
    "leaf_mold": "leaf_mold",
    "septoria": "septoria_leaf_spot",
    "spider": "spider_mites",
    "target": "target_spot",
    "yellow": "yellow_leaf_curl_virus",
    "mosaic": "mosaic_virus",
    "healthy": "healthy",
}
CLASS_ORDER = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot",
               "spider_mites", "target_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy"]


def find_color_root(search_root: Path) -> Path:
    """Find a directory named 'color' that contains Tomato* subfolders."""
    cands = []
    for d in search_root.rglob("*"):
        if d.is_dir() and d.name.lower() == "color":
            if any(c.is_dir() and c.name.lower().startswith("tomato") for c in d.iterdir()):
                cands.append(d)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        sys.exit(f"[ERROR] No 'color' folder with Tomato* subfolders under {search_root}. "
                 "Is the dataset attached? Use --root to point at it.")
    sys.exit("[ERROR] Several candidates, pass one with --root:\n  " + "\n  ".join(map(str, cands)))


def map_folders(color_root: Path) -> dict:
    tomato_dirs = sorted(d for d in color_root.iterdir() if d.is_dir() and d.name.lower().startswith("tomato"))
    mapping, used = {}, set()
    for d in tomato_dirs:
        key = d.name.lower().replace(" ", "_")
        hits = [k for k in KEYWORDS if k in key]
        if len(hits) != 1:
            sys.exit(f"[ERROR] Cannot map folder {d.name!r} (keyword hits: {hits}). Fix KEYWORDS.")
        cls = KEYWORDS[hits[0]]
        if cls in used:
            sys.exit(f"[ERROR] Two folders map to class {cls}: {d.name!r}")
        used.add(cls)
        mapping[d] = cls
    missing = set(CLASS_ORDER) - used
    if missing:
        sys.exit(f"[ERROR] No folder found for classes: {sorted(missing)}")
    return mapping


def inspect(path: Path) -> dict:
    """Fully decode one image. Never raises; reports problems in the 'error' field."""
    rec = {"bytes": path.stat().st_size, "md5": "", "width": 0, "height": 0, "mode": "", "error": ""}
    try:
        data = path.read_bytes()
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            im.load()                      # full decode: catches truncated files
            rec["width"], rec["height"], rec["mode"] = im.width, im.height, im.mode
    except Exception as e:                 # noqa: BLE001 - we want to record any failure
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="path of the PlantVillage 'color' folder")
    ap.add_argument("--search", default=None, help="where to look for it (default: data_root)")
    ap.add_argument("--out", default=None, help="output dir (default: <work_root>/results/d3)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    roots = default_roots(Path(__file__).resolve().parents[1])
    color_root = Path(args.root) if args.root else find_color_root(Path(args.search or roots["data_root"]))
    out = Path(args.out) if args.out else Path(roots["work_root"]) / "results" / "d3"
    out.mkdir(parents=True, exist_ok=True)
    print(f"color root : {color_root}\noutput dir : {out}\n")

    mapping = map_folders(color_root)
    files, non_images = [], Counter()
    for d, cls in mapping.items():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                files.append((f, d.name, cls))
            elif f.is_file():
                non_images[f.suffix.lower() or "(none)"] += 1
    print(f"found {len(files)} image files in {len(mapping)} tomato folders; "
          f"non-image files: {dict(non_images) or 'none'}")

    try:
        from tqdm import tqdm
        it = lambda x, n: tqdm(x, total=n)  # noqa: E731
    except ImportError:
        it = lambda x, n: x  # noqa: E731
    with ThreadPoolExecutor(args.workers) as ex:
        infos = list(it(ex.map(lambda t: inspect(t[0]), files), len(files)))

    rows = []
    for (f, folder, cls), info in zip(files, infos):
        rows.append({"image_id": len(rows), "rel_path": str(f.relative_to(color_root)),
                     "filename": f.name, "folder": folder, "class_name": cls,
                     "label": CLASS_ORDER.index(cls),
                     "binary_label": 0 if cls == "healthy" else 1, **info})

    fields = list(rows[0].keys())
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["error"]]
    with open(out / "bad_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rel_path", "error"])
        w.writeheader()
        for r in bad:
            w.writerow({"rel_path": r["rel_path"], "error": r["error"]})

    by_md5 = defaultdict(list)
    for r in rows:
        if r["md5"] and not r["error"]:
            by_md5[r["md5"]].append(r)
    dup_groups = {h: v for h, v in by_md5.items() if len(v) > 1}
    with open(out / "exact_duplicates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["md5", "n_files", "classes", "rel_paths"])
        for h, v in dup_groups.items():
            w.writerow([h, len(v), "|".join(sorted({x["class_name"] for x in v})),
                        "|".join(x["rel_path"] for x in v)])

    # ---- per-class table
    per = defaultdict(lambda: {"n": 0, "bad": 0, "extra_dups": 0})
    for r in rows:
        per[r["class_name"]]["n"] += 1
        per[r["class_name"]]["bad"] += bool(r["error"])
    for v in dup_groups.values():
        per[v[0]["class_name"]]["extra_dups"] += len(v) - 1
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_name", "folder", "n_files", "n_unreadable", "n_extra_exact_dups"])
        folder_of = {c: d.name for d, c in mapping.items()}
        for c in CLASS_ORDER:
            w.writerow([c, folder_of[c], per[c]["n"], per[c]["bad"], per[c]["extra_dups"]])

    counts = [per[c]["n"] for c in CLASS_ORDER]
    sizes = Counter((r["width"], r["height"]) for r in rows if not r["error"])
    modes = Counter(r["mode"] for r in rows if not r["error"])
    cross = [v for v in dup_groups.values() if len({x["class_name"] for x in v}) > 1]
    summary = {
        "color_root": str(color_root), "n_images": len(rows), "n_unreadable": len(bad),
        "class_counts": dict(zip(CLASS_ORDER, counts)),
        "imbalance_ratio_max_over_min": round(max(counts) / min(counts), 2),
        "n_exact_duplicate_groups": len(dup_groups),
        "n_extra_files_in_exact_duplicate_groups": sum(len(v) - 1 for v in dup_groups.values()),
        "n_duplicate_groups_spanning_multiple_classes": len(cross),
        "top_image_sizes": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(5)],
        "modes": dict(modes),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== CLASS COUNTS (verify against the dataset page) ===")
    print(f"{'class':26s}{'files':>7s}{'bad':>5s}{'extra exact dups':>18s}   folder")
    for c in CLASS_ORDER:
        print(f"{c:26s}{per[c]['n']:7d}{per[c]['bad']:5d}{per[c]['extra_dups']:18d}   {folder_of[c]}")
    print(f"{'TOTAL':26s}{len(rows):7d}{len(bad):5d}{summary['n_extra_files_in_exact_duplicate_groups']:18d}")
    print(f"\nimbalance (largest/smallest class): {summary['imbalance_ratio_max_over_min']}x")
    print(f"image sizes (top 5): {summary['top_image_sizes']}   modes: {dict(modes)}")
    print(f"exact-duplicate groups: {len(dup_groups)} "
          f"(spanning >1 class: {len(cross)})   unreadable files: {len(bad)}")
    print("\n=== 3 sample file names per class (to understand the naming pattern) ===")
    for c in CLASS_ORDER:
        print(f"  {c:24s}", [r["filename"] for r in rows if r["class_name"] == c][:3])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
