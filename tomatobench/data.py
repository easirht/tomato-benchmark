"""Data: reads the FROZEN split CSV, loads images with PIL, geometric-only augmentation, class weights.

Design notes
  * No torchvision dependency: augmentation is plain PIL/numpy/random, so it behaves the same on Kaggle,
    Colab and a laptop.
  * Datasets return uint8 CHW tensors; normalisation happens on the GPU (4x less data to move).
  * Baseline augmentation is GEOMETRIC ONLY (random resized crop, 90-degree rotations, flips). There is no
    colour / brightness augmentation on purpose: RQ2/RQ3 test lighting augmentation as a clean intervention.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

_TR = getattr(Image, "Transpose", Image)          # Pillow >= 9.1 has Image.Transpose
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
_FILL = 124                                        # mid-gray used for masked-out pixels (~ImageNet mean)


def find_color_root(search_root: str | Path) -> Path:
    """Find the PlantVillage `color` folder (a dir named 'color' that contains Tomato* subfolders).
    The Kaggle dataset has a duplicate nested copy; the shallowest match is used."""
    cands = []
    for d in Path(search_root).rglob("*"):
        if d.is_dir() and d.name.lower() == "color" and any(
                c.is_dir() and c.name.lower().startswith("tomato") for c in d.iterdir()):
            cands.append(d)
    if not cands:
        sys.exit(f"[ERROR] no 'color' folder with Tomato* subfolders under {search_root}")
    return sorted(cands, key=lambda p: len(p.parts))[0]


def image_root(cfg: dict) -> Path:
    r = cfg["data"].get("image_root")
    return Path(r) if r else find_color_root(cfg["paths"]["data_root"])


def load_split(cfg: dict) -> dict:
    """Return {'train': df, 'val': df, 'test': df} from the frozen split CSV."""
    f = Path(cfg["data"]["split_file"])
    if not f.is_absolute():
        f = Path(cfg["paths"]["splits_dir"]) / f
    df = pd.read_csv(f)
    need = {"image_id", "rel_path", "label", "split"}
    assert need <= set(df.columns), f"split file misses columns {need - set(df.columns)}"
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def class_weights(train_df: pd.DataFrame, num_classes: int, power: float = 1.0) -> torch.Tensor:
    """(N / (K * n_c)) ** power from TRAIN counts only, then rescaled to mean 1."""
    n_c = np.bincount(train_df.label.values, minlength=num_classes).astype(np.float64)
    w = (len(train_df) / (num_classes * np.maximum(n_c, 1))) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ----------------------------------------------------------------------------- augmentation
def random_resized_crop(img: Image.Image, size: int, scale, ratio) -> Image.Image:
    W, H = img.size
    area = W * H
    for _ in range(10):
        target = area * random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w, h = int(round(math.sqrt(target * ar))), int(round(math.sqrt(target / ar)))
        if 0 < w <= W and 0 < h <= H:
            i, j = random.randint(0, H - h), random.randint(0, W - w)
            return img.resize((size, size), _BILINEAR, box=(j, i, j + w, i + h))
    return img.resize((size, size), _BILINEAR)


def random_dihedral(img: Image.Image) -> Image.Image:
    k = random.randrange(4)
    if k == 1:
        img = img.transpose(_TR.ROTATE_90)
    elif k == 2:
        img = img.transpose(_TR.ROTATE_180)
    elif k == 3:
        img = img.transpose(_TR.ROTATE_270)
    if random.random() < 0.5:
        img = img.transpose(_TR.FLIP_LEFT_RIGHT)
    return img


def corners_only(arr: np.ndarray, c: int) -> np.ndarray:
    """Keep only the four c x c corner patches; everything else becomes mid-gray."""
    out = np.full_like(arr, _FILL)
    out[:c, :c], out[:c, -c:] = arr[:c, :c], arr[:c, -c:]
    out[-c:, :c], out[-c:, -c:] = arr[-c:, :c], arr[-c:, -c:]
    return out


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, train: bool, size: int = 224, aug: dict | None = None,
                 input_mask: str = "none", mask_corner: int = 32):
        self.paths = [str(Path(root) / p) for p in df.rel_path]
        self.labels = df.label.values.astype(np.int64)
        self.ids = df.image_id.values.astype(np.int64)
        self.train, self.size = train, size
        self.aug = aug or {}
        self.input_mask, self.mask_corner = input_mask, mask_corner

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            img = random_resized_crop(img, self.size, self.aug.get("crop_scale", (0.6, 1.0)),
                                      self.aug.get("crop_ratio", (0.9, 1.1)))
            if self.aug.get("dihedral", True):
                img = random_dihedral(img)
        elif img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), _BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if self.input_mask == "corners":
            arr = corners_only(arr, self.mask_corner)
        x = torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1)         # uint8, CHW (own writable copy)
        return x, int(self.labels[i]), int(self.ids[i])


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin: bool) -> torch.utils.data.DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin,
        drop_last=False, generator=g, worker_init_fn=seed_worker,
        prefetch_factor=4 if num_workers > 0 else None)
