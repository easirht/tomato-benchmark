"""Platform detection (Kaggle / Colab / local) and environment snapshots.

Everything that differs between Kaggle, Colab and a local machine lives here, so the rest
of the code never hard-codes a path like /kaggle/working.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform as _platform
import shutil
import subprocess
import sys
from pathlib import Path

KEY_PACKAGES = [
    "torch", "torchvision", "timm", "segmentation-models-pytorch", "transformers",
    "albumentations", "opencv-python-headless", "numpy", "pandas", "scikit-learn",
    "scipy", "statsmodels", "imagehash", "PyYAML",
]


def detect_platform() -> str:
    """Return 'kaggle', 'colab' or 'local'."""
    if os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def default_roots(repo_root: str | Path = ".") -> dict:
    """Default (data_root, work_root) for the detected platform.

    Kaggle: datasets are read-only under /kaggle/input, outputs go to /kaggle/working.
    Colab : /content is fast but wiped on disconnect (mount Drive to persist).
    Local : ./data and ./work inside the repo.
    """
    p = detect_platform()
    repo_root = Path(repo_root).resolve()
    if p == "kaggle":
        return {"platform": p, "data_root": "/kaggle/input", "work_root": "/kaggle/working"}
    if p == "colab":
        return {"platform": p, "data_root": "/content/data", "work_root": "/content/work"}
    return {"platform": p, "data_root": str(repo_root / "data"), "work_root": str(repo_root / "work")}


def git_commit(repo_root: str | Path = ".") -> str:
    """Short git commit hash (+'-dirty' if there are uncommitted changes), or 'nogit'."""
    try:
        h = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "-C", str(repo_root), "diff", "--quiet"],
                                stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def package_versions(names=KEY_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not-installed"
    return out


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"cuda": True, "name": props.name,
                    "memory_gb": round(props.total_memory / 1024 ** 3, 1),
                    "cuda_version": torch.version.cuda, "count": torch.cuda.device_count()}
        return {"cuda": False}
    except Exception as e:  # torch missing or broken
        return {"cuda": False, "error": repr(e)}


def env_snapshot(repo_root: str | Path = ".") -> dict:
    """Everything needed to describe the software/hardware a run happened on."""
    total, used, free = shutil.disk_usage(".")
    return {
        "platform": detect_platform(),
        "python": sys.version.split()[0],
        "os": _platform.platform(),
        "git_commit": git_commit(repo_root),
        "gpu": gpu_info(),
        "disk_free_gb": round(free / 1024 ** 3, 1),
        "packages": package_versions(),
    }
