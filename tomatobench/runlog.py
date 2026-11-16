"""One CSV row per run (results/runs.csv): run_id, model, seed, config hash, git commit,
library versions, status and metrics. Every table in the report is built from this CSV only."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from .config import config_hash, get_by_path, make_run_id
from .env import env_snapshot

BASE_FIELDS = ["run_id", "exp_name", "model", "family", "seed", "config_hash", "git_commit",
               "platform", "gpu", "python", "torch", "timm", "status",
               "best_epoch", "best_val_macro_f1", "test_macro_f1", "train_seconds", "updated_at"]


def make_record(cfg: dict, status: str, repo_root: str = ".", **metrics) -> dict:
    snap = env_snapshot(repo_root)
    pk = snap["packages"]
    rec = {
        "run_id": make_run_id(cfg),
        "exp_name": cfg.get("exp_name"),
        "model": get_by_path(cfg, "model.name"),
        "family": get_by_path(cfg, "model.family"),
        "seed": get_by_path(cfg, "run.seed"),
        "config_hash": config_hash(cfg),
        "git_commit": snap["git_commit"],
        "platform": snap["platform"],
        "gpu": snap["gpu"].get("name", "cpu"),
        "python": snap["python"],
        "torch": pk.get("torch"),
        "timm": pk.get("timm"),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rec.update(metrics)
    return rec


def upsert_row(csv_path: str | Path, row: dict, key: str = "run_id") -> None:
    """Insert the row, or replace the existing row with the same run_id. New columns are added
    automatically. The file is rewritten atomically."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get(key) != str(row.get(key))]
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    fields = list(BASE_FIELDS)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)
