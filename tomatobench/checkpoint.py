"""Checkpoint / resume for Kaggle and Colab sessions that can die at any moment.

Design rules
  1. Atomic writes: a checkpoint is written to a temp file, then renamed. A crash mid-write
     never leaves a corrupt last.pt.
  2. last.pt  = full training state (model, optimizer, scheduler, scaler, RNG, early-stopping
     counters, history) saved EVERY epoch -> exact resume.
     best.pt  = best-on-VALIDATION weights. The test set is never used to pick checkpoints.
  3. DONE.json marks a finished run, so re-launching a 60-run grid skips finished runs.
  4. Test-set guard: mark_test_evaluated() can succeed only ONCE per run (anti-cherry-picking).
  5. Resume refuses to continue if the config hash changed (prevents silently mixing configs).
  6. Persistence across Kaggle sessions: files in /kaggle/working vanish when the session ends
     unless you 'Save Version'. On the next session attach that output as an input dataset and put
     its path in paths.persist_dirs; load_last() copies the newest checkpoint back automatically.

Security note: torch.load(weights_only=False) is used because we store RNG states. Only load
checkpoints that YOU created.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


class ConfigMismatchError(RuntimeError):
    pass


class TestSetAlreadyUsedError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- RNG state
def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
        except Exception:
            pass  # e.g. different GPU count after a session change; not fatal


# ----------------------------------------------------------------------------- atomic IO
def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    _fsync_path(tmp)
    os.replace(tmp, path)


def atomic_json_write(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- state helpers
def build_state(*, model, optimizer, scheduler=None, scaler=None, epoch: int,
                best_metric: float, best_epoch: int, bad_epochs: int,
                history: list, extra: Optional[dict] = None) -> dict:
    """Bundle everything needed to resume. `epoch` = last COMPLETED epoch (0-based)."""
    scaler_state = scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "history": history,
        "rng": capture_rng_state(),
        "extra": extra or {},
    }


def restore_state(state: dict, *, model, optimizer, scheduler=None, scaler=None) -> dict:
    """Load a state from build_state() into live objects. Returns the loop counters."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"])
    return {"start_epoch": state["epoch"] + 1, "best_metric": state["best_metric"],
            "best_epoch": state["best_epoch"], "bad_epochs": state["bad_epochs"],
            "history": state["history"], "extra": state.get("extra", {})}


# ----------------------------------------------------------------------------- manager
class CheckpointManager:
    def __init__(self, run_dir: str | Path, run_id: str, config_hash: str,
                 search_dirs: Iterable[str | Path] = ()):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config_hash = config_hash
        self.search_dirs = [Path(d) for d in search_dirs]

    # paths
    @property
    def last_path(self) -> Path: return self.run_dir / "last.pt"
    @property
    def best_path(self) -> Path: return self.run_dir / "best.pt"
    @property
    def meta_path(self) -> Path: return self.run_dir / "last.meta.json"
    @property
    def done_path(self) -> Path: return self.run_dir / "DONE.json"

    # saving
    def save_last(self, state: dict) -> None:
        state = dict(state)
        state["config_hash"] = self.config_hash
        state["run_id"] = self.run_id
        atomic_torch_save(state, self.last_path)
        atomic_json_write({"epoch": state["epoch"], "saved_at": time.time(),
                           "config_hash": self.config_hash}, self.meta_path)

    def save_best(self, state: dict) -> None:
        state = dict(state)
        state["config_hash"] = self.config_hash
        atomic_torch_save(state, self.best_path)

    # finding / importing from persisted storage
    @staticmethod
    def _epoch_in(dir_: Path) -> int:
        m = dir_ / "last.meta.json"
        if m.exists():
            try:
                return int(json.loads(m.read_text())["epoch"])
            except Exception:
                return -1
        return -1

    def _candidate_dirs(self):
        for d in self.search_dirs:
            for c in (d / self.run_id, d / "runs" / self.run_id):
                if c.is_dir():
                    yield c

    def import_from_persist(self) -> bool:
        """Copy the newest persisted copy of this run into run_dir if it is ahead of what we
        have locally. Returns True if something was imported."""
        local_epoch = self._epoch_in(self.run_dir)
        local_done = self.done_path.exists()
        best, best_key = None, (local_epoch, 1 if local_done else 0)
        for c in self._candidate_dirs():
            key = (self._epoch_in(c), 1 if (c / "DONE.json").exists() else 0)
            if key > best_key:  # further epoch, or same epoch but marked DONE
                best, best_key = c, key
        if best is None:
            return False
        shutil.copytree(best, self.run_dir, dirs_exist_ok=True)
        return True

    # loading
    def load_last(self, map_location="cpu", allow_hash_mismatch: bool = False) -> Optional[dict]:
        """Return the saved state (or None if there is nothing to resume)."""
        self.import_from_persist()
        if not self.last_path.exists():
            return None
        state = torch.load(self.last_path, map_location=map_location, weights_only=False)
        if state.get("config_hash") != self.config_hash and not allow_hash_mismatch:
            raise ConfigMismatchError(
                f"Checkpoint config hash {state.get('config_hash')} != current {self.config_hash}. "
                "The config changed since this run started. Use a new run, or pass "
                "allow_hash_mismatch=True if you KNOW the change is harmless.")
        return state

    def load_best(self, map_location="cpu") -> Optional[dict]:
        if not self.best_path.exists():
            return None
        return torch.load(self.best_path, map_location=map_location, weights_only=False)

    # run completion + test-set guard
    def is_done(self) -> bool:
        self.import_from_persist()
        return self.done_path.exists()

    def mark_done(self, best_metric: float, best_epoch: int, extra: Optional[dict] = None) -> None:
        rec = {"run_id": self.run_id, "config_hash": self.config_hash, "finished_at": time.time(),
               "best_val_metric": best_metric, "best_epoch": best_epoch,
               "test_evaluated": False, "test_metrics": None}
        rec.update(extra or {})
        atomic_json_write(rec, self.done_path)

    def mark_test_evaluated(self, test_metrics: dict) -> None:
        """Record the (single) test evaluation. Raises if the run was already tested."""
        if not self.done_path.exists():
            raise RuntimeError("Run is not finished (no DONE.json); do not touch the test set yet.")
        rec = json.loads(self.done_path.read_text())
        if rec.get("test_evaluated"):
            raise TestSetAlreadyUsedError(
                f"{self.run_id}: test set was already evaluated once. Re-evaluating would allow "
                "cherry-picking. Report the stored result.")
        rec["test_evaluated"] = True
        rec["test_metrics"] = test_metrics
        rec["test_evaluated_at"] = time.time()
        atomic_json_write(rec, self.done_path)

    def test_already_evaluated(self) -> bool:
        return self.done_path.exists() and bool(json.loads(self.done_path.read_text()).get("test_evaluated"))
