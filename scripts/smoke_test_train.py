"""CPU self-test for the Deliverable 4 code (synthetic images, tiny CNN, no GPU, no timm, ~1 minute).

    python scripts/smoke_test_train.py

Checks: config grids, training learns something, exact resume (interrupt + resume == uninterrupted),
test set evaluated once, eval.test=false never touches the test set, the queue runner with two shards.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tomatobench.checkpoint import CheckpointManager, TestSetAlreadyUsedError  # noqa: E402
from tomatobench.config import config_hash, get_by_path, iter_queue, load_config, make_run_id, run_dir_for  # noqa: E402
from tomatobench.train import train_one  # noqa: E402


def make_fake_data(tmp: Path, per_class=(30, 10, 10), size=64):
    rng = np.random.default_rng(0)
    cfg = load_config(REPO / "configs", platform="local", repo_root=REPO)
    names = cfg["data"]["class_names"]
    rows = []
    for c, name in enumerate(names):
        (tmp / "color" / f"Tomato___{name}").mkdir(parents=True, exist_ok=True)
        base = np.array([30 + 20 * c, 230 - 20 * c, 60 + 12 * (c % 5)])      # clearly separable colours
        for s, n in zip(("train", "val", "test"), per_class):
            for i in range(n):
                arr = np.clip(base + rng.normal(0, 25, (size, size, 3)), 0, 255).astype(np.uint8)
                rel = f"Tomato___{name}/{s}_{i}.JPG"
                Image.fromarray(arr).save(tmp / "color" / rel)
                rows.append({"image_id": len(rows), "rel_path": rel, "class_name": name, "label": c,
                             "binary_label": int(name != "healthy"), "split": s})
    (tmp / "splits").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp / "splits" / "split_fake.csv", index=False)


def overrides(tmp: Path, work: str, extra=()):
    return [f"paths.work_root={tmp / work}", f"paths.splits_dir={tmp / 'splits'}", f"data.image_root={tmp / 'color'}",
            "data.split_file=split_fake.csv", "data.mask_corner=8"] + list(extra)


class StopAfterFirstEpoch:                       # simulates a session that is about to die
    def should_stop(self, est): return True


def same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tb_train_"))
    ok = True
    make_fake_data(tmp)
    quiet = lambda *a, **k: None  # noqa: E731

    # ---- 1. run grids
    for exp, n_expected in (("main_benchmark", 30), ("lr_check", 8), ("bg_test", 1)):
        c = load_config(REPO / "configs", exp=exp, platform="local", repo_root=REPO)
        runs = list(iter_queue(c))
        assert len(runs) == n_expected and len({make_run_id(r) for r in runs}) == n_expected, exp
    print("[1] run grids (main 30, lr_check 8, bg_test 1, unique ids): PASS")

    # ---- 2. training learns, writes everything, test evaluated once
    ov = overrides(tmp, "A", ["run.seed=1", "train.early_stopping.patience=50", "train.epochs=12", "train.horizon_epochs=12"])
    cfgA = load_config(REPO / "configs", exp="smoke", overrides=ov, platform="local", repo_root=REPO)
    recA = train_one(cfgA, str(REPO), log=quiet)
    rd = run_dir_for(cfgA)
    files_ok = all((rd / f).exists() for f in ("best.pt", "DONE.json", "history.csv", "metrics_val.json",
                                               "metrics_test.json", "test_logits.npy", "final_record.json"))
    learned = recA["best_val_macro_f1"] > 0.6
    print(f"[2] trains + saves outputs (val macro-F1 {recA['best_val_macro_f1']:.3f} vs chance 0.1): "
          f"{'PASS' if files_ok and learned and not (rd / 'last.pt').exists() else 'FAIL'}")
    ok &= files_ok and learned
    mgr = CheckpointManager(rd, make_run_id(cfgA), config_hash(cfgA))
    try:
        mgr.mark_test_evaluated({"x": 1})
        print("[3] test-once guard: FAIL"); ok = False
    except TestSetAlreadyUsedError:
        print("[3] test-once guard (second test evaluation refused): PASS")
    again = train_one(cfgA, str(REPO), log=quiet)
    print(f"[4] finished run is skipped on relaunch: {'PASS' if again['status'] == 'already_done' else 'FAIL'}")
    ok &= again["status"] == "already_done"

    # ---- 5. interrupt + resume == uninterrupted
    cfgB = load_config(REPO / "configs", exp="smoke", overrides=overrides(tmp, "B", ["run.seed=1", "train.early_stopping.patience=50", "train.epochs=12", "train.horizon_epochs=12"]),
                       platform="local", repo_root=REPO)
    r1 = train_one(cfgB, str(REPO), budget=StopAfterFirstEpoch(), log=quiet)
    r2 = train_one(cfgB, str(REPO), log=quiet)
    wA = torch.load(run_dir_for(cfgA) / "best.pt", weights_only=False)["model"]
    wB = torch.load(run_dir_for(cfgB) / "best.pt", weights_only=False)["model"]
    lA, lB = np.load(run_dir_for(cfgA) / "val_logits.npy"), np.load(run_dir_for(cfgB) / "val_logits.npy")
    exact = same_weights(wA, wB) and np.allclose(lA, lB)
    print(f"[5] interrupted at epoch {r1.get('epochs_done')} then resumed == uninterrupted run: {'PASS' if exact else 'FAIL'}")
    ok &= r1["status"] == "interrupted" and exact

    # ---- 6. validation-only mode never touches the test set
    cfgC = load_config(REPO / "configs", exp="smoke", overrides=overrides(tmp, "C", ["run.seed=2", "eval.test=false"]),
                       platform="local", repo_root=REPO)
    recC = train_one(cfgC, str(REPO), log=quiet)
    rdC = run_dir_for(cfgC)
    no_test = (not (rdC / "metrics_test.json").exists()) and recC["test_macro_f1"] is None \
        and not json.loads((rdC / "DONE.json").read_text())["test_evaluated"]
    print(f"[6] eval.test=false leaves the test set untouched: {'PASS' if no_test else 'FAIL'}")
    ok &= no_test

    # ---- 7. background-only mask + queue runner with two shards
    cfgD = load_config(REPO / "configs", exp="smoke", overrides=overrides(tmp, "D", ["data.input_mask=corners", "run.seed=3"]),
                       platform="local", repo_root=REPO)
    recD = train_one(cfgD, str(REPO), log=quiet)
    print(f"[7] corners-only input runs (val macro-F1 {recD['best_val_macro_f1']:.3f}): PASS")
    base = overrides(tmp, "Q")
    flat = sum([["--set", o] for o in base], [])
    for sh in ("0/2", "1/2"):
        subprocess.run([sys.executable, str(REPO / "scripts" / "run_queue.py"), "--exp", "smoke", "--shard", sh] + flat,
                       check=True, capture_output=True)
    csv = pd.read_csv(tmp / "Q" / "results" / "runs.csv")
    q_ok = len(csv) == 2 and (csv.status == "done").all()
    print(f"[8] queue runner, 2 shards -> runs.csv has {len(csv)} done rows: {'PASS' if q_ok else 'FAIL'}")
    ok &= q_ok
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
