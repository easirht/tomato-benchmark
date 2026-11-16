"""Self-test for config + checkpoint/resume + run logging. Runs on CPU in seconds.

    python scripts/smoke_test.py

The key test simulates a Kaggle/Colab crash: train 6 epochs straight vs. train 3 epochs, 'die',
move the files to a fresh working dir (like attaching last session's output as an input dataset)
and resume. The final weights must be IDENTICAL. If this prints PASS, resume logic works.
"""
import copy
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tomatobench.checkpoint import (CheckpointManager, ConfigMismatchError,  # noqa: E402
                                    TestSetAlreadyUsedError, build_state, restore_state)
from tomatobench.config import (apply_overrides, config_hash, iter_runs,  # noqa: E402
                                load_config, make_run_id)
from tomatobench.runlog import make_record, upsert_row  # noqa: E402
from tomatobench.utils import TimeBudget, make_generator, set_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def make_data(seed=0, n=256, d=16, k=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, k, generator=g)
    y = (x @ w).argmax(1)
    return x[:192], y[:192], x[192:], y[192:]


def train(run_dir, run_id, cfg_hash, total_epochs, seed=0, stop_after=None, search_dirs=()):
    """Tiny training loop with exactly the structure the real trainer will have."""
    set_seed(seed)
    xtr, ytr, xva, yva = make_data(seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    mgr = CheckpointManager(run_dir, run_id, cfg_hash, search_dirs=search_dirs)

    if mgr.is_done():
        return "already_done", model

    start, best, best_epoch, bad, history = 0, float("-inf"), -1, 0, []
    state = mgr.load_last()
    if state is not None:
        c = restore_state(state, model=model, optimizer=opt, scheduler=sched)
        start, best, best_epoch, bad, history = (c["start_epoch"], c["best_metric"],
                                                 c["best_epoch"], c["bad_epochs"], c["history"])
        print(f"  resumed from epoch {start}")

    for epoch in range(start, total_epochs):
        gen = make_generator(seed * 1000 + epoch)          # epoch-level seed -> resume-safe shuffle
        perm = torch.randperm(len(xtr), generator=gen)
        model.train()
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(xva).argmax(1) == yva).float().mean().item()
        history.append({"epoch": epoch, "val_acc": val_acc})
        improved = val_acc > best
        if improved:
            best, best_epoch, bad = val_acc, epoch, 0
        else:
            bad += 1
        st = build_state(model=model, optimizer=opt, scheduler=sched, epoch=epoch, best_metric=best,
                         best_epoch=best_epoch, bad_epochs=bad, history=history)
        mgr.save_last(st)
        if improved:
            mgr.save_best(st)
        if stop_after is not None and epoch + 1 >= stop_after:
            return "interrupted", model                     # simulate a dead session
    mgr.mark_done(best, best_epoch)
    return "done", model


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tb_smoke_"))
    ok = True
    try:
        # ---- 1. config system
        cfg = load_config(REPO / "configs", exp="smoke", platform="local", repo_root=REPO)
        h = config_hash(cfg)
        cfg2 = apply_overrides(cfg, ["run.seed=99", "paths.work_root=/somewhere/else"])
        assert config_hash(cfg2) == h, "seed/paths must not change the config hash"
        cfg3 = apply_overrides(cfg, ["train.lr=0.1"])
        assert config_hash(cfg3) != h, "a real setting must change the config hash"
        full = load_config(REPO / "configs", exp="main_benchmark", platform="local", repo_root=REPO)
        grid = list(iter_runs(full))
        assert len(grid) == 6 * 5, f"expected 30 runs, got {len(grid)}"
        assert len({make_run_id(c) for c in grid}) == 30, "run IDs must be unique"
        print(f"[1] config OK  (hash {h}, grid = {len(grid)} runs, e.g. {make_run_id(grid[0])})")

        # ---- 2. resume gives identical weights
        rid = "smoke__demo__s0__x"
        status, m_full = train(tmp / "A" / rid, rid, "h1", 6)
        status1, _ = train(tmp / "B" / rid, rid, "h1", 6, stop_after=3)
        assert status1 == "interrupted"
        # new session: empty work dir, previous output attached as persisted dir
        shutil.copytree(tmp / "B", tmp / "persist")
        shutil.rmtree(tmp / "B")
        status2, m_res = train(tmp / "C" / rid, rid, "h1", 6, search_dirs=[tmp / "persist"])
        assert status2 == "done"
        res = same_weights(m_full, m_res)
        print(f"[2] resume reproduces uninterrupted training exactly: {'PASS' if res else 'FAIL'}")
        ok &= res

        # ---- 3. finished runs are skipped
        again, _ = train(tmp / "C" / rid, rid, "h1", 6)
        assert again == "already_done"
        print("[3] finished run is skipped on relaunch: PASS")

        # ---- 4. config change blocks resume
        train(tmp / "D" / rid, rid, "h1", 6, stop_after=2)
        try:
            CheckpointManager(tmp / "D" / rid, rid, "OTHER_HASH").load_last()
            print("[4] config-mismatch guard: FAIL"); ok = False
        except ConfigMismatchError:
            print("[4] config-mismatch guard: PASS")

        # ---- 5. test set can be evaluated only once
        mgr = CheckpointManager(tmp / "C" / rid, rid, "h1")
        mgr.mark_test_evaluated({"macro_f1": 0.5})
        try:
            mgr.mark_test_evaluated({"macro_f1": 0.6})
            print("[5] test-once guard: FAIL"); ok = False
        except TestSetAlreadyUsedError:
            print("[5] test-once guard: PASS")

        # ---- 6. run log CSV (upsert)
        csv_path = tmp / "results" / "runs.csv"
        upsert_row(csv_path, make_record(cfg, "running", REPO))
        upsert_row(csv_path, make_record(cfg, "done", REPO, best_val_macro_f1=0.9))
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 2, "same run_id must be updated, not duplicated"
        print("[6] runs.csv upsert: PASS")

        # ---- 7. time budget
        b = TimeBudget(max_hours=0.0001, safety_minutes=0)
        assert b.should_stop(0) or b.remaining() < 1
        print("[7] time budget: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
