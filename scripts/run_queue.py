"""Run an experiment's queue of training runs (resume-safe, shardable over the two Kaggle GPUs).

    python scripts/run_queue.py --exp lr_check                       # list is defined in configs/exp/lr_check.yaml
    python scripts/run_queue.py --exp main_benchmark --shard 0/2     # every 2nd run, starting at 0   (GPU 0)
    python scripts/run_queue.py --exp main_benchmark --shard 1/2     # every 2nd run, starting at 1   (GPU 1)
    python scripts/run_queue.py --exp main_benchmark --dry-run       # only list the runs and their status
    python scripts/run_queue.py --exp lr_check --set train.num_workers=2

Behaviour
  * Finished runs (DONE.json) are skipped; unfinished runs resume from last.pt.
  * Stops starting new runs when less than checkpoint.min_minutes_to_start_run is left of the session
    budget (checkpoint.max_wall_hours), and a running run stops gracefully after saving its checkpoint.
  * On Kaggle, outputs of earlier versions that are attached as Input are found automatically
    (see autodetect_persist_dirs) so that a new commit run continues where the last one stopped.
  * Writes results/runs.csv (rebuilt from every run folder's final_record.json).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tomatobench.config import iter_queue, load_config, make_run_id, run_dir_for  # noqa: E402
from tomatobench.env import detect_platform  # noqa: E402
from tomatobench.runlog import upsert_row  # noqa: E402
from tomatobench.utils import TimeBudget, human_time  # noqa: E402


def autodetect_persist_dirs(root="/kaggle/input", max_depth=5):
    """Directories under `root` that contain a `runs/` folder with finished or half-finished runs."""
    found = []
    root = Path(root)
    if not root.exists():
        return found
    base_depth = len(root.parts)
    for dp, dn, _ in os.walk(root):
        depth = len(Path(dp).parts) - base_depth
        dn[:] = [d for d in dn if d not in ("color", "grayscale", "segmented") and depth < max_depth]
        if Path(dp).name == "runs":
            if any((Path(dp) / r / f).exists() for r in os.listdir(dp) for f in ("DONE.json", "last.pt")):
                found.append(str(Path(dp).parent))
    return found


def rebuild_runs_csv(runs_dir: Path, csv_path: Path):
    n = 0
    for rec_file in sorted(runs_dir.glob("*/final_record.json")) + sorted(runs_dir.glob("*/failed_record.json")):
        upsert_row(csv_path, json.loads(rec_file.read_text()))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: run only every n-th run starting at i")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. train.lr=0.0001")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    cfg0 = load_config(REPO / "configs", exp=args.exp, overrides=args.set, repo_root=REPO)
    if detect_platform() == "kaggle":
        extra = [d for d in autodetect_persist_dirs() if d not in cfg0["paths"]["persist_dirs"]]
        cfg0["paths"]["persist_dirs"] = list(cfg0["paths"]["persist_dirs"]) + extra
        if extra:
            print("persisted run folders found (will resume from them):", extra)
    i, n = (int(x) for x in args.shard.split("/"))
    runs = list(iter_queue(cfg0))[i::n]
    if args.max_runs:
        runs = runs[:args.max_runs]
    runs_dir = Path(cfg0["paths"]["runs_dir"])
    csv_path = Path(cfg0["paths"]["results_dir"]) / cfg0["logging"]["results_csv"]
    print(f"experiment {args.exp}: {len(runs)} runs in this shard ({args.shard}); runs dir {runs_dir}")

    if args.dry_run:
        for c in runs:
            print(" ", make_run_id(c), "| lr", c["train"]["lr"], "| DONE" if (run_dir_for(c) / "DONE.json").exists() else "")
        return

    budget = TimeBudget(cfg0["checkpoint"]["max_wall_hours"], cfg0["checkpoint"]["safety_minutes"])
    min_start = float(cfg0["checkpoint"]["min_minutes_to_start_run"]) * 60
    from tomatobench.train import train_one            # imported late: needs torch

    failures, summary = 0, []
    for k, cfg in enumerate(runs):
        if budget.remaining() < min_start:
            print(f"\nless than {min_start / 60:.0f} min of session budget left: not starting run {k + 1}/{len(runs)}. "
                  "Start a new session (attach this version's output as Input) to continue.")
            break
        print(f"\n===== run {k + 1}/{len(runs)}: {make_run_id(cfg)}  (budget left {human_time(budget.remaining())}) =====")
        t0 = time.time()
        try:
            res = train_one(cfg, str(REPO), budget=budget)
            failures = 0
        except Exception as e:                           # noqa: BLE001 - record and go on
            traceback.print_exc()
            failures += 1
            rd = run_dir_for(cfg)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "failed_record.json").write_text(json.dumps(
                {"run_id": make_run_id(cfg), "exp_name": cfg["exp_name"], "model": cfg["model"]["name"],
                 "seed": cfg["run"]["seed"], "status": "failed", "error": repr(e)[:300]}))
            res = {"run_id": make_run_id(cfg), "status": "failed"}
            if failures >= 2:
                print("two failures in a row: stopping the queue (probably a systematic problem)")
                summary.append((res["run_id"], res["status"], time.time() - t0))
                break
        summary.append((res["run_id"], res["status"], time.time() - t0))
        if res["status"] == "interrupted":
            break

    n_rows = rebuild_runs_csv(runs_dir, csv_path)
    print(f"\n===== summary ({args.shard}) =====")
    for rid, st, sec in summary:
        print(f"  {st:14s} {human_time(sec):>10s}  {rid}")
    print(f"{n_rows} run records in {csv_path}")


if __name__ == "__main__":
    main()
