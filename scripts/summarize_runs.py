"""Read results/runs.csv and print summaries.

    python scripts/summarize_runs.py --exp lr_check      # learning-rate check + the pre-declared selection rule
    python scripts/summarize_runs.py --exp main_benchmark
    python scripts/summarize_runs.py --exp bg_test

Full tables (LaTeX, confidence intervals, significance tests) come later from scripts/make_report_tables.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tomatobench.env import default_roots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    csv = Path(args.csv) if args.csv else Path(default_roots(REPO)["work_root"]) / "results" / "runs.csv"
    df = pd.read_csv(csv)
    df = df[(df.exp_name == args.exp) & (df.status == "done")]
    if df.empty:
        sys.exit(f"no finished runs of experiment {args.exp} in {csv}")
    pd.set_option("display.width", 200)

    if args.exp == "lr_check":
        piv = df.pivot_table(index="lr", columns="model", values="best_val_macro_f1", aggfunc="mean")
        piv["MEAN"] = piv.mean(axis=1)
        print("best validation macro-F1 (seed 1000, 6 epochs):")
        print(piv.round(4).to_string())
        best = piv.MEAN.max()
        near = piv.index[piv.MEAN >= best - 0.002]          # rule fixed before the run: within 0.002 -> smaller LR
        chosen = min(near)
        print(f"\nrule: highest MEAN; LRs within 0.002 of it: {list(near)} -> choose the smaller one")
        print(f"==> CHOSEN LEARNING RATE: {chosen}")
        print("Now edit configs/base.yaml (train.lr) and commit it. After that the recipe is FROZEN.")
        return

    cols = ["best_val_macro_f1", "test_macro_f1", "test_accuracy", "test_balanced_accuracy", "test_binary_f1_diseased",
            "epochs_run", "train_seconds", "n_params_m", "model_mb"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("model")[cols]
    out = g.mean().round(4).astype(str) + " +/- " + g.std(ddof=1).fillna(0).round(4).astype(str)
    out.insert(0, "n_runs", g.size())
    print(out.to_string())
    print("\n(mean +/- std over seeds; no best-seed picking. Test numbers exist only for runs with eval.test=true.)")


if __name__ == "__main__":
    main()
