"""Print what machine/software we are on. Run this first in every new Kaggle/Colab session.

    python scripts/check_env.py            # prints a summary
    python scripts/check_env.py --save     # also writes env_snapshot.json next to results
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tomatobench.env import env_snapshot, default_roots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    snap = env_snapshot(Path(__file__).resolve().parents[1])
    snap["roots"] = default_roots(Path(__file__).resolve().parents[1])
    print(json.dumps(snap, indent=2))
    if not snap["gpu"].get("cuda"):
        print("\n[WARNING] No GPU visible. On Kaggle: Settings > Accelerator > GPU T4/P100 "
              "(phone verification required). Training on CPU is far too slow.")
    if args.save:
        out = Path(snap["roots"]["work_root"]) / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "env_snapshot.json").write_text(json.dumps(snap, indent=2))
        print(f"\nSaved {out / 'env_snapshot.json'}")


if __name__ == "__main__":
    main()
