# Tomato Leaf Disease Benchmark (leakage-controlled)

Working title: *Robust Early Plant Disease Detection, Segmentation and Severity Estimation from
Leaf Images: A Leakage-Controlled Benchmark of CNNs and Vision Transformers under Image
Enhancement, Lighting Variation, Limited Data and Lab-to-Field Shift.*

Status: Deliverable 2 (project skeleton). No results exist yet. See `PROJECT_PLAN.md` for all
decisions and the current stage.

## Folder structure

```
tomato-benchmark/
  PROJECT_PLAN.md        decisions + status (read this first in a new session)
  README.md  requirements.txt  requirements-deploy.txt  .gitignore
  configs/
    base.yaml            shared defaults (seeds, models, protocol, training)
    platform/            kaggle.yaml  colab.yaml  local.yaml  (session limits etc.)
    exp/                 one small file per experiment (main_benchmark, smoke, ...)
  tomatobench/           importable code
    config.py            layered YAML configs, overrides, config hash, run IDs, run grid
    checkpoint.py        atomic checkpoints, resume, DONE marker, test-once guard
    runlog.py            results/runs.csv (one row per run)
    env.py  utils.py     platform detection, env snapshot, seeding, time budget
  scripts/               check_env.py  smoke_test.py   (more added per deliverable)
  tests/  notebooks/  docs/
  data/  splits/  runs/  results/  figures/     (data, runs, *.pt are git-ignored)
```

## Quick start (Kaggle)

1. Kaggle account: verify phone (needed for GPU + internet). Create a Notebook,
   Settings > Internet ON, Accelerator = GPU T4 (or P100).
2. In the first cell:
   ```
   !git clone https://github.com/<your-user>/tomato-benchmark.git
   %cd tomato-benchmark
   !pip install -q -r requirements.txt
   !python scripts/check_env.py
   !python scripts/smoke_test.py
   ```
   `smoke_test.py` must end with `ALL CHECKS PASSED`. It runs on CPU, so run it with GPU OFF to
   save GPU hours.
3. Use GPU ON only for training cells.

If `pip install` reports a dependency conflict on Kaggle, copy the error and send it to your
supervisor-assistant; do not force-reinstall torch.

## How resume works across sessions

- Every epoch, `last.pt` (full state) is written atomically to `<work_root>/runs/<run_id>/`.
- A finished run gets `DONE.json`; relaunching the whole grid skips finished runs.
- Kaggle deletes `/kaggle/working` when the session ends **unless you Save Version**. Before the
  session ends (or when `max_wall_hours` stops training for you):
  1. Click **Save Version > Save & Run All (Commit)** *or* download `runs/`.
  2. In the next session: Add Data > your previous notebook's *Output* (or a Dataset you made
     from it).
  3. Put its path in `configs/base.yaml` -> `paths.persist_dirs`, e.g.
     `["/kaggle/input/tomato-bench-run1"]`. Checkpoints are copied back automatically.
- Colab: mount Google Drive and set `paths.work_root` to a Drive folder (slower I/O, but persistent).
- Resume refuses to continue if the config changed (config hash mismatch).

## Rules enforced by the code

- Model selection uses validation macro-F1 only.
- `CheckpointManager.mark_test_evaluated()` works once per run (no repeated test peeking).
- Seeds are listed in `configs/base.yaml` and must not change after the first result.
- Every run writes a row to `results/runs.csv` with run_id, seed, config hash, git commit,
  library versions. All report tables are generated from that CSV only.

## Licence / data

Code licence: choose before release (MIT is typical). Dataset licences are listed in the dataset
datasheet (Deliverable 3). Do not commit images.
