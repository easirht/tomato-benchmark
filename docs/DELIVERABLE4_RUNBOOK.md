# Deliverable 4 runbook (baseline classification on Kaggle)

Order of experiments (each is one Kaggle commit run; EXP in notebooks/kaggle_commit_cell.txt):
1. `lr_check`  - 8 short runs (2 models x 4 learning rates, 6 epochs, seed 1000, VALIDATION ONLY).
   Then `python scripts/summarize_runs.py --exp lr_check` prints the chosen LR by the pre-declared rule.
   Edit `train.lr` in configs/base.yaml, commit to GitHub. The recipe is then FROZEN.
2. `bg_test`   - background-only sanity test (1 run, ResNet50, corners only).
3. `main_benchmark` - 6 models x 5 seeds = 30 runs, test set evaluated once per run.

Resuming: if a commit run stops before all runs are done (time budget), start the next commit run and
attach the previous version's Output as Input (Add Input > Notebook Output Files). run_queue.py finds
`runs/` folders under /kaggle/input by itself, skips DONE runs and resumes half-finished ones.

Never do: change the seed list; evaluate the test set outside run_queue; pick a "best seed";
edit configs/base.yaml after the LR is frozen (it changes the config hash, i.e. creates different runs).
