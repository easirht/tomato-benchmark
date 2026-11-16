# PROJECT_PLAN.md  (read this first when resuming in a new session)

Last updated: Deliverable 2. Nothing below is a result; no experiment has been run.

## 1. Project
Tomato-only, leakage-controlled benchmark of CNN / ViT / hybrid models for leaf disease
analysis: (T1) healthy vs diseased (derived label), (T2) 10-class classification,
(T3) lesion segmentation, (T4) severity (% infected area -> healthy/low/moderate/severe).
Researcher: beginner-to-intermediate in PyTorch. Explanations to the user in Banglish; code,
paper and report text in English. User is the author; AI-assisted writing must be disclosed.

## 2. Constraints and agreed scope
- Compute: Kaggle GPU (T4/P100, ~30 h/week). No local GPU (laptop is CPU only). Colab as backup.
- Deadline: about 1 month. Realistic output = complete project report + strong, honest core
  study; journal paper (IEEE Access / conference first, Q1 later) is an extension.
- Report template / page limit: NOT YET GIVEN (assume "none" until the user says otherwise).
- Own phone images: UNDECIDED (user has not confirmed). If not collected, RQ5 uses PlantDoc +
  one field dataset only and the "released phone test set" contribution is dropped.

### 1-month scope (agreed)
| Item | Plan |
|---|---|
| RQ1 models | ResNet50, EfficientNet-B0, ConvNeXt-Tiny, DeiT-Small, Swin-Tiny, MobileViT (6) |
| Seeds | 5 fixed seeds: 42-46 (reserve 47-51 only if compute allows; decided now) |
| RQ2 enhancement | top 1-2 models, 3 seeds, CLAHE / gamma / gray-world / denoise / combined |
| RQ3 robustness | inference-only on trained models (corruptions x 3-5 severities) |
| RQ4 low-data | 1/10/100% (maybe 5/25/50 if time), 1-2 models, 3 seeds, scratch vs fine-tune vs linear probe |
| RQ5 lab-to-field | train PlantVillage only; test PlantDoc + 1 field set (+ phone set if collected); NEVER train on field data |
| RQ6 resolution | optional, 1 model |
| Segmentation + severity | U-Net (ResNet enc.), maybe SegFormer-B0; 100-150 manual masks, rest pseudo; severity thresholds fixed BEFORE test |
| Multitask, self-supervised | future work |

## 3. Protocol rules (do not break)
- Split 70/15/15, stratified, group-aware (near-duplicate groups), ONE fixed split (seed 1234).
- Identical training recipe across models: AdamW, cosine + warmup, AMP, label smoothing, class-weighted
  loss or focal; max 100 epochs, early stopping patience 10 on VALIDATION macro-F1.
- Main metric: macro-F1. Also accuracy, balanced accuracy, precision, recall, per-class F1,
  ROC-AUC, confusion matrix.
- Anti-cherry-picking: all seeds reported (mean +/- std, 95% CI); seed list never changed after
  results; checkpoint choice on validation only; test set evaluated once per final run (enforced
  in code); deployment model chosen by validation or median seed.
- Statistics: paired tests (McNemar / Wilcoxon / paired t), Holm correction, effect sizes.
- Efficiency: params, FLOPs, size, GPU/CPU latency.
- Never invent numbers or citations. Every number in the report comes from a results CSV via
  scripts/make_report_tables.py (Deliverable 7). Missing result -> "[RESULT PENDING: ...]".
- Report "too good" results (e.g. 99.9% on PlantVillage) honestly and test for leakage /
  background bias (background-only sanity test).

## 4. Literature status (Deliverable 1, closed)
- 36 papers verified by the user in a Word table (all marked VERIFY until final bibliography).
- Dropped (not found): Too et al. 2019 (fine-tuning comparison). Ferentinos 2018 not added.
- Closest prior work found (read these in full): Sudir et al. Research Square preprint
  (leakage-audited cross-dataset evaluation, apple + maize, single seed per model, McNemar + Holm);
  Noyan 2022 (PlantVillage background bias, arXiv 2206.04374); Ahmad et al. IEEE Access 2023
  (corn, controlled vs field); AgriPath-LF16 (lab vs field benchmark, 2026 arXiv).
- Gap (provisional): no paper found (by this search; absence is not proof) that does tomato-only
  multi-seed benchmark + enhancement + corruption + low-data + lab-to-field + released phone
  test set + masks.
- Working novelty statement (final wording depends on real results and on the user's own
  Scholar check): "To the best of our knowledge, this work provides a multi-seed,
  leakage-controlled benchmark of CNN, Vision Transformer and hybrid models for tomato leaf
  disease analysis, jointly testing image enhancement, controlled corruption robustness, reduced
  training data and lab-to-field transfer, and releases a multi-lighting smartphone test set with
  lesion masks." Do not use the word "novel" without literature support.
- If results are unremarkable: main contribution = released test set + masks + rigorous protocol.

## 5. Deliverable status
| # | Deliverable | Status |
|---|---|---|
| 1 | Literature gap, novelty, formulation | done |
| 2 | Folder structure, env, YAML config, checkpoint/resume, this file | done (code tested on CPU only) |
| 3 | Dataset download, count check, cleaning, near-duplicate detection, group-aware split, EDA | next |
| 4 | Baseline classification notebook + scripts | pending |
| 5 | Segmentation + severity (annotation workflow) | pending |
| 6 | Enhancement / robustness / low-data / resolution / lab-to-field configs + GPU estimates | pending |
| 7 | Aggregation: LaTeX tables, plots, stats | pending |
| 8 | Demo (Gradio/Streamlit, ONNX optional) | pending |
| 9 | Early report sections (no results needed) | pending |
| 10 | Journal list, checklist, reproducibility package | pending |
| 11-12 | Final report + generation rules | pending (after all experiments) |

## 6. Open decisions / assumptions to confirm
1. Report template / page limit (assumed none).
2. Will phone images be collected? (100-200 images suggested).
3. GitHub repo created? (needed for Kaggle clone and reproducibility package).
4. Placeholder hyper-parameters in configs/base.yaml (lr 2e-4, wd 0.05, bs 32) must be fixed in
   Deliverable 4 using VALIDATION data only, then frozen before the full benchmark.
5. PlantVillage version/folder names on Kaggle are unverified; class-name mapping and counts are
   verified by script in Deliverable 3 (the counts in the original brief were from memory).

## 7. How to resume in a new session
Paste this file, then say which deliverable to continue. Run `python scripts/smoke_test.py` after
cloning to confirm the environment.
