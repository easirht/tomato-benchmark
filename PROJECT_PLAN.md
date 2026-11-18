# PROJECT_PLAN.md  (read this first when resuming in a new session)

Last updated: Deliverable 3 essentially complete. split_v2 (v2.1 script: size-capped groups + exact purge, 40 folds, pre-purge 22/9/9) ran on real data in a Kaggle commit run (Version 2, 6 min 18 s) and gave 69.3/15.6/15.2 %. split_v2.csv is now on GitHub (splits/split_v2.csv, 3.5 MB, commit 5549287, uploaded via browser) and is FROZEN: never regenerate or edit it. (split_v2_summary was uploaded as .txt; it is JSON, optional rename.) Nothing below is a result; no experiment has been run.

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
| 3 | Dataset download, count check, cleaning, near-duplicate detection, group-aware split, EDA | DONE; split FROZEN on GitHub; dataset URL + licence recorded. Still open (small): optional PlantVillage leaf-map check, dataset datasheet. History: Step A DONE on real data (scripts/d3_01_build_manifest.py). Step B1 (scripts/d3_02_near_duplicates.py) DONE on real data (ResNet50 embeddings, Kaggle T4). Step B2 v2 (scripts/d3_03_groups_and_split.py: capped groups + StratifiedGroupKFold + exact purge + leakage report + EDA) written/tested on synthetic only; next: run v2 on real data |
| 4 | Baseline classification notebook + scripts | NEXT (GPU needed; start with a short validation-only learning-rate check, then freeze the recipe) |
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
5. DATASET SOURCE (observed on the Kaggle page, NOT yet verified by script): Kaggle 'PlantVillage Dataset'
   (2.18 GB; folders color / grayscale / segmented; we use `color` only). Tomato folder counts read from the
   page preview: bacterial spot 2127, early blight 1000, late blight 1909, leaf mold 952, septoria 1771,
   spider mites 1676, target spot 1404, YLCV 5357, mosaic 373, healthy 1591 -> 18,160 images, imbalance
   about 14.4x (YLCV vs mosaic). This DIFFERS from the brief's from-memory numbers (YLCV ~3209, total ~16,000,
   imbalance ~8.6x); the brief's numbers are superseded once the script confirms these. Dataset owner/URL
   and LICENCE: still to be recorded from the dataset page. Leaf IDs are NOT in this Kaggle version; the original
   PlantVillage GitHub repo reportedly has leaf_grouping metadata (VERIFY) -> may be used for the group-aware split.
   Old note: class-name mapping and counts are
   verified by script in Deliverable 3 (the counts in the original brief were from memory).


## 6b. Deliverable 3 findings so far (REAL data, step A; from the user's Kaggle run)
- Counts VERIFIED by script: 18,160 tomato images, 10 classes (bacterial 2127, early 1000, late 1909, leaf mold 952,
  septoria 1771, spider mites 1676, target spot 1404, YLCV 5357, mosaic 373, healthy 1591); imbalance 14.36x;
  0 unreadable; all 256x256 RGB. The brief's from-memory counts (~16,000; 8.6x) are superseded.
- Dataset has TWO identical copies of `color` (top-level and nested "plantvillage dataset/"); only the top-level
  /kaggle/input/plantvillage-dataset/color is used (file lists identical, 18,160 each).
- Exact duplicates: 14 MD5 groups (late blight 8, healthy 6), none across classes. DECISION (fixed now): keep the first
  file of each group, drop the other 14 -> 18,146 images; list in exact_dup_dropped.csv.
- Filenames: `<uuid>___<CODE> <number>.JPG`. 20 "source families" after removing the leaf/day suffix. Each family is
  almost entirely one class (family->class guess accuracy 0.9999). CAUTION: this is partly a naming artefact (codes
  contain disease abbreviations, e.g. Erly.B, Late.B, YLCV), so it is NOT evidence by itself that a model can
  shortcut on acquisition source. The real test is behavioural (background-only / source-held-out tests).
- Classes with >1 family (source-held-out analysis possible): bacterial spot (2), healthy (2), late blight (GHLB* and
  RS_Late.B), septoria (3), YLCV (3). Single-family classes: early blight, leaf mold, mosaic, spider mites, target spot.
  Source-held-out analysis = OPTIONAL, decide after the Deliverable 4 baseline (extra compute).
- Late-blight leaf codes like "GHLB_PS Leaf 23.4 Day" look like leaf-number + sub-image + day -> candidate leaf key
  (hypothesis, to be tested via image similarity). Other families only have a running photo number.
- Suspect images (kept, flagged, mention in Limitations; do not delete/relabel without expert confirmation):
  2 late-blight files with healthy-style code GH_HL (Leaf 173.2, Leaf 170); 3 healthy files with irregular names
  (CG1.JPG, a Flickr-style numeric name, GH_HL Leaf 380.1JPG.JPG); 4 late-blight files with a trailing " Day N .jpg".

### B1 results on REAL data (ResNet50 ImageNet embeddings, 18,146 images after dropping 14 exact duplicates)
- Top-1 nearest-neighbour cosine: median 0.896, p90 0.936, p99 0.960 (generic features: all PlantVillage leaves look alike).
- Unique top-10 pairs by cosine: >=0.99: 0; 0.97-0.99: 19; 0.95-0.97: 479 (1 cross-class); 0.92-0.95: 6,953 (45 cross-class);
  0.90-0.92: 15,029 (876 cross-class = 5.8%). Cross-class pairs are certainly different leaves, so this rate is a LOWER BOUND
  of the false-match rate. pHash<=4: 30 pairs, <=8: 331 pairs (among cos>=0.90 pairs).
- Visual review by the user (12 random pairs per bin, subjective, small sample; 12/12 only implies purity >= ~78% at 95% confidence):
  same leaf: pHash 0-4 12/12, pHash 5-8 12/12, cos 0.97-0.99 / 0.95-0.97 / 0.92-0.95 / 0.90-0.92 all 12/12;
  cos 0.88-0.90 4/12, 0.86-0.88 2/12, 0.84-0.86 0/12, 0.80-0.84 4/12, 0.70-0.80 3/12 (non-monotonic -> noisy judgement or
  same-leaf pairs with rotation/flip/crop that the CNN scores low; UNRESOLVED).
- Conclusion so far: PlantVillage-tomato contains many photos of the same leaf beyond exact duplicates. PROVISIONAL grouping
  threshold T=0.90 (cosine); final choice only after B2 diagnostics (watch for one giant chained group). Same-leaf pairs
  below 0.90 probably exist -> reported as RESIDUAL leakage (max-cosine-to-train, random vs group split) and as a limitation.
- Possible upgrade if residual leakage looks large: rotation/flip-aware matching (embed 8 dihedral variants) or DINOv2 features.


### FINAL D3 RESULT: split_v2 (REAL data, Kaggle commit Version 2; all numbers copied from the run log)
Parameters (fixed): embeddings = timm resnet50 (ImageNet) global-pool, L2-normalised; top-10 neighbour edges; T0 = 0.90;
component cap 100 (refined in steps of 0.005); StratifiedGroupKFold 40 folds, seed 1234, folds 22/9/9 before purge
(chosen after the first run showed the purge removes ~32% of val/test images); exact purge at cosine >= 0.90 over all pairs.
- Exact duplicates dropped: 14 -> 18,146 images. 141,773 unique top-10 neighbour pairs; 22,469 with cosine >= 0.90.
- Plain components (single linkage): T=0.97 largest 2; 0.95 largest 98; 0.92 largest 874 (12 mixed-class groups);
  0.90 largest 5,830 (13 mixed); 0.88 largest 10,877. -> chaining starts around 0.92, hence the cap.
- Refined groups (cap 100): 13,631 groups, largest 98, p99 size 7, 0 unsplittable; 5,830 images were re-split;
  severed edges 12,522 of 22,469 (55.7%), all handled by the purge. 49 groups still mix classes (label noise or similar-looking
  cross-class images; list them in Limitations / check later).
- Before purge 9,916/4,321/3,909 (54.6/23.8/21.5 %). Purge moved 2,657 images to train (val 1,499, test 1,158) in 6 rounds.
- FINAL: train 12,573 (69.3%), val 2,822 (15.6%), test 2,751 (15.2%). No group spans two splits. Verified: no val/test image has
  cosine >= 0.90 to any train image, and none of val to test.
- Per class train/val/test: bacterial 1721/205/201; early blight 579/213/208; late blight 1132/386/383; leaf mold 552/205/195;
  septoria 1074/366/331; spider mites 1106/293/277; target spot 919/249/236; YLCV 3927/714/716; mosaic 280/38/55; healthy 1283/153/149.
  NOTE: val/test class shares differ from train shares (purge removes images with near-copies; test is closer to class-balanced).
  Use macro-F1, class-weighted loss from TRAIN counts, and bootstrap CIs over test images for per-class metrics (mosaic val/test only 38/55).
- Leakage without any model (max cosine of each val/test image to ANY train image):
  random stratified split: val mean 0.890, >=0.90: 39.4%, >=0.88: 64.4%, >=0.86: 81.8%; test: 41.0%, 64.0%, 81.7%.
  group split + purge: val mean 0.867, >=0.90: 0.0%, >=0.88: 37.7%, >=0.86: 68.9%; test mean 0.867, 0.0%, 38.1%, 68.8%.
  Headline (state carefully, similarity = ResNet50 cosine, same-leaf validity only spot-checked by eye on 12 pairs/bin):
  a random split puts a >=0.90 near-copy of about 41% of test images into train.
  Residual risk: ~38% of test images still have a train neighbour in [0.88,0.90); visual review suggested some of these are the same
  leaf -> quantify after training with accuracy vs max_sim_train (column saved in split_v2.csv) and, optionally, a manual audit.
- Pipeline is deterministic enough: Step A and B1 numbers were identical in two independent runs.
- Limitations to write: similarity from generic ImageNet features; rotated/flipped/cropped same-leaf pairs may be missed; test set excludes
  images with near-copies in train (less typical than a random draw, but the right target for unseen-leaf generalisation); class proportions
  differ between train and val/test; PlantVillage has class-specific acquisition sources (family->class guess accuracy 0.9999, partly a
  naming artefact) so source shortcuts remain possible -> background-only test in Deliverable 4.

### B2 v1 on REAL data (abandoned) and the v2 design
- v1 = single-linkage components at cosine>=0.90: ONE group of 5,830 images (32% of data) containing all 10 classes
  (YLCV 2279, bacterial 1184, healthy 929, ...) = chaining through generic/hub images, NOT one leaf. Other groups are small
  (next largest 31, all one class). At T=0.86 the largest component is 14,608 images. Result: split 78.5/10.9/10.6 %
  instead of 70/15/15, class shares badly off. split_v1.csv is NOT to be used (do not commit it as the project split).
- v1 leakage numbers (useful, provisional): random stratified split -> 41.0% of test images (39.4% of val) have a train image
  with cosine>=0.90 (ResNet50 features); mean max-cosine 0.890. Group split v1: 0.0% by construction; 42.5% of test images
  still have max cosine in [0.88,0.90) -> residual same-leaf risk, to be audited.
- v2 rules (fixed BEFORE any model training): edges cosine>=T0=0.90; components larger than 100 images are re-split by raising
  the threshold in steps of 0.005 inside that component; severed edges counted; StratifiedGroupKFold 20 folds -> 14/3/3;
  then an exact PURGE over all pairs moves any val/test image with a >=0.90 near-copy in train (or any val image near a
  test image) to train until none remains (verified by assertion). Val/test end slightly below 15% (printed).
  Expected cost: images in chained regions end up in train. Report the final sizes and how many were moved.

### Dataset licence notes (all VERIFY; not legal advice; ask the supervisor/university before releasing anything)
- CONFIRMED from the Kaggle dataset page (screenshot, user, 18 Nov 2026): kaggle.com/datasets/abdallahalidev/plantvillage-dataset, owner
  "Abdallah Ali and 1 collaborator", updated 7 years ago, 2.18 GB, folders color/grayscale/segmented, page licence field = CC BY-NC-SA 4.0.
  The dataset description on that page is the abstract of Hughes & Salathe (2015). Cite the Kaggle copy AND the original papers.
- Other sources disagree: the original PlantVillage paper (Hughes & Salathe, arXiv:1511.08060) is summarised by a secondary site as CC BY-SA 3.0
  with an extra clause that models trained on the data fall under the same licence (read the paper's own text); a Zenodo copy says CC BY-SA 3.0 US;
  derived collections (Meta-Album, a Mendeley release) list CC0 for the "original data release". An uploader cannot relicense someone else's data.
- Working rule: comply with the strictest combination: attribute (Hughes & Salathe 2015; Mohanty et al. 2016), non-commercial research use only,
  share-alike for derivatives. Do NOT redistribute PlantVillage images in the repo (only split CSVs with file names and labels). Lesion masks derived
  from PlantVillage images and released model weights must be checked against the share-alike clause before release (Deliverable 10).
- PlantDoc, the second field dataset and the phone-image licence/consent: still to check.
- Still open (partly answered above): dataset owner/URL + LICENCE; output of the GitHub leaf_grouping / leaf-map.json check (cell 2).


## 6c. Where we stopped (resume here) + practical Kaggle/GitHub lessons
STATUS (older, superseded by FINAL D3 RESULT above): Deliverable 3 almost done. Real results of the FIRST v2 run (20 folds 14/3/3): refined groups + purge moved 1,643
images to train; final split 14,466 / 1,803 / 1,877 (79.7/9.9/10.3 %); 0 val/test images with cosine>=0.90 to train (verified by
assertion); random split baseline: 39-41% of val/test images have a >=0.90 train neighbour; group split still has 37-42% in
[0.88,0.90) -> residual leakage, to be quantified later with accuracy-vs-max_sim_train bins. mosaic_virus val/test only 33/29
images (small). PENDING RUN: v2.1 (scripts/d3_03_groups_and_split.py with --pre-folds 22 9 9, --folds 40, column max_sim_train)
via Kaggle "Save & Run All (Commit)"; first attempt failed ON PURPOSE (guard: the older script was still on GitHub, cost 19 s).
NEXT STEPS (old list; steps 1-3 are DONE, only step 4 onwards remains): (1) upload latest d3_03 script to GitHub scripts/ (verify with Ctrl+F "pre-folds" on GitHub); (2) run the commit
notebook (one %%bash cell: clone repo, pip install timm==1.0.11, d3_01, d3_02, d3_03 --out-name split_v2); (3) read FINAL split
sizes / class table / LEAKAGE table from the version log; (4) download results/d3/split_v2.csv and upload it to GitHub splits/
(the notebook cannot push to GitHub) -> FREEZE it as the project split; (5) record dataset owner/URL + licence, check the original
PlantVillage leaf_grouping/leaf-map.json (cell not yet run), decide about own phone images and the report template; then start
Deliverable 4 (baseline notebook; the background-only sanity test also belongs there because it needs a model).
KAGGLE FACTS LEARNED: GitHub repo easirht/tomato-benchmark (public). Notebook "PlantVillage Dataset" (copied from another user,
environment pinned 2025-11-07, input /kaggle/input/plantvillage-dataset/color - the dataset ALSO has a duplicate nested copy,
ignore it). Changing Accelerator or stopping a session wipes /kaggle/working. Set Accelerator BEFORE starting. CPU-only work
(B2, tables) can run with Accelerator None; GPU only for embeddings (d3_02) and training. Commit runs ("Save & Run All") keep
running after the tab is closed and stop by themselves; interactive sessions must be stopped with the power icon. Quota shown
in Session options (about 0:53 of 30 h used when last seen). Downloading big zips from the Output panel failed for the user:
keep outputs small (never zip embeddings.npy) and re-generate from scripts instead. GitHub uploads must go INTO the scripts/
folder (Add file > Upload files) and the file name must match exactly.
USER NOTES: explanations in Banglish; user is a beginner; give one step at a time with exact clicks; always say CPU vs GPU.

## 7. How to resume in a new session
Paste this file, then say which deliverable to continue. Run `python scripts/smoke_test.py` after
cloning to confirm the environment.
