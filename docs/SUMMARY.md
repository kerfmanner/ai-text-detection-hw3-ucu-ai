# Work summary

Handover notes on what was built, what was measured while building it, and what it costs to run. This is raw material, not report text. Layout and run instructions are in [STRUCTURE.md](STRUCTURE.md); sources are in [CITATIONS.md](CITATIONS.md).

## What was added

- **Environment:** `pyproject.toml`, `uv.lock` and `.python-version` (uv, Python 3.12, exact pins), plus `.gitignore` for `data/`, `artifacts/`, `.venv/`.
- **Code:** `src/hw3/` (about 1,000 lines, no comments) and `configs/` (one YAML per stage, each with a `smoke` override).
- **Notebooks:** `notebooks/01…07`. 01 starts with the Kaggle download cell. 04 and 05 start with a Colab/Kaggle setup cell and have a `SMOKE` switch.
- **Split:** `splits/split.csv`, the fixed validation split, generated on this machine. Commit it so cloud runs use the same split.
- **Docs:** `STRUCTURE.md`, `CITATIONS.md`, this file.
- **Not touched:** `README.md`, `report.md`, `index.html`, `assets/`, `.github/`, `AGENTS.md`, `CLAUDE.md`, `docs/`.

## Pipeline

| stage | id | idea |
|---|---|---|
| 3.1 split | – | Groups come from rare-word TF-IDF cosine ≥ 0.5 between top-5 neighbours inside each language × domain block, plus shared 10-word openings used by ≤ 4 texts. Connected components form the groups; `StratifiedGroupKFold` holds out 10%. |
| 3.3 baseline / reference | B0 | Hashed word TF-IDF; grid over n-gram range × {LinearSVC C, ComplementNB α} on a 40k subsample, then refit on the whole train part. Frozen as the reference. |
| 3.4 encoder | E1 | XLM-R base, 1 epoch. Each step trains on a random 512-token window per text. Prediction averages evenly spaced windows (cap 8 EN / 4 DE) and keeps per-window min/max/std. |
| 3.5 experiment | X1 | One TF-IDF model per language with its own search. German is also offered cased words and char 3–5-grams. |
| 3.5 experiment | X2 / X2g | Two-stage: D1 domain classifier → per-domain expert (B0 features/model), with fallback to the global model below 3,000 texts. X2g routes by the given `domain` column instead. |
| 3.5 experiment | X3 | Qwen2.5-0.5B (observer) + Qwen2.5-0.5B-Instruct (performer). Features are token log-probs, entropy, cross-entropy and the Binoculars ratio, plus their spread over 64-token segments, fed to gradient boosting. |
| 3.5 experiment / final candidate | X4 | Two-stage stacked ensemble. Stage 1 is the D1 domain. Stage 2 is a per-language gradient-boosting stacker over all member outputs, E1 window stats, LM features and text stats. It is scored out-of-fold with 5-fold group CV inside the validation split and refit on all validation rows for test. |
| 3.6 final | – | Highest validation macro-F1 in `experiments/log.csv` → `submission.csv`, checked before writing and again after reading back from disk. |

Every run writes `artifacts/runs/<id>.json` (config, compute, versions, runtime, status, and the error if it failed) and predictions to `artifacts/preds/`. The experiment log is rebuilt from these files, so cloud runs only need their zip unpacked into `artifacts/`.

## Measurements made while building (CPU, this machine)

### Split (`splits/split.csv`)

- 272,352 texts form 179,119 groups: 119,586 singletons, and the largest group has 1,292 texts.
- Validation has 27,234 texts (10.0%). Language/domain shares match train to 3 decimals. Labels: ai 0.448, fusion 0.307, human 0.245 in both parts.
- Similarity, cnn_news example:
  - The nearest-neighbour cosine has a median of 0.83 for fusion and 0.38 for ai.
  - The gap between first and second neighbour has a median of 0.44 for human and 0.04 for ai. Human/fusion families are detectable; AI texts have no specific lexical source.
- Audit:
  - The p99 of a validation text's highest similarity to train is 0.48–0.49, and no top-5 neighbour reaches 0.5 or more.
  - 10-word openings shared with train: 13–15% of validation texts. They are concentrated in bundestag (34%), euro_court_cases (31%) and house_of_commons (22%), and are templates shared by a median of 12–48 train texts.
  - Rare openings still shared with train: 0.3% (ai), 1.7% (fusion), 1.3% (human).
- Limits:
  - AI texts cannot be tied to their source, so topic overlap can cross the split.
  - Pairs below the threshold or beyond the top 5 neighbours are missed.

### English vs German, and what was tuned per language

| observation | EN | DE | used in |
|---|---|---|---|
| domains | 7 | only bundestag (52,343) and gutenberg (428) | X1 per-language models, X4 `stack_by: lang` |
| median words ai / human / fusion | 481 / 439 / 389 | 430 / 584 / 584 | text stats in the per-language stacker |
| XLM-R tokens per word, share over 510 tokens, windows p99 | 1.41, 60%, 8 | 1.50, 84%, 4 | `encoder.yaml` `max_chunks` |
| Qwen tokens per word | 1.23 | 1.85 | `lm_features.yaml` `max_tokens` 512 / 768 (≈ 415 words each) |
| fusion vs its human source: median share of word 3-grams kept / length ratio (3,000 train pairs in 01) | 0.82 / 0.86 | 0.75 / 0.98 | segment features in X3, window stats in E1 |
| where fusion rewrites (3-grams kept by position in the text) | falls steadily from 0.86 at the start to 0.70 at the end | U-shaped: 0.88 at the start, about 0.69 through the middle, 0.77 at the end | segment features in X3, window stats in E1 |
| casing and compounding | – | nouns capitalised, compounds | X1 German grid: cased words, char n-grams |

### Other findings

- The test domain mix differs from train: gutenberg (both languages) is 7.4% of test vs 2.7% of train, euro_court 6.4% vs 10.1%, bundestag 14.3% vs 19.2%.
- Class priors per domain are very uneven; for example, gutenberg EN is 92% ai and arXiv 84% ai. Every classifier therefore uses `class_weight: balanced`.
- German length tails:
  - German fusion texts have almost no short tail (10th percentile 357 words vs 160 for human and 136 for ai).
  - The short German human texts are the ones without a fusion version: human texts with no near-duplicate have a median of 130 words, those with one 621. Only 16% of German human texts under 200 words have a near-duplicate, vs 86% overall.
  - So a short German text is rarely fusion. In the shortest validation decile (under 200 words), 13% are fusion.
- English: texts over about 1,000 words are rarely ai (90th percentile: ai 908, human 1,189, fusion 1,298).
- How much length adds (German only, a gradient-boosting model on out-of-fold word-TF-IDF SVC scores, trained on the train part and scored on validation):

  | model | macro-F1 |
  |---|---|
  | length alone | 0.507 |
  | SVC | 0.905 |
  | SVC scores stacked | 0.923 |
  | + length | 0.926 |
  | + all text stats | 0.929 |

  The extra value is captured by X4's `text_stats`.
- Digits: 39% (DE) and 52% (EN) of ai texts contain a digit, vs 82% and 76% of human texts. This enters X4 through `digit_ratio`.
- No text contains a newline, so there is no line-break feature.

### B0 verification run on full data

This used the first grid, C ∈ {0.1, 0.3, 1.0}. The best point was at the grid edge, so C = 3.0 was added afterwards and the outputs were deleted; notebook 03 recomputes them.

- **Best configuration:** word 1–2-grams + LinearSVC with C = 1.0, balanced.
- **Validation macro-F1:** 0.8697 (human 0.821, ai 0.981, fusion 0.808).
- **By domain:** lowest is blogs at 0.737, highest is student essays at 0.914. German gutenberg scores 0.50 on only 39 texts.
- **Cost:** 112 s wall time, 11.6 GB peak RAM.
- **Char n-grams:** at 40k texts they added only +0.0014 macro-F1 for 2.4× the memory, so B0 uses words only.

## Compute

### GPU (estimates)

Basis:

- E1 ≈ 6 × 86M non-embedding params × 125M training tokens ≈ 7·10¹⁶ FLOPs, plus inference over about 115k windows.
- X3 ≈ 2 models × 2 × 0.49B params × about 45M tokens ≈ 9·10¹⁶ FLOPs.
- Sustained throughput assumed: about 15 TFLOP/s on a T4 (fp16) and about 100 TFLOP/s on an A100 (bf16).

| run | T4 16 GB (Kaggle / Colab) | RTX 3090/4090 24 GB | A100 | GPU memory |
|---|---|---|---|---|
| E1 encoder (train on 245k texts + predict val/test) | 1.5–2 h | 35–50 min | 20–30 min | about 11–13 GB at batch 16 |
| X3 LM features (40k train + 27k val + 25k test texts) | 1.5–2 h | 30–45 min | 15–20 min | about 8 GB at batch 8 |
| **total** | **3–4 h** | **1–1.5 h** | **35–50 min** | |

- Both GPU runs fit in single Kaggle sessions (12 h limit, 30 GPU-h per week). Pick a T4, not a P100, which has no fast fp16.
- Optional X3 crop ablation (`lm_crop_start.yaml` / `lm_crop_middle.yaml`, 10k train + 6k val + 6k test each): about 45–60 min in total on a T4, 15–25 min on a 3090/4090. On CPU it measured 4–6 s per long text per crop, so it was not run here.
- Faster options: `train_docs: 120000` in `encoder.yaml` roughly halves E1; `train_docs: 20000` in `lm_features.yaml` cuts X3 by about 20%.
- If E1 runs out of memory: `batch_size: 8`, `grad_accum: 2`.
- If LM features contain NaN under fp16: `dtype: float32` in `lm_features.yaml` (about 2–3× slower on a T4).
- Disk: the E1 model takes about 1.1 GB (`save_model: true`), and the Hugging Face cache about 3 GB.

### CPU (22 cores, 30 GB)

| stage | time | peak RAM |
|---|---|---|
| split | ~6 min (measured, mostly the neighbour search, cached in `artifacts/split/neighbors.parquet`) | ~5 GB |
| EDA | ~5–10 min (estimated) | ~6 GB |
| B0 | ~2–3 min (measured 112 s with 10 grid points; now 12) | 11.6 GB (measured) |
| X1, X2 | ~5–10 min each (estimated; X2 also fits the 9-class domain router) | ~12 GB |
| X3 classifier, X4 | a few minutes (estimated) | < 6 GB |

## Verification status

- **Run at full scale:**
  - split construction and audit
  - B0 on the full data with the first grid
  - tokenizer fertility on 3,000 texts per language
- **Run as CPU smoke tests on small subsets:**
  - E1 training and inference
  - X3 feature extraction
  - B0, X1, X2/X2g, the X3 classifier and X4
  - the experiment log, report/compare/export helpers and EDA helpers
  - the submission writer and checks, on a full-size dummy prediction plus 4 deliberately broken variants: wrong order, uppercase labels, extra column, duplicate id
- **Not run:**
  - the notebooks themselves (as requested)
  - the CUDA branches (fp16/bf16 autocast, GradScaler), since there is no GPU here

## Points to decide or watch

- `test.csv` contains `domain`. Stage-1 domain prediction is therefore optional; X2 vs X2g shows what predicted routing costs. If X2g beats X2, set `domain_source: given` in `ensemble.yaml`.
- The X4 validation score is out-of-fold within the validation split, while the other models are trained on train and scored on validation. Both use the same validation rows.
- B0/X1 hyperparameters are selected on the validation split they are reported on. This is standard model selection with slight optimism, the same for all models.
- The cloud setup cell installs `transformers==5.24.0`. If the platform's preinstalled torch is too old for it, delete that line and use the preinstalled transformers. Versions are recorded in every run's JSON.
- GitHub Pages deploys the whole repository. `data/` and `artifacts/` are ignored, but once you run and commit the notebooks, their outputs (the 01 examples and distinctive words) are public. Clear those outputs before pushing if no dataset text should be published.
- `submission.csv` is only written, not uploaded. The upload command is in the last cell of 07.
