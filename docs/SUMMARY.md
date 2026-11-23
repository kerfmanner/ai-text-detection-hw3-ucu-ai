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
| 3.4 encoder | E1 | XLM-R base, 1 epoch. Each step trains on a random 512-token window per text. Prediction averages evenly spaced windows (cap 8 EN / 4 DE) and keeps per-window min/max/std plus the first and last window's probabilities (where fusion rewrites concentrate). The per-window table is saved in `features/E1_windows_*.parquet`. |
| 3.4 encoder / ensemble member | E2 | Token-level XLM-R base, 1 epoch. Each token is tagged original / rewritten / ai. Labels come from train fusion texts with a known human source (`splits/fusion_sources.csv`): a word is rewritten when none of its word 3-grams occurs in the source. Human and ai texts give all-original and all-ai tokens, and fusion texts without a source are left out (207,717 training texts, 158M tokens: 45% original, 3.5% rewritten, 51% ai). Fusion training windows contain a rewritten token 80% of the time, and the class weights are balanced^0.5 (0.86 / 3.1 / 0.81). Prediction covers the whole text with 510-token windows at stride 384. The stacker gets per-text summaries: class means, max of a 16-token moving average, share above 0.5, and the position of the rewritten peak. |
| 3.5 experiment | X1 | One TF-IDF model per language with its own search. German is also offered cased words and char 3–5-grams. |
| 3.5 experiment | X2 / X2g | Two-stage: D1 domain classifier → per-domain expert (B0 features/model), with fallback to the global model below 3,000 texts. X2g routes by the given `domain` column instead. |
| 3.5 experiment | X5 | Two-stage label cascade: stage 1 ai vs not-ai, stage 2 a human-vs-fusion specialist trained only on human and fusion texts (both with B0 features/model). Replaces domain routing as the two-stage idea, because test has the domain. |
| 3.5 experiment / ensemble member | X6 | Character n-grams (char_wb 3–5, cased, first 2,000 characters), C search, fit on 120k texts. Weaker alone, added for ensemble diversity. |
| 3.5 experiment / ensemble member | X8 | Masked word n-grams (text distortion): the 1,000 most frequent words of the fitted texts and punctuation are kept, every other word becomes `*` of the same length; word 1–3-gram TF-IDF + LinearSVC, fit on 120k texts. Style and syntax with almost no topic signal. |
| 3.5 experiment | S1 | Stacker trained on the train part: 5-fold group out-of-fold scores of B0, X6 and X8 over all 245k train texts, plus domain and text stats, per-language gradient boosting. Its val/test inputs are the members' usual full-train predictions. |
| 3.5 experiment | X7 | Two-stage sentence model. Train fusion texts are paired with their human source (TF-IDF neighbour, cosine ≥ 0.5). Each sentence is labelled *rewritten* (< 30% of its word 3-grams are in the source) or *original* (≥ 99%), and sentences of human and ai texts get their text's label. Stage 1 is a LinearSVC sentence classifier on B0 features (100k texts). Stage 2 is gradient boosting over per-text summaries of the sentence scores (mean/max/min/std, share flagged, rewritten score per third of the text), fitted on 60k texts from disjoint train groups. |
| 3.5 experiment | X3 | Qwen2.5-0.5B (observer) + Qwen2.5-0.5B-Instruct (performer). Features are token log-probs, entropy, cross-entropy and the Binoculars ratio, plus their spread over 64-token segments, fed to gradient boosting. |
| 3.5 experiment / final candidate | X4 | Two-stage stacked ensemble. Stage 1 is a set of diverse base models (B0, X1, X2g, X5, X6, X7, E1, X3). Stage 2 is a per-language gradient-boosting stacker over their outputs, the given domain, E1 window stats and text stats (raw LM features off; X3's scores carry them). It is scored out-of-fold with 5-fold group CV inside the validation split and refit on all validation rows for test. |
| 3.5 experiment | X4b | X4 plus per-language class offsets added to the stacker's log-probabilities, chosen by grid search for macro-F1. Offsets for each validation fold are tuned on the other folds, so the score stays out-of-sample. |
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

## Ensemble design results (CPU, full data, measured)

Validation macro-F1 of the single models:

| model | macro-F1 | human | ai | fusion |
|---|---|---|---|---|
| B0 | 0.8657 | 0.813 | 0.978 | 0.806 |
| X1 | 0.8693 | 0.819 | 0.980 | 0.810 |
| X2 / X2g | 0.8665 / 0.8665 | 0.817 | 0.978 | 0.804 |
| X5 cascade | 0.8833 | 0.827 | 0.988 | 0.836 |
| X6 char | 0.8080 | 0.730 | 0.978 | 0.716 |
| X3 LM features (GPU features, CPU classifier) | 0.6818 | 0.614 | 0.876 | 0.556 |
| X8 masked n-grams | 0.8667 | 0.811 | 0.982 | 0.807 |
| X7 sentence model | 0.9077 | 0.871 | 0.983 | 0.870 |
| S1 = stack of B0 + X6 + X8 on train OOF | 0.9098 | 0.863 | 0.993 | 0.873 |
| E1 XLM-R | **0.9304** | 0.888 | 0.993 | 0.911 |

- **X2g per domain:** experts help arXiv (0.849 → 0.879) and house_of_commons (0.873 → 0.887) but hurt gutenberg EN (0.737 → 0.691), so the overall score barely moves. Per-domain models are more useful as ensemble members than as replacements.
- **X5:** the best single CPU model, +0.018 over B0, and its dedicated stage 1 also raises ai F1.
- **X3 alone is weak:** 0.68. LM likelihood separates ai (median Binoculars 0.93 vs 1.00 for fusion and 1.01 for human) but barely separates human from fusion.
- **E1 is the strongest single model** by +0.047 over X5. It is best on most domains, for example cnn_news 0.977, student_essays 0.960 and blog_authorship 0.877.
  - **arXiv weakness:** E1 scores 0.628 on arXiv because it never predicts human there; all 41 human arXiv texts in validation become fusion. Their probabilities still differ (mean human probability 0.137 vs 0.081 for true fusion), so the stacker, which sees the given domain, fixes this.
  - **By window count:** macro-F1 is 0.93–0.96 for texts with 1–2 windows, 0.91 for 3–4 and 0.79 for 5 or more (1,007 texts). Long texts are where E1 is weakest.
  - **Window position:** taking the argmax of a single window scores 0.888 for the first window and 0.902 for the last, vs 0.930 for the mean and 0.937 for the max.
- **X7 is the best CPU model**, +0.024 over X5 and +0.042 over B0. It beats even the CPU-only stack (0.9050).
  - **Sentence labels:** only half of the fusion texts (42k of 83,682) have a human source among their neighbours at cosine ≥ 0.5. Their sentences are bimodal: about 77% are copied unchanged and 15–23% fully rewritten (word 3-gram share ≤ 0.1), with almost none in between. Stage 1 is fitted on 1.07M original, 83k rewritten and 1.13M ai sentences.
  - **Sentence level:** sentence F1 on validation pairs is original 0.917, ai 0.894 and rewritten only 0.305. Rewritten sentences are hard to recognise one by one.
  - **Text level:** the per-text summaries still separate human from fusion well (human 0.871, fusion 0.870 vs 0.827 / 0.836 for X5).
  - **By domain:** X7 is strongest where E1 is weak, for example arXiv 0.910 vs 0.628, gutenberg EN 0.939 vs 0.861 and euro_court_cases 0.945 vs 0.901.
  - **Error overlap with E1:** both are wrong on 1.9% of texts, the lowest of any pair.
- **X8 (masked n-grams)** matches B0 (0.8667 vs 0.8657) with no topic words. 1,000 kept words beat 300 (0.829 vs 0.812 on the 20k search). C = 1 is the edge of the grid, so a larger C may be slightly better. It is better than B0 on arXiv (0.889 vs 0.849) and house_of_commons (0.899 vs 0.873), and worse on gutenberg EN (0.651 vs 0.737). It is wrong on different texts than B0: both wrong on 6.7%, only one wrong on 9.4%.
- **S1 (train-part stacker)** reaches 0.9098 from just B0 + X6 + X8. The validation stacker over five TF-IDF members reaches 0.9025, so 245k stacking rows beat 27k. S1's score is fully out-of-sample, because it never sees validation rows. It costs 15 extra fits (about 20 min) and needs about 17 GB of RAM.
- **Stacker without E1:** per language, 5-fold group OOF inside validation, with the given domain.

  | stacker inputs | macro-F1 |
  |---|---|
  | B0 + X1 + X2g | 0.8913 |
  | + X5 | 0.8894 |
  | + X6 | 0.8991 |
  | + X5 + X6 | 0.8987 |
  | B0 + X1 + X2g + text stats | 0.8967 |
  | + X5 | 0.8970 |
  | + X6 | 0.9021 |
  | + X5 + X6 | **0.9025** |
  | same, with macro-F1 class offsets (X4b) | 0.9012 |

- **Adding X3** to B0 + X1 + X2g + X5 + X6 + text stats:

  | LM input to the stacker | X4 | X4b | human | ai | fusion |
  |---|---|---|---|---|---|
  | none | 0.9025 | 0.9012 | 0.854 | 0.991 | 0.862 |
  | X3 classifier scores only (3 columns) | **0.9050** | 0.9052 | 0.857 | 0.992 | 0.866 |
  | raw LM features only (17 columns) | 0.9049 | 0.9045 | 0.858 | 0.991 | 0.866 |
  | both (`ensemble.yaml` as written) | 0.9041 | 0.9038 | 0.855 | 0.992 | 0.865 |

  - The LM signal adds about +0.0025, mostly on human and fusion. The differences between the three ways of feeding it in (≤ 0.001) are below the noise of a 27k-row validation set.
  - The X3 scores alone are the simplest option: their classifier is fitted on 40k train texts, while raw features have to be learned by the stacker from validation rows.
  - X3 is wrong on texts the others get right (`only_one_wrong` 27%), but mostly because it is wrong often (32% error rate), so that number overstates its diversity.
- **Stacker with E1** (`ensemble.yaml`: all seven members, text stats, no raw LM features):

  | stacker members | X4 | X4b | human | ai | fusion |
  |---|---|---|---|---|---|
  | all: B0, X1, X2g, X5, X6, E1, X3 | **0.9596** | 0.9587 | 0.936 | 0.997 | 0.946 |
  | all, E1 without first/last-window columns | 0.9597 | 0.9595 | 0.936 | 0.997 | 0.946 |
  | drop B0 / X1 / X2g / X5 / X6 / X3 (one at a time) | 0.9593 / 0.9587 / 0.9595 / 0.9596 / 0.9592 / 0.9596 | | | | |
  | drop E1 | 0.9050 | 0.9052 | 0.857 | 0.992 | 0.866 |
  | E1 + X5 + X6 + X3 | 0.9592 | 0.9595 | | | |
  | E1 + X6 | 0.9594 | 0.9598 | | | |
  | E1 only (+ domain, window stats, text stats) | 0.9584 | 0.9578 | 0.935 | 0.996 | 0.944 |
  | all + X7 (current `ensemble.yaml`) | **0.9612** | 0.9608 | 0.939 | 0.997 | 0.948 |
  | all + X7, without E1 | 0.9243 | 0.9244 | 0.886 | 0.993 | 0.894 |
  | all + X7 + X8 | 0.9614 | 0.9615 | 0.939 | 0.997 | 0.948 |
  | all + X7 + S1 | 0.9610 | 0.9612 | 0.938 | 0.997 | 0.948 |
  | all + X7 + X8 + S1 | 0.9611 | 0.9607 | 0.938 | 0.997 | 0.948 |
  | all + X7, S1 instead of B0 and X6 | 0.9615 | 0.9613 | 0.939 | 0.998 | 0.948 |
  | all + X7 + X8 + S1, without E1 | 0.9292 | 0.9290 | 0.892 | 0.994 | 0.901 |

  - Most of the gain from E1 alone (0.930) to the ensemble (0.960) comes from stacking E1 itself: the given domain, the window statistics and per-language boundaries take it to 0.958. arXiv, for example, goes from 0.63 to 0.97.
  - The other six members add only +0.001 together. Each one-at-a-time drop changes the score by ≤ 0.001, which is noise on 27k rows. X1 has the largest drop (−0.0009).
  - E1 and the TF-IDF models are both wrong on only 2.2–3.1% of texts.
  - **X7** adds +0.0016 on top of E1, the only member that adds more than noise once E1 is in, mostly on human and fusion. Without E1 it adds +0.019 (0.9050 → 0.9243).
  - **X8 and S1** change the full ensemble by −0.0002 to +0.0003, which is noise, so they are not in `ensemble.yaml`. Without E1 they add +0.005 (0.9243 → 0.9292), so they are useful only for a CPU-only pipeline.
  - **First/last-window columns (idea from the fusion position finding):** they add nothing (0.9597 without them vs 0.9596 with them). The window max/min/std already carries this signal.
  - DE gutenberg has only 39 validation texts (1 human, 2 fusion), so its per-domain score (0.5 or 1.0) carries no information.
 The change is −0.0013 without X3 and ±0.0005 with it, which is noise. Without X3 the tuned offsets favour ai (DE +0.5, EN +1.0 at the grid edge); that only moves a few texts, because ai is already well separated. The stacker's balanced class weights already put the human/fusion boundary near the macro-F1 optimum.
- **Why X6 helps and X5 does not:** X6 (char) is the weakest model but adds the most. X5 is the strongest but adds nothing, because it uses B0's features and the stacker already re-weights B0 the same way.
- **Error overlap** on validation (`runs.error_overlap`):
  - The word-based pairs are both wrong on 7–9% of texts and only one wrong on 5–7%.
  - Pairs with X6 are both wrong on 5.5–6.4% and only one wrong on 14–16%.
  - Different mistakes matter more than individual strength.

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
| E2 token encoder (train on 208k texts + predict val/test over whole texts) | about 2–2.5 h (from E1's measured 2.06 it/s: 12,983 steps ≈ 1 h 45 min, plus 118k prediction windows ≈ 25 min) | 40–55 min | 20–30 min | as E1 |
| **total** | **5–6.5 h** | **1.5–2.5 h** | **1–1.5 h** | |

- Both GPU runs fit in single Kaggle sessions (12 h limit, 30 GPU-h per week). Pick a T4, not a P100, which has no fast fp16.
- E1, measured on a Colab T4 on 22 Nov 2026: 8,888 s (2 h 28 min) for training plus prediction, of which training is about 2 h 04 min (15,320 steps at 2.06 it/s, fp16, batch 16). Re-predicting val + test from the saved model to add the window table took about 24 min (106k windows). Measured GPU time for E1 + X3 in total: about 5 h 15 min on a T4.
- X3 full feature run, measured on a Colab T4 on 22 Nov 2026: 40k train + 27,234 val + 25,000 test texts in 8,555 s (2 h 23 min). The X3 classifier takes 6 s on CPU, and one X4 run 60–90 s and about 6.5 GB of RAM.
- X3 crop ablation, run on a Colab T4 on 22 Nov 2026 (`lm_crop_start.yaml` / `lm_crop_middle.yaml`; 10k train + 6k val + 6k test each; about 32 min per arm, 64 min in total). The X3 classifier was fitted on CPU for each crop on its train features.

  Validation macro-F1, start vs middle crop (6,000 texts, same ids; 43% are shorter than the budget, so both crops are identical for them):

  | subset | n | start | middle | Δ middle − start, 95% paired bootstrap interval |
  |---|---|---|---|---|
  | all | 6000 | 0.672 | 0.662 | −0.009 [−0.022, +0.003] |
  | long texts | 3415 | 0.691 | 0.678 | −0.013 [−0.033, +0.008] |
  | German, long | 806 | 0.720 | 0.722 | +0.002 [−0.030, +0.035] |
  | English, long | 2609 | 0.676 | 0.656 | −0.021 [−0.043, +0.002] |

  - In German, the middle crop makes the Binoculars score separate human from fusion slightly better: on long texts, ROC AUC 0.28 vs 0.32, where 0.5 means no separation. This matches the U-shaped fusion profile, but the classifier does not gain from it.
  - In English, the start crop is better.
  - Decision: keep `crop: start` in `lm_features.yaml`.
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
| X5 | 71 s (measured, full train) | 11.2 GB (measured) |
| X6 | ~2 min (measured: search on 20k + fit on 120k) | 11.2 GB (measured) |
| X7 | 18 min (measured: 2.27M training sentences, then 60k + val + test texts scored) | 10.9 GB (measured) |
| X8 | 2 min (measured: search on 20k + fit on 120k) | < 11 GB |
| S1 | 20 min (measured: 5 folds × B0, X6, X8 + stacker on 245k rows) | 16.8 GB peak (measured) |
| X3 classifier | 6 s (measured) | < 2 GB |
| X4 + X4b, all members | 60–90 s (measured) | about 7 GB peak (measured) |

## Verification status

- **Run at full scale:**
  - split construction and audit
  - B0 on the full data with the first grid
  - tokenizer fertility on 3,000 texts per language
  - E1 and X3 features on a Colab T4, including the CUDA fp16 branches
  - X5, X6, X7, X8, S1, the X3 classifier and X4/X4b with all members, on copies of the real outputs
- **Run as CPU smoke tests on small subsets:**
  - E1 training and inference
  - X3 feature extraction
  - B0, X1, X2/X2g, the X3 classifier and X4
  - the experiment log, report/compare/export helpers and EDA helpers
  - the submission writer and checks, on a full-size dummy prediction plus 4 deliberately broken variants: wrong order, uppercase labels, extra column, duplicate id
- **Not run:**
  - the notebooks themselves (as requested)
  - the bf16 branch (Ampere or newer GPUs); the T4 runs used fp16

## Points to decide or watch

- `test.csv` contains `domain`, so X4 uses the given domain (`domain_source: given`). D1 predicts it with 98.8% accuracy, and X2 and X2g both score 0.8665.
- The X4 validation score is out-of-fold within the validation split, while the other models are trained on train and scored on validation. Both use the same validation rows.
- B0/X1 hyperparameters are selected on the validation split they are reported on. This is standard model selection with slight optimism, the same for all models.
- The cloud setup cell installs `transformers==5.24.0`. If the platform's preinstalled torch is too old for it, delete that line and use the preinstalled transformers. Versions are recorded in every run's JSON.
- GitHub Pages deploys the whole repository. `data/` and `artifacts/` are ignored, but once you run and commit the notebooks, their outputs (the 01 examples and distinctive words) are public. Clear those outputs before pushing if no dataset text should be published.
- `submission.csv` is only written, not uploaded. The upload command is in the last cell of 07.
