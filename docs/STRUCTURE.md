# Structure and run instructions

## Layout

| path | content |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | pinned environment (uv, Python 3.12) |
| `configs/*.yaml` | one config per stage: seed, data sizes, grids, model settings, `smoke` overrides |
| `src/hw3/config.py` | paths (`data/`, `artifacts/`, Kaggle input mount), config loading, smoke switch, seeding |
| `src/hw3/data.py` | download, loading, the single label mapping (`LABELS`), the fixed split loader |
| `src/hw3/metrics.py` | the single metric implementation (`f1_score(average="macro")`), per-class table, per-group F1, confusion matrix |
| `src/hw3/runs.py` | saved predictions, run metadata (config, compute, runtime, status, error), reports, error overlap between models, experiment log, export zip |
| `src/hw3/split.py` | group detection, the fixed validation split, leakage audit |
| `src/hw3/textstats.py` | stylometric statistics (EDA and stacker features) |
| `src/hw3/eda.py` | EDA tables and figures |
| `src/hw3/tfidf.py` | B0 baseline (also used for X6 char n-grams), X1 per-language models, X2 domain routing and the D1 domain classifier, X5 label cascade, X8 masked n-grams (`keep_words` option) |
| `src/hw3/tokens.py` | E2 token-level XLM-R and E3 token-level mDeBERTa (original / rewritten / ai tags from fusion/source pairs), CLI `python -m hw3.tokens` (`--config tokens_mdeberta` for E3, `--repredict` re-runs inference from the saved model) |
| `src/hw3/encoder.py` | E1 chunked XLM-R fine-tuning and inference, CLI `python -m hw3.encoder` (`--repredict` re-runs inference from the saved model) |
| `src/hw3/lm.py` | X3 LM likelihood features (CLI `python -m hw3.lm`) and their classifier |
| `src/hw3/sentences.py` | X7 sentence-level model: sentence labels from fusion/source pairs, sentence classifier, per-text summaries |
| `src/hw3/ensemble.py` | X4 two-stage stacked ensemble, X4b class offsets, S1 stacker on train out-of-fold scores |
| `src/hw3/submission.py` | final model selection, submission writer, automated checks |
| `notebooks/01…07` | thin notebooks calling `src/hw3`, run in order |
| `splits/split.csv` | the fixed split (`id`, `group`, `split`), tracked in git so every machine uses the same one |
| `splits/fusion_sources.csv` | fusion text id → id of its human source (TF-IDF neighbour, cosine ≥ 0.5), ids only; used by X7 and E2, tracked in git so cloud runs do not recompute neighbours |
| `experiments/log.csv` | experiment log, rebuilt from `artifacts/runs/` by notebooks 06 and 07 |
| `images/` | figures written by the notebooks |
| `artifacts/` (git-ignored) | `preds/<id>_{val,test}.parquet`, `features/`, `runs/<id>.json`, `logs/`, `models/`; smoke runs write to `artifacts/smoke/` |
| `data/` (git-ignored) | competition files |

## Model ids

| id | what | produced by | compute |
|---|---|---|---|
| B0 | reference: word TF-IDF + linear model, grid search | 03 | CPU |
| E1 | fine-tuned XLM-R base, windowed long-text handling | 04 | GPU |
| E2 | token-level XLM-R: tags rewritten spans, per-text summaries | 04b | GPU |
| E3 | E2 with mDeBERTa-v3-base (bf16 only: L4 / A100), resumable from Drive checkpoints | 04c | GPU |
| X1 | per-language TF-IDF models with per-language search | 06 | CPU |
| D1 | stage-1 domain classifier (predictions only) | 06 (X2) | CPU |
| X2 / X2g | two-stage: domain routing to per-domain experts (predicted / given domain) | 06 | CPU |
| X5 | two-stage label cascade: ai vs not-ai, then human-vs-fusion specialist | 06 | CPU |
| X6 | character n-gram TF-IDF + linear SVM (diverse ensemble member) | 06 | CPU |
| X8 | masked word n-grams (text distortion) + linear SVM | 06 | CPU |
| S1 | stacker over B0, X6, X8 trained on train out-of-fold scores | 06 | CPU |
| X7 | two-stage sentence model: sentence labels from train fusion/source pairs → per-text summaries | 06 | CPU |
| X3_features / X3 | LM likelihood features / their gradient-boosting classifier | 05 / 06 | GPU / CPU |
| X4 | two-stage stacked ensemble | 07 | CPU |
| X4b | X4 with per-language class offsets tuned for macro-F1 | 07 | CPU |

## Run

```bash
uv sync
```

Open the notebooks with the `.venv` kernel and run them in order:

1. `01_eda` (its first cell downloads the data with the Kaggle CLI)
2. `02_split` (loads `splits/split.csv`; `REBUILD = True` recomputes it)
3. `03_baseline`
4. `04_encoder` (GPU)
   - `04b_token_encoder` (GPU)
   - `04c_token_mdeberta` (GPU, Colab L4 / A100, artifacts on Google Drive)
5. `05_lm_features` (GPU)
6. `06_experiments`
7. `07_ensemble_submission` (writes and checks `submission.csv`)

### GPU stages

- **Local GPU machine:** `uv sync`, then `uv run python -m hw3.encoder`, `uv run python -m hw3.tokens` and `uv run python -m hw3.lm`. Bring `artifacts/preds`, `artifacts/features`, `artifacts/runs` and `artifacts/logs` back to the machine that runs notebooks 06–07, or run everything there.
- **Kaggle:** push the repo first, including `splits/split.csv` and `splits/fusion_sources.csv`. Open `04_encoder` / `04b_token_encoder` / `05_lm_features`, add the competition as input data, and turn on internet and a GPU. Run all, then download `artifacts/E1_outputs.zip` / `artifacts/E2_outputs.zip` / `artifacts/X3_features_outputs.zip` from the output and unzip them into the local `artifacts/`.
- **Colab:** same as Kaggle, plus Kaggle credentials (`KAGGLE_API_TOKEN`, or `~/.kaggle/kaggle.json`) for the download. Set `HW3_ARTIFACTS` to a Drive folder to keep outputs if the session dies.
- **X3 crop ablation (optional, before the full X3 run):** `HW3_ARTIFACTS=$PWD/artifacts/lm_crop_start uv run python -m hw3.lm --config lm_crop_start`, then the same with `middle`; or the optional cell in `05_lm_features`. It compares reading the start vs the middle of texts longer than the token budget.
- **Colab with Drive (`04c_token_mdeberta`):** the setup cell mounts Google Drive and uses `DRIVE_ARTIFACTS` as `HW3_ARTIFACTS`. Training writes `checkpoints/E3.pt` every `checkpoint_every` micro-batches (`configs/tokens_mdeberta.yaml`) and deletes it at the end, so after a disconnect running all cells again resumes training, or only predicts if the model in `models/E3` is already saved. Copy `preds/E3_*`, `runs/E3*` and `logs/E3*` from Drive into the local `artifacts/`.
- **Smoke check (any CPU):** `uv run python -m hw3.encoder --smoke`, `uv run python -m hw3.tokens --smoke`, `uv run python -m hw3.lm --smoke`, or `SMOKE = True` in the notebook.

Environment variables: `HW3_DATA` (data folder) and `HW3_ARTIFACTS` (artifacts folder).
