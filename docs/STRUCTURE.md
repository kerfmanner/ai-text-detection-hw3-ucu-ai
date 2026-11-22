# Structure and run instructions

## Layout

| path | content |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | pinned environment (uv, Python 3.12) |
| `configs/*.yaml` | one config per stage: seed, data sizes, grids, model settings, `smoke` overrides |
| `src/hw3/config.py` | paths (`data/`, `artifacts/`, Kaggle input mount), config loading, smoke switch, seeding |
| `src/hw3/data.py` | download, loading, the single label mapping (`LABELS`), the fixed split loader |
| `src/hw3/metrics.py` | the single metric implementation (`f1_score(average="macro")`), per-class table, per-group F1, confusion matrix |
| `src/hw3/runs.py` | saved predictions, run metadata (config, compute, runtime, status, error), reports, experiment log, export zip |
| `src/hw3/split.py` | group detection, the fixed validation split, leakage audit |
| `src/hw3/textstats.py` | stylometric statistics (EDA and stacker features) |
| `src/hw3/eda.py` | EDA tables and figures |
| `src/hw3/tfidf.py` | B0 baseline, X1 per-language models, X2 two-stage routing and the D1 domain classifier |
| `src/hw3/encoder.py` | E1 chunked XLM-R fine-tuning and inference, CLI `python -m hw3.encoder` |
| `src/hw3/lm.py` | X3 LM likelihood features (CLI `python -m hw3.lm`) and their classifier |
| `src/hw3/ensemble.py` | X4 two-stage stacked ensemble |
| `src/hw3/submission.py` | final model selection, submission writer, automated checks |
| `notebooks/01…07` | thin notebooks calling `src/hw3`, run in order |
| `splits/split.csv` | the fixed split (`id`, `group`, `split`), tracked in git so every machine uses the same one |
| `experiments/log.csv` | experiment log, rebuilt from `artifacts/runs/` by notebooks 06 and 07 |
| `images/` | figures written by the notebooks |
| `artifacts/` (git-ignored) | `preds/<id>_{val,test}.parquet`, `features/`, `runs/<id>.json`, `logs/`, `models/`; smoke runs write to `artifacts/smoke/` |
| `data/` (git-ignored) | competition files |

## Model ids

| id | what | produced by | compute |
|---|---|---|---|
| B0 | reference: word TF-IDF + linear model, grid search | 03 | CPU |
| E1 | fine-tuned XLM-R base, windowed long-text handling | 04 | GPU |
| X1 | per-language TF-IDF models with per-language search | 06 | CPU |
| D1 | stage-1 domain classifier (predictions only) | 06 (X2) | CPU |
| X2 / X2g | two-stage: domain routing to per-domain experts (predicted / given domain) | 06 | CPU |
| X3_features / X3 | LM likelihood features / their gradient-boosting classifier | 05 / 06 | GPU / CPU |
| X4 | two-stage stacked ensemble | 07 | CPU |

## Run

```bash
uv sync
```

Open the notebooks with the `.venv` kernel and run them in order:

1. `01_eda` (its first cell downloads the data with the Kaggle CLI)
2. `02_split` (loads `splits/split.csv`; `REBUILD = True` recomputes it)
3. `03_baseline`
4. `04_encoder` (GPU)
5. `05_lm_features` (GPU)
6. `06_experiments`
7. `07_ensemble_submission` (writes and checks `submission.csv`)

### GPU stages

- **Local GPU machine:** `uv sync`, then `uv run python -m hw3.encoder` and `uv run python -m hw3.lm`. Bring `artifacts/preds`, `artifacts/features`, `artifacts/runs` and `artifacts/logs` back to the machine that runs notebooks 06–07, or run everything there.
- **Kaggle:** push the repo first, including `splits/split.csv`. Open `04_encoder` / `05_lm_features`, add the competition as input data, and turn on internet and a GPU. Run all, then download `artifacts/E1_outputs.zip` / `artifacts/X3_features_outputs.zip` from the output and unzip them into the local `artifacts/`.
- **Colab:** same as Kaggle, plus Kaggle credentials (`KAGGLE_API_TOKEN`, or `~/.kaggle/kaggle.json`) for the download. Set `HW3_ARTIFACTS` to a Drive folder to keep outputs if the session dies.
- **X3 crop ablation (optional, before the full X3 run):** `HW3_ARTIFACTS=$PWD/artifacts/lm_crop_start uv run python -m hw3.lm --config lm_crop_start`, then the same with `middle`; or the optional cell in `05_lm_features`. It compares reading the start vs the middle of texts longer than the token budget.
- **Smoke check (any CPU):** `uv run python -m hw3.encoder --smoke`, `uv run python -m hw3.lm --smoke`, or `SMOKE = True` in the notebook.

Environment variables: `HW3_DATA` (data folder) and `HW3_ARTIFACTS` (artifacts folder).
