import pandas as pd

from .config import ROOT
from .data import LABELS, load_sample_submission
from .runs import has_preds, load_preds, to_labels


def select_final(log):
    candidates = log[log.macro_f1.notna() & log.id.map(lambda m: has_preds(m, "test"))]
    return candidates.sort_values("macro_f1", ascending=False).iloc[0]


def make_submission(model_id):
    preds = load_preds(model_id, "test")
    labels = pd.Series(to_labels(preds), index=preds.id)
    sample = load_sample_submission()
    return pd.DataFrame({"id": sample.id, "label": labels.reindex(sample.id).to_numpy()})


def validate_submission(sub):
    sample = load_sample_submission()
    checks = {
        "columns are exactly [id, label]": list(sub.columns) == ["id", "label"],
        "row count equals sample_submission": len(sub) == len(sample),
        "every id exactly once": sub.id.is_unique,
        "same id set as sample_submission": set(sub.id) == set(sample.id),
        "same row order as sample_submission": sub.id.tolist() == sample.id.tolist(),
        "labels are lowercase human/ai/fusion": sub.label.isin(LABELS).all(),
        "no missing values": not sub.isna().any().any(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"submission check failed: {failed}")
    return pd.Series(checks)


def save_submission(model_id, path=ROOT / "submission.csv"):
    sub = make_submission(model_id)
    validate_submission(sub)
    sub.to_csv(path, index=False)
    return validate_submission(pd.read_csv(path))
