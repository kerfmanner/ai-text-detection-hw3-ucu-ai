import itertools
import json
import os
import platform
import time
import zipfile
from contextlib import contextmanager
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .config import ROOT, art
from .data import LABELS, load_train
from .metrics import by_group, evaluate, per_class, plot_confusion


def save_preds(model_id, part, df):
    df.to_parquet(art("preds", f"{model_id}_{part}.parquet"), index=False)


def load_preds(model_id, part):
    return pd.read_parquet(art("preds", f"{model_id}_{part}.parquet"))


def has_preds(model_id, part):
    return art("preds", f"{model_id}_{part}.parquet").exists()


def to_labels(scores):
    return np.array(LABELS)[scores[LABELS].to_numpy().argmax(1)]


def compute_info():
    import torch

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
        "versions": {p: version(p) for p in ["torch", "transformers", "scikit-learn", "pandas", "numpy"]},
    }


def save_run(run_id, cfg, **info):
    meta = {"id": run_id, "config": cfg, "compute": compute_info(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    art("runs", f"{run_id}.json").write_text(json.dumps(meta, indent=2, default=str))


@contextmanager
def track(run_id, cfg):
    info, start = {"status": "failed"}, time.time()
    try:
        yield info
        info["status"] = "ok"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        save_run(run_id, cfg, runtime_s=round(time.time() - start, 1), **info)


def val_report(model_id):
    preds = load_preds(model_id, "val").merge(load_train()[["id", "lang", "domain", "label"]], on="id")
    return preds.assign(pred=to_labels(preds))


def report(model_id):
    r = val_report(model_id)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    plot_confusion(r.label, r.pred, title=f"{model_id} (validation)", ax=ax)
    return {"per_class": per_class(r.label, r.pred), "by_domain": by_group(r, ["lang", "domain"]), "figure": fig}


def compare(model_ids, by=("lang", "domain")):
    by = list(by)
    table = {}
    for m in [m for m in model_ids if has_preds(m, "val")]:
        r = val_report(m)
        table[m] = by_group(pd.concat([r.assign(**{c: "all" for c in by}), r]), by).macro_f1
    return pd.DataFrame(table)



def error_overlap(model_ids):
    wrong = {m: val_report(m).set_index("id").pipe(lambda r: r.pred != r.label) for m in model_ids if has_preds(m, "val")}
    rows = {}
    for a, b in itertools.combinations(wrong, 2):
        wa, wb = wrong[a].align(wrong[b], join="inner")
        rows[(a, b)] = {"n": len(wa), "both_wrong": (wa & wb).mean(), "only_one_wrong": (wa ^ wb).mean()}
    return pd.DataFrame(rows).T.astype({"n": int}).round(4)


def export_outputs(name, patterns):
    root = art()
    path = root / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as z:
        for pattern in patterns:
            for f in root.glob(pattern):
                z.write(f, f.relative_to(root))
    return path


def experiment_log():
    rows = []
    for path in sorted(art("runs").glob("*.json")):
        meta = json.loads(path.read_text())
        cfg = meta["config"]
        row = {
            "id": meta["id"],
            "reference": cfg.get("reference"),
            "change": cfg.get("change"),
            "status": meta["status"],
            "device": meta["compute"]["gpu"] or meta["compute"]["device"],
            "runtime_s": meta.get("runtime_s"),
        }
        if meta["status"] == "ok" and has_preds(meta["id"], "val"):
            r = val_report(meta["id"])
            row |= {k: round(v, 4) for k, v in evaluate(r.label, r.pred).items() if k == "macro_f1" or k.startswith("f1_")}
        row["notes"] = meta.get("error") or cfg.get("notes")
        rows.append(row)
    log = pd.DataFrame(rows)
    if "macro_f1" in log:
        ref = log.set_index("id").macro_f1
        log.insert(log.columns.get_loc("macro_f1") + 1, "delta_vs_ref", log.macro_f1 - log.reference.map(ref))
    out = art("experiment_log.csv") if config.SMOKE else ROOT / "experiments" / "log.csv"
    out.parent.mkdir(exist_ok=True)
    log.to_csv(out, index=False)
    return log
