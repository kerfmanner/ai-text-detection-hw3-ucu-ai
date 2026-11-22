import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from .config import art
from .data import LABELS, load_test, splits, subsample
from .runs import has_preds, load_preds, save_preds, track
from .textstats import text_stats


def prefixed(df, prefix):
    return df.set_index("id").add_prefix(prefix).reset_index()


def stack_frame(cfg, df, part, members, categories):
    if cfg["domain_source"] == "predicted":
        domain = load_preds("D1", part).set_index("id").domain_pred.reindex(df.id).to_numpy()
    else:
        domain = (df.lang + "|" + df.domain).to_numpy()
    X = pd.DataFrame({"id": df.id.to_numpy(), "lang": pd.Categorical(df.lang, categories["lang"]), "domain": pd.Categorical(domain, categories["domain"])})
    for m in members:
        X = X.merge(prefixed(load_preds(m, part), f"{m}_"), on="id", how="left")
    if cfg["lm_features"] and art("features", f"lm_{part}.parquet").exists():
        X = X.merge(prefixed(pd.read_parquet(art("features", f"lm_{part}.parquet")), "lm_"), on="id", how="left")
    if cfg["text_stats"]:
        X = pd.concat([X, text_stats(df.text).reset_index(drop=True)], axis=1)
    return X.drop(columns="id")


def gbm(cfg):
    return HistGradientBoostingClassifier(**cfg["model"], categorical_features="from_dtype", class_weight="balanced", random_state=cfg["seed"])


def proba(model, X):
    return pd.DataFrame(model.predict_proba(X), columns=model.classes_)[LABELS].to_numpy()


def run_ensemble(cfg):
    train, val = splits()
    val, test = subsample(val, cfg.get("eval_docs"), cfg["seed"]), subsample(load_test(), cfg.get("eval_docs"), cfg["seed"])
    members = [m for m in cfg["members"] if has_preds(m, "val") and has_preds(m, "test")]
    categories = {"lang": sorted(set(train.lang)), "domain": sorted(set(train.lang + "|" + train.domain))}
    Xv, Xt = stack_frame(cfg, val, "val", members, categories), stack_frame(cfg, test, "test", members, categories)
    by_v = val.lang.to_numpy() if cfg["stack_by"] == "lang" else np.zeros(len(val))
    by_t = test.lang.to_numpy() if cfg["stack_by"] == "lang" else np.zeros(len(test))
    oof, out = np.zeros((len(val), len(LABELS))), np.zeros((len(test), len(LABELS)))
    with track(cfg["id"], cfg) as info:
        info |= {"members": members, "features": list(Xv.columns)}
        for key in np.unique(by_v):
            v, t = np.flatnonzero(by_v == key), np.flatnonzero(by_t == key)
            y, groups = val.label.to_numpy()[v], val.group.to_numpy()[v]
            folds = StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"])
            for fit_idx, pred_idx in folds.split(v, y, groups):
                oof[v[pred_idx]] = proba(gbm(cfg).fit(Xv.iloc[v[fit_idx]], y[fit_idx]), Xv.iloc[v[pred_idx]])
            out[t] = proba(gbm(cfg).fit(Xv.iloc[v], y), Xt.iloc[t])
        save_preds(cfg["id"], "val", pd.DataFrame(oof, columns=LABELS).assign(id=val.id.to_numpy()))
        save_preds(cfg["id"], "test", pd.DataFrame(out, columns=LABELS).assign(id=test.id.to_numpy()))
    return Xv, oof
