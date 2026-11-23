import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from .config import art
from .data import LABELS, load_test, splits, subsample
from .runs import has_preds, load_preds, save_preds, track
from .textstats import text_stats
from .tfidf import fit_predict


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


def macro_f1(codes, pred):
    k = len(LABELS)
    cm = np.bincount(codes * k + pred, minlength=k * k).reshape(k, k)
    return np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1))


def tune_bias(logp, codes, grid):
    best = max(itertools.product(grid, grid), key=lambda b: macro_f1(codes, (logp + [0, *b]).argmax(1)))
    return np.array([0, *best])


def with_bias(grid, oof, out, y, folds):
    codes, grid = pd.Categorical(y, LABELS).codes, np.linspace(*grid)
    lo, lt = np.log(oof + 1e-9), np.log(out + 1e-9)
    honest = np.zeros_like(lo)
    for fit_idx, pred_idx in folds:
        honest[pred_idx] = lo[pred_idx] + tune_bias(lo[fit_idx], codes[fit_idx], grid)
    bias = tune_bias(lo, codes, grid)
    return honest, lt + bias, bias


def run_ensemble(cfg):
    train, val = splits()
    val, test = subsample(val, cfg.get("eval_docs"), cfg["seed"]), subsample(load_test(), cfg.get("eval_docs"), cfg["seed"])
    members = [m for m in cfg["members"] if has_preds(m, "val") and has_preds(m, "test")]
    categories = {"lang": sorted(set(train.lang)), "domain": sorted(set(train.lang + "|" + train.domain))}
    Xv, Xt = stack_frame(cfg, val, "val", members, categories), stack_frame(cfg, test, "test", members, categories)
    by_v = val.lang.to_numpy() if cfg["stack_by"] == "lang" else np.zeros(len(val))
    by_t = test.lang.to_numpy() if cfg["stack_by"] == "lang" else np.zeros(len(test))
    oof, out = np.zeros((len(val), len(LABELS))), np.zeros((len(test), len(LABELS)))
    oof_b, out_b, offsets = np.zeros_like(oof), np.zeros_like(out), {}
    with track(cfg["id"], cfg) as info:
        info |= {"members": members, "features": list(Xv.columns)}
        for key in np.unique(by_v):
            v, t = np.flatnonzero(by_v == key), np.flatnonzero(by_t == key)
            y, groups = val.label.to_numpy()[v], val.group.to_numpy()[v]
            folds = list(StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"]).split(v, y, groups))
            for fit_idx, pred_idx in folds:
                oof[v[pred_idx]] = proba(gbm(cfg).fit(Xv.iloc[v[fit_idx]], y[fit_idx]), Xv.iloc[v[pred_idx]])
            out[t] = proba(gbm(cfg).fit(Xv.iloc[v], y), Xt.iloc[t])
            oof_b[v], out_b[t], offsets[key] = with_bias(cfg["bias"]["grid"], oof[v], out[t], y, folds)
        save_preds(cfg["id"], "val", pd.DataFrame(oof, columns=LABELS).assign(id=val.id.to_numpy()))
        save_preds(cfg["id"], "test", pd.DataFrame(out, columns=LABELS).assign(id=test.id.to_numpy()))
    with track(cfg["bias"]["id"], cfg["bias"]) as info:
        info["offsets"] = {key: dict(zip(LABELS, b.round(3).tolist())) for key, b in offsets.items()}
        save_preds(cfg["bias"]["id"], "val", pd.DataFrame(oof_b, columns=LABELS).assign(id=val.id.to_numpy()))
        save_preds(cfg["bias"]["id"], "test", pd.DataFrame(out_b, columns=LABELS).assign(id=test.id.to_numpy()))
    return Xv, oof


def member_oof(meta, train, folds, seed):
    best, oof = meta["best"], np.zeros((len(train), len(LABELS)))
    for fit_idx, pred_idx in folds:
        part = fit_predict(train.iloc[fit_idx], {"oof": train.iloc[pred_idx]}, best["features"], best["model"], meta["config"].get("fit_docs"), seed)
        oof[pred_idx] = part["oof"][LABELS].to_numpy()
    return pd.DataFrame(oof, columns=LABELS).assign(id=train.id.to_numpy())


def run_train_stack(cfg):
    train, val = splits()
    train = subsample(train, cfg.get("train_docs"), cfg["seed"])
    parts = {"trainoof": train, "val": subsample(val, cfg.get("eval_docs"), cfg["seed"]), "test": subsample(load_test(), cfg.get("eval_docs"), cfg["seed"])}
    categories = {"lang": sorted(set(train.lang)), "domain": sorted(set(train.lang + "|" + train.domain))}
    folds = list(StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"]).split(train, train.label, train.group))
    out = {p: np.zeros((len(parts[p]), len(LABELS))) for p in ["val", "test"]}
    with track(cfg["id"], cfg):
        for m in cfg["members"]:
            if not has_preds(m, "trainoof"):
                save_preds(m, "trainoof", member_oof(json.loads(art("runs", f"{m}.json").read_text()), train, folds, cfg["seed"]))
        X = {p: stack_frame(cfg, df, p, cfg["members"], categories) for p, df in parts.items()}
        for lang in categories["lang"]:
            rows = (train.lang == lang).to_numpy()
            model = gbm(cfg).fit(X["trainoof"][rows], train.label[rows])
            for p in out:
                r = (parts[p].lang == lang).to_numpy()
                out[p][r] = proba(model, X[p][r])
        for p, o in out.items():
            save_preds(cfg["id"], p, pd.DataFrame(o, columns=LABELS).assign(id=parts[p].id.to_numpy()))
