import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .config import art
from .data import LABELS, load_test, splits, subsample
from .metrics import evaluate
from .runs import save_preds, save_run, to_labels, track

MODELS = {"linear_svc": LinearSVC, "complement_nb": ComplementNB}


def hash_counts(spec, texts):
    hv = HashingVectorizer(
        analyzer=spec["analyzer"],
        ngram_range=tuple(spec["ngram_range"]),
        lowercase=spec["lowercase"],
        n_features=2**21,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )
    texts = texts.str[: spec["max_chars"]] if spec.get("max_chars") else texts
    parts = np.array_split(texts.to_numpy(), max(1, len(texts) // 5000))
    return vstack(Parallel(n_jobs=-1)(delayed(hv.transform)(p) for p in parts)).tocsr()


class Features:
    def __init__(self, specs):
        self.specs = specs

    def fit_transform(self, texts):
        counts = [hash_counts(s, texts) for s in self.specs]
        self.tfidf = [TfidfTransformer(sublinear_tf=True).fit(c) for c in counts]
        return hstack([t.transform(c) for t, c in zip(self.tfidf, counts)]).tocsr()

    def transform(self, texts):
        return hstack([t.transform(hash_counts(s, texts)) for t, s in zip(self.tfidf, self.specs)]).tocsr()


def make_model(spec):
    spec = dict(spec)
    return MODELS[spec.pop("name")](**spec)


def scores(model, X):
    s = model.decision_function(X) if hasattr(model, "decision_function") else model.predict_log_proba(X)
    return pd.DataFrame(s, columns=model.classes_)[LABELS]


def search(cfg, train, val):
    train = subsample(train, cfg["search_docs"], cfg["seed"])
    rows = []
    for names in cfg["grid"]["features"]:
        feats = Features([cfg["features"][n] for n in names])
        Xtr, Xva = feats.fit_transform(train.text), feats.transform(val.text)
        for spec in cfg["grid"]["models"]:
            pred = to_labels(scores(make_model(spec).fit(Xtr, train.label), Xva))
            rows.append({"features": names, "model": spec, **evaluate(val.label, pred)})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)


def fit_predict(train, parts, feature_specs, model_spec, fit_docs, seed):
    train = subsample(train, fit_docs, seed)
    feats = Features(feature_specs)
    model = make_model(model_spec).fit(feats.fit_transform(train.text), train.label)
    return {name: scores(model, feats.transform(df.text)).assign(id=df.id.to_numpy()) for name, df in parts.items()}


def load_data(cfg):
    train, val = splits()
    return train, subsample(val, cfg.get("eval_docs"), cfg["seed"]), subsample(load_test(), cfg.get("eval_docs"), cfg["seed"])


def best_of(cfg, grid):
    best = grid.iloc[0]
    return {"features": [cfg["features"][n] for n in best.features], "model": best.model, "feature_names": best.features}


def run_baseline(cfg):
    train, val, test = load_data(cfg)
    with track(cfg["id"], cfg) as info:
        grid = search(cfg, train, val)
        grid.to_csv(art("runs", f"{cfg['id']}_grid.csv"), index=False)
        info["best"] = best = best_of(cfg, grid)
        preds = fit_predict(train, {"val": val, "test": test}, best["features"], best["model"], cfg["fit_docs"], cfg["seed"])
        for part, p in preds.items():
            save_preds(cfg["id"], part, p)
    return grid


def run_lang_tuned(cfg):
    train, val, test = load_data(cfg)
    grids, preds = {}, {"val": [], "test": []}
    with track(cfg["id"], cfg) as info:
        for lang in cfg["langs"]:
            lcfg = {**cfg, **cfg["langs"][lang]}
            grids[lang] = search(lcfg, train[train.lang == lang], val[val.lang == lang])
            best = best_of(lcfg, grids[lang])
            info[f"best_{lang}"] = best
            parts = {"val": val[val.lang == lang], "test": test[test.lang == lang]}
            for part, p in fit_predict(train[train.lang == lang], parts, best["features"], best["model"], cfg["fit_docs"], cfg["seed"]).items():
                preds[part].append(p)
        for part, p in preds.items():
            save_preds(cfg["id"], part, pd.concat(p, ignore_index=True))
        pd.concat(grids, names=["lang"]).to_csv(art("runs", f"{cfg['id']}_grid.csv"))
    return grids


def run_two_stage(cfg):
    train, val, test = load_data(cfg)
    ref = json.loads(art("runs", f"{cfg['reference']}.json").read_text())["best"]
    domain_of = lambda df: df.lang + "|" + df.domain
    with track(cfg["id"], cfg) as info:
        fit = subsample(train, cfg["fit_docs"], cfg["seed"])
        feats = Features(ref["features"])
        Xtr, Xva, Xte = feats.fit_transform(fit.text), feats.transform(val.text), feats.transform(test.text)
        router = make_model(cfg["domain_model"]).fit(Xtr, domain_of(fit))
        routes = {"val": {"given": domain_of(val), "predicted": router.predict(Xva)}, "test": {"given": domain_of(test), "predicted": router.predict(Xte)}}
        info["domain_accuracy_val"] = float((routes["val"]["predicted"] == routes["val"]["given"]).mean())
        for part, df in {"val": val, "test": test}.items():
            save_preds("D1", part, pd.DataFrame({"id": df.id, "domain_pred": routes[part]["predicted"]}))
        experts = {"*": make_model(ref["model"]).fit(Xtr, fit.label)}
        for dom, rows in fit.groupby(domain_of(fit)).indices.items():
            if len(rows) >= cfg["min_domain_docs"]:
                experts[dom] = make_model(ref["model"]).fit(Xtr[rows], fit.label.to_numpy()[rows])
        info["experts"] = sorted(experts)
        for routing, run_id in cfg["routings"].items():
            for part, df, X in [("val", val, Xva), ("test", test, Xte)]:
                route = np.asarray(routes[part][routing])
                out = pd.DataFrame(np.nan, index=range(len(df)), columns=LABELS)
                for dom in np.unique(route):
                    rows = np.flatnonzero(route == dom)
                    out.iloc[rows] = scores(experts.get(dom, experts["*"]), X[rows]).to_numpy()
                save_preds(run_id, part, out.assign(id=df.id.to_numpy()))
            if run_id != cfg["id"]:
                save_run(run_id, {**cfg, "change": cfg["change_" + routing]}, status="ok")
    return routes
