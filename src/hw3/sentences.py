import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from .config import load_config
from .data import LABELS, SPLIT_PATH, load_train, subsample
from .eda import top_neighbor, trigrams
from .runs import save_preds, track
from .split import neighbors
from .textstats import SENTENCE
from .tfidf import Features, load_data, make_model, reference_best

SENT = ["original", "rewritten", "ai"]
PAIRS_PATH = SPLIT_PATH.parent / "fusion_sources.csv"


def fusion_sources():
    df = load_train()
    if not PAIRS_PATH.exists():
        cfg = load_config("split")
        top = top_neighbor(df, neighbors(cfg))
        pairs = top[(top.label == "fusion") & (top.neighbor_label == "human") & (top.sim >= cfg["link_threshold"])]
        pd.DataFrame({"id": df.id.to_numpy()[pairs.i], "source_id": df.id.to_numpy()[pairs.j]}).to_csv(PAIRS_PATH, index=False)
    pairs = pd.read_csv(PAIRS_PATH)
    return pd.Series(pairs.source_id.map(df.set_index("id").text).to_numpy(), index=pairs.id.to_numpy())


def split_sentences(texts):
    rows = [(d, s) for d, t in enumerate(texts) for s in SENTENCE.split(t) if s.strip()]
    return pd.DataFrame(rows, columns=["doc", "text"])


def kept_share(sentence, source):
    w = sentence.split()
    grams = [tuple(w[k : k + 3]) for k in range(len(w) - 2)]
    return np.mean([g in source for g in grams]) if grams else np.nan


def sentence_labels(cfg, df, sources):
    sents = split_sentences(df.text)
    source = {d: trigrams(t.split()) for d, t in enumerate(df.id.map(sources)) if isinstance(t, str)}
    kept = np.array([kept_share(t, source[d]) if d in source else np.nan for d, t in zip(sents.doc, sents.text)])
    label = df.label.to_numpy()[sents.doc]
    y = np.where(label == "ai", "ai", np.where(kept < cfg["rewritten_below"], "rewritten", "original"))
    keep = ((label != "fusion") | (kept < cfg["rewritten_below"]) | (kept >= cfg["kept_above"])) & (sents.text.str.split().str.len() >= cfg["min_words"])
    return sents[keep.to_numpy()], y[keep.to_numpy()]


def sentence_scores(stage1, feats, sents):
    return pd.DataFrame(stage1.decision_function(feats.transform(sents.text)), columns=stage1.classes_)[SENT].assign(doc=sents.doc.to_numpy())


def doc_features(s, n_docs):
    size = s.groupby("doc").doc.transform("size")
    top = s[SENT].to_numpy().argmax(1)
    s = s.assign(flag_rewritten=top == 1, flag_ai=top == 2, third=np.minimum(s.groupby("doc").cumcount() * 3 // size, 2))
    g = s.groupby("doc")
    out = g[SENT].agg(["mean", "max", "min", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    thirds = s.pivot_table(index="doc", columns="third", values="rewritten", aggfunc="mean").add_prefix("rewritten_third")
    return out.join(g[["flag_rewritten", "flag_ai"]].mean()).join(thirds).assign(n_sentences=g.size()).reindex(range(n_docs)).reset_index(drop=True)


def doc_frame(stage1, feats, df):
    return doc_features(sentence_scores(stage1, feats, split_sentences(df.text)), len(df))


def run_sentences(cfg):
    train, val, test = load_data(cfg)
    sources, ref = fusion_sources(), reference_best(cfg)
    one, two = next(GroupShuffleSplit(1, train_size=0.5, random_state=cfg["seed"]).split(train, groups=train.group))
    fit1 = subsample(train.iloc[one].reset_index(drop=True), cfg["stage1_docs"], cfg["seed"])
    fit2 = subsample(train.iloc[two].reset_index(drop=True), cfg["stage2_docs"], cfg["seed"])
    with track(cfg["id"], cfg) as info:
        sents, y = sentence_labels(cfg, fit1, sources)
        feats = Features(ref["features"])
        stage1 = make_model(cfg["sentence_model"]).fit(feats.fit_transform(sents.text), y)
        vs, vy = sentence_labels(cfg, val, sources)
        info["sentences_fit"] = pd.Series(y).value_counts().to_dict()
        info["sentence_f1_val"] = dict(zip(SENT, f1_score(vy, stage1.predict(feats.transform(vs.text)), labels=SENT, average=None).round(4).tolist()))
        frames = {name: doc_frame(stage1, feats, df) for name, df in {"fit": fit2, "val": val, "test": test}.items()}
        stage2 = HistGradientBoostingClassifier(**cfg["doc_model"], class_weight="balanced", random_state=cfg["seed"]).fit(frames["fit"], fit2.label)
        for part, df in {"val": val, "test": test}.items():
            proba = pd.DataFrame(stage2.predict_proba(frames[part]), columns=stage2.classes_)[LABELS]
            save_preds(cfg["id"], part, pd.concat([proba, frames[part]], axis=1).assign(id=df.id.to_numpy()))
    return stage1, feats
