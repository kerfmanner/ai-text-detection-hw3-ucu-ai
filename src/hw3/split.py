import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from .config import art, data_dir
from .data import SPLIT_PATH


def nearest_neighbors(df, k, min_df, max_df, chunk=1000):
    out = []
    for _, block in df.groupby(["lang", "domain"]):
        pos = block.index.to_numpy()
        X = TfidfVectorizer(min_df=min_df, max_df=max_df, sublinear_tf=True, dtype=np.float32).fit_transform(block.text)
        XT = X.T.tocsr()
        kk = min(k, len(block) - 1)
        for s in range(0, X.shape[0], chunk):
            S = (X[s : s + chunk] @ XT).toarray()
            rows = np.arange(S.shape[0])
            S[rows, rows + s] = -1
            nb = np.argpartition(-S, kk, axis=1)[:, :kk]
            out.append(pd.DataFrame({"i": np.repeat(pos[s : s + chunk], kk), "j": pos[nb].ravel(), "sim": np.take_along_axis(S, nb, 1).ravel()}))
    return pd.concat(out, ignore_index=True)


def neighbors(cfg):
    path = art("split", "neighbors.parquet")
    if not path.exists():
        df = pd.read_csv(data_dir() / "train.csv")
        nearest_neighbors(df, cfg["neighbors"], cfg["min_df"], cfg["max_df"]).to_parquet(path, index=False)
    return pd.read_parquet(path)


def prefixes(df, n_words):
    return df.text.str.split().str[:n_words].str.join(" ")


def prefix_edges(df, n_words, max_docs):
    prefix = prefixes(df, n_words)
    codes = pd.factorize(prefix)[0]
    size = np.bincount(codes)[codes]
    anchor = pd.Series(np.arange(len(df))).groupby(codes).transform("first").to_numpy()
    i = np.flatnonzero((size > 1) & (size <= max_docs))
    return i, anchor[i]


def build_split(cfg):
    df = pd.read_csv(data_dir() / "train.csv")
    nb = neighbors(cfg)
    sim_edges = nb[nb.sim >= cfg["link_threshold"]]
    pi, pj = prefix_edges(df, cfg["prefix_words"], cfg["prefix_max_docs"])
    i, j = np.r_[sim_edges.i, pi], np.r_[sim_edges.j, pj]
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(df), len(df)))
    _, group = connected_components(graph, directed=False)
    strata = df.lang + "|" + df.domain + "|" + df.label
    folds = StratifiedGroupKFold(n_splits=round(1 / cfg["val_fraction"]), shuffle=True, random_state=cfg["seed"])
    _, val_idx = next(folds.split(df, strata, group))
    split = pd.DataFrame({"id": df.id, "group": group, "split": "train"})
    split.loc[val_idx, "split"] = "val"
    SPLIT_PATH.parent.mkdir(exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)
    return split


def leakage_audit(df, nb, cfg, by="label"):
    split = df.split.to_numpy()
    val = df[split == "val"]
    cross = nb[(split[nb.i] == "val") & (split[nb.j] == "train")]
    top = cross.groupby("i").sim.max().reindex(val.index, fill_value=0)
    prefix = prefixes(df, cfg["prefix_words"])
    train_counts = prefix[split == "train"].value_counts()
    shared = prefix[val.index].map(train_counts).fillna(0)
    return pd.DataFrame(
        {
            "val_docs": val.groupby(by).size(),
            "max_sim_to_train_p99": top.groupby(val[by]).quantile(0.99),
            "share_sim_ge_threshold": (top >= cfg["link_threshold"]).groupby(val[by]).mean(),
            "share_prefix_in_train": (shared > 0).groupby(val[by]).mean(),
            "share_rare_prefix_in_train": shared.between(1, cfg["prefix_max_docs"] - 1).groupby(val[by]).mean(),
        }
    ).round(4)
