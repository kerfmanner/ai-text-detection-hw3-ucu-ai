import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from .data import LABELS


def class_balance(df, normalize=False):
    table = pd.crosstab([df.lang, df.domain], df.label, margins=not normalize, normalize="index" if normalize else False)
    return table[LABELS + ([] if normalize else ["All"])].round(3)


def plot_class_balance(df, ax=None):
    share = class_balance(df, normalize=True)
    ax = share.plot.barh(stacked=True, figsize=(7, 4), ax=ax)
    ax.set(xlabel="share of texts", ylabel="")
    return ax


def length_table(df):
    words = df.text.str.split().str.len()
    return words.groupby([df.lang, df.label]).describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%", "max"]]


def plot_lengths(df):
    words = df.text.str.split().str.len()
    fig, axes = plt.subplots(1, df.lang.nunique(), figsize=(11, 3.2), sharey=True)
    for ax, (lang, g) in zip(axes, df.groupby("lang")):
        for label in LABELS:
            ax.hist(np.log10(words[g.index][g.label == label]), bins=50, histtype="step", density=True, label=label)
        ax.set(title=lang, xlabel="log10(words)")
    axes[0].legend()
    return fig


def examples(df, per_group=1, chars=220, seed=0):
    sample = df.groupby(["lang", "label"]).sample(per_group, random_state=seed)
    snippet = sample.text.str.replace(r"\s+", " ", regex=True).str[:chars] + " …"
    return sample[["lang", "domain", "label"]].assign(snippet=snippet).reset_index(drop=True)


def stats_by_class(df, stats):
    return stats.groupby([df.lang, df.label]).median().T.round(3)


def token_fertility(df, model_names, n=3000, seed=0, body=510):
    hf_logging.set_verbosity_error()
    sample = df.groupby("lang").sample(n, random_state=seed)
    words = sample.text.str.split().str.len().to_numpy()
    rows = []
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name)
        tokens = np.array([len(x) for x in tok(sample.text.tolist(), add_special_tokens=False)["input_ids"]])
        frame = pd.DataFrame({"lang": sample.lang.to_numpy(), "tpw": tokens / words, "tokens": tokens, "windows": np.ceil(tokens / body)})
        for lang, g in frame.groupby("lang"):
            rows.append(
                {
                    "model": name,
                    "lang": lang,
                    "tokens_per_word": g.tpw.median(),
                    "tokens_p50": g.tokens.median(),
                    "tokens_p95": g.tokens.quantile(0.95),
                    f"share_over_{body}": (g.tokens > body).mean(),
                    "windows_p95": g.windows.quantile(0.95),
                    "windows_p99": g.windows.quantile(0.99),
                }
            )
    return pd.DataFrame(rows).round(3)


def distinctive_words(df, lang, label, versus="human", top=15, n=6000, seed=0):
    sub = df[(df.lang == lang) & df.label.isin([label, versus])]
    sub = sub.groupby("label").sample(min(n, sub.label.value_counts().min()), random_state=seed)
    cv = CountVectorizer(min_df=20, lowercase=True)
    X = cv.fit_transform(sub.text)
    a = np.asarray(X[(sub.label == label).to_numpy()].sum(0)).ravel() + 1
    b = np.asarray(X[(sub.label == versus).to_numpy()].sum(0)).ravel() + 1
    log_odds = np.log(a / (a.sum() - a)) - np.log(b / (b.sum() - b))
    z = log_odds / np.sqrt(1 / a + 1 / b)
    vocab = cv.get_feature_names_out()
    order = np.argsort(z)
    return pd.DataFrame({f"more in {label}": vocab[order[::-1][:top]], f"more in {versus}": vocab[order[:top]]})


def top_neighbor(df, nb):
    top = nb.loc[nb.groupby("i").sim.idxmax()]
    labels = df.label.to_numpy()
    return pd.DataFrame({"i": top.i, "j": top.j, "sim": top.sim, "label": labels[top.i], "neighbor_label": labels[top.j], "lang": df.lang.to_numpy()[top.i]})


def plot_neighbor_similarity(top, threshold):
    fig, axes = plt.subplots(1, top.lang.nunique(), figsize=(11, 3.2), sharey=True)
    for ax, (lang, g) in zip(axes, top.groupby("lang")):
        for label in LABELS:
            ax.hist(g.sim[g.label == label], bins=50, range=(0, 1), histtype="step", density=True, label=label)
        ax.axvline(threshold, color="k", ls="--", lw=1)
        ax.set(title=lang, xlabel="cosine similarity to nearest train text")
    axes[0].legend()
    return fig


def trigrams(words):
    return {tuple(words[k : k + 3]) for k in range(len(words) - 2)}


def fusion_profile(df, top, threshold, n=3000, bins=10, seed=0):
    pairs = top[(top.label == "fusion") & (top.neighbor_label == "human") & (top.sim >= threshold)]
    pairs = pairs.sample(min(n, len(pairs)), random_state=seed)
    rows = []
    for i, j in zip(pairs.i, pairs.j):
        fw, hw = df.text.iat[i].split(), df.text.iat[j].split()
        human = trigrams(hw)
        kept = np.array([tuple(fw[k : k + 3]) in human for k in range(len(fw) - 2)], float)
        position = np.minimum((np.arange(len(kept)) / max(len(kept), 1) * bins).astype(int), bins - 1)
        profile = np.bincount(position, kept, bins) / np.maximum(np.bincount(position, minlength=bins), 1)
        rows.append({"lang": df.lang.iat[i], "domain": df.domain.iat[i], "length_ratio": len(fw) / len(hw), "kept": kept.mean(), **{f"bin{b}": p for b, p in enumerate(profile)}})
    return pd.DataFrame(rows)


def plot_fusion_profile(profile, bins=10):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for lang, g in profile.groupby("lang"):
        ax.plot(np.arange(bins) / bins, g[[f"bin{b}" for b in range(bins)]].mean(), marker="o", label=lang)
    ax.set(xlabel="relative position in fusion text", ylabel="share of word 3-grams kept from human source", ylim=(0, 1))
    ax.legend()
    return fig
