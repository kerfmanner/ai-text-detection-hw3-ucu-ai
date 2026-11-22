import re

import numpy as np
import pandas as pd

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sentence_lengths(text):
    lengths = [len(s.split()) for s in SENTENCE.split(text) if s.strip()]
    return np.mean(lengths), np.std(lengths)


def text_stats(texts):
    words = texts.str.split()
    n_words, n_chars = words.str.len(), texts.str.len()
    sentences = np.array([sentence_lengths(t) for t in texts])
    return pd.DataFrame(
        {
            "n_words": n_words,
            "n_chars": n_chars,
            "word_len": words.apply(lambda w: np.mean([len(x) for x in w])),
            "sent_len_mean": sentences[:, 0],
            "sent_len_std": sentences[:, 1],
            "ttr_200": words.apply(lambda w: len(set(w[:200])) / len(w[:200])),
            "upper_ratio": texts.str.count(r"[A-ZÄÖÜ]") / n_chars,
            "digit_ratio": texts.str.count(r"\d") / n_chars,
            "punct_per_word": texts.str.count(r"[,;:!?()\"'„“”–—-]") / n_words,
        },
        index=texts.index,
    )
