import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .config import art, load_config, seed_everything, set_smoke
from .data import splits, subsample
from .eda import trigrams
from .encoder import amp_dtype, batchify, eval_parts, setup_logging, train_loop
from .runs import save_preds, track
from .sentences import fusion_sources

TAGS = ["original", "rewritten", "ai"]


def word_tags(words, label, source):
    if label == "ai":
        return np.full(len(words), 2, np.int8)
    if label == "fusion" and not isinstance(source, str):
        return np.full(len(words), -1, np.int8)
    covered = np.zeros(len(words), bool)
    if label == "fusion":
        src = trigrams(source.split())
        for k in range(len(words) - 2):
            if tuple(words[k : k + 3]) in src:
                covered[k : k + 3] = True
    return np.where(covered | (label != "fusion"), 0, 1).astype(np.int8)


def encode(tok, df, sources=None, batch=512):
    ids, tags = [], []
    for s in range(0, len(df), batch):
        part = df.iloc[s : s + batch]
        words = [t.split() for t in part.text]
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        for i, w in enumerate(words):
            ids.append(np.asarray(enc["input_ids"][i], np.int32))
            if sources is not None:
                tags.append(word_tags(w, part.label.iat[i], sources.get(part.id.iat[i]))[np.asarray(enc.word_ids(i))])
    return ids, tags


def train_window(n, body, tags, focus, rng):
    hi, rewritten = max(n - body, 0), np.flatnonzero(tags == 1)
    if len(rewritten) and rng.random() < focus:
        return int(np.clip(rng.choice(rewritten) - rng.integers(0, body), 0, hi))
    return int(rng.integers(0, hi + 1))


def fit(cfg, model, tok, ids, tags, device):
    body = cfg["max_len"] - 2
    counts = np.bincount(np.concatenate(tags), minlength=len(TAGS))
    weight = torch.tensor((counts.sum() / (len(TAGS) * np.maximum(counts, 1))) ** cfg["weight_power"], dtype=torch.float32, device=device)

    def make_batch(idx, rng):
        offsets = [train_window(len(ids[i]), body, tags[i], cfg["focus"], rng) for i in idx]
        x, mask = batchify([ids[i][o : o + body] for i, o in zip(idx, offsets)], tok, device)
        y = torch.full_like(x, -100)
        for r, (i, o) in enumerate(zip(idx, offsets)):
            t = tags[i][o : o + body]
            y[r, 1 : 1 + len(t)] = torch.as_tensor(t.astype(np.int64))
        return x, mask, y

    train_loop(cfg, model, len(ids), make_batch, weight, device)


def cover_starts(n, body, stride):
    hi = max(n - body, 0)
    return np.unique(np.r_[np.arange(0, hi + 1, stride), hi])


@torch.no_grad()
def token_probs(cfg, model, tok, docs, device):
    body, dtype = cfg["max_len"] - 2, amp_dtype(device)
    items = [(d, s) for d, toks in enumerate(docs) for s in cover_starts(len(toks), body, cfg["stride"])]
    order = sorted(range(len(items)), key=lambda k: len(docs[items[k][0]]) - items[k][1])
    sums, counts = [np.zeros((len(t), len(TAGS)), np.float32) for t in docs], [np.zeros(len(t), np.float32) for t in docs]
    model.eval()
    for start in tqdm(range(0, len(order), cfg["eval_batch_size"]), desc="predict", mininterval=30):
        ks = order[start : start + cfg["eval_batch_size"]]
        x, mask = batchify([docs[items[k][0]][items[k][1] : items[k][1] + body] for k in ks], tok, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            probs = model(input_ids=x, attention_mask=mask).logits.float().softmax(-1).cpu().numpy()
        for r, k in enumerate(ks):
            d, s = items[k]
            n = min(body, len(docs[d]) - s)
            sums[d][s : s + n] += probs[r, 1 : 1 + n]
            counts[d][s : s + n] += 1
    return [s / c[:, None] for s, c in zip(sums, counts)]


def doc_features(p, smooth):
    k = min(smooth, len(p))
    sm = np.stack([np.convolve(p[:, j], np.ones(k) / k, "valid") for j in range(len(TAGS))], 1)
    out = {f"{t}_mean": p[:, j].mean() for j, t in enumerate(TAGS)}
    out |= {f"{t}_smax": sm[:, j].max() for j, t in enumerate(TAGS)} | {f"{t}_share": (p[:, j] > 0.5).mean() for j, t in enumerate(TAGS)}
    return out | {"rewritten_peak_pos": sm[:, 1].argmax() / max(len(sm) - 1, 1), "n_tokens": len(p)}


def doc_frame(probs, smooth):
    f = pd.DataFrame([doc_features(p, smooth) for p in probs])
    return f.assign(human=(1 - f.ai_mean) * (1 - f.rewritten_smax), ai=f.ai_mean, fusion=(1 - f.ai_mean) * f.rewritten_smax)


def token_f1(probs, tags):
    true, pred = np.concatenate(tags), np.concatenate([p.argmax(1) for p in probs])
    keep = true >= 0
    return dict(zip(TAGS, f1_score(true[keep], pred[keep], labels=[0, 1, 2], average=None).round(4).tolist()))


def predict_parts(cfg, model, tok, sources, device, info):
    for part, df in eval_parts(cfg).items():
        ids, tags = encode(tok, df, sources if part == "val" else None)
        probs = token_probs(cfg, model, tok, ids, device)
        if part == "val":
            info["token_f1_val"] = token_f1(probs, tags)
        save_preds(cfg["id"], part, doc_frame(probs, cfg["smooth"]).assign(id=df.id.to_numpy()))


def run(cfg):
    setup_logging(cfg["id"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources, train = fusion_sources(), splits()[0]
    train = subsample(train[(train.label != "fusion") | train.id.isin(sources.index)].reset_index(drop=True), cfg["train_docs"], cfg["seed"])
    with track(cfg["id"], cfg) as info:
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AutoModelForTokenClassification.from_pretrained(
            cfg["model_name"], num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id={t: i for i, t in enumerate(TAGS)}
        ).to(device)
        ids, tags = encode(tok, train, sources)
        info |= {"train_docs": len(ids), "amp": str(amp_dtype(device)), "train_tokens": dict(zip(TAGS, np.bincount(np.concatenate(tags), minlength=3).tolist()))}
        fit(cfg, model, tok, ids, tags, device)
        if cfg["save_model"]:
            model.save_pretrained(art("models", cfg["id"]))
            tok.save_pretrained(art("models", cfg["id"]))
        predict_parts(cfg, model, tok, sources, device, info)
    return info


def repredict(cfg):
    setup_logging(cfg["id"] + "_repredict")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with track(cfg["id"] + "_repredict", cfg) as info:
        tok = AutoTokenizer.from_pretrained(art("models", cfg["id"]))
        model = AutoModelForTokenClassification.from_pretrained(art("models", cfg["id"])).to(device)
        predict_parts(cfg, model, tok, fusion_sources(), device, info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tokens")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repredict", action="store_true")
    args = parser.parse_args()
    set_smoke(args.smoke)
    print((repredict if args.repredict else run)(load_config(args.config)))
