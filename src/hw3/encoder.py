import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from .config import art, load_config, seed_everything, set_smoke
from .data import LABEL2ID, LABELS, load_test, splits, subsample
from .runs import save_preds, track

log = logging.getLogger("hw3.encoder")


def setup_logging(run_id):
    logging.basicConfig(filename=art("logs", f"{run_id}.log"), level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    hf_logging.set_verbosity_error()


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def tokenize(tok, texts, batch=1024):
    texts, out = list(texts), []
    for s in range(0, len(texts), batch):
        out += [np.asarray(ids, np.int32) for ids in tok(texts[s : s + batch], add_special_tokens=False)["input_ids"]]
    return out


def window_starts(n_tokens, body, max_chunks):
    n = min(max(1, math.ceil(n_tokens / body)), max_chunks)
    return np.linspace(0, max(n_tokens - body, 0), n).round().astype(int)


def batchify(seqs, tok, device):
    seqs = [[tok.cls_token_id, *s.tolist(), tok.sep_token_id] for s in seqs]
    ids = torch.full((len(seqs), max(map(len, seqs))), tok.pad_token_id)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def class_weights(y, mode, device):
    if mode != "balanced":
        return None
    counts = np.bincount(y, minlength=len(LABELS))
    return torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float32, device=device)


def train_loop(cfg, model, n_docs, make_batch, weight, device):
    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    total = max(1, cfg["epochs"] * math.ceil(n_docs / bs) // accum)
    warm = max(1, int(total * cfg["warmup_ratio"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    rng = np.random.default_rng(cfg["seed"])
    model.train()
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(n_docs)
        for step, start in enumerate(tqdm(range(0, len(order), bs), desc=f"train epoch {epoch}", mininterval=30)):
            ids, mask, y = make_batch(order[start : start + bs], rng)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), weight=weight, ignore_index=-100)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if step % cfg["log_every"] == 0:
                log.info("epoch %d step %d/%d loss %.4f lr %.2e", epoch, step, math.ceil(n_docs / bs), loss.item(), sched.get_last_lr()[0])


def fit(cfg, model, tok, docs, y, device):
    body = cfg["max_len"] - 2

    def make_batch(idx, rng):
        offsets = [rng.integers(0, max(len(docs[i]) - body, 0) + 1) for i in idx]
        ids, mask = batchify([docs[i][o : o + body] for i, o in zip(idx, offsets)], tok, device)
        return ids, mask, torch.as_tensor(y[idx], device=device)

    train_loop(cfg, model, len(docs), make_batch, class_weights(y, cfg["class_weights"], device), device)


@torch.no_grad()
def window_probs(cfg, model, tok, docs, langs, device):
    body, dtype = cfg["max_len"] - 2, amp_dtype(device)
    items = [(d, s) for d, (toks, lang) in enumerate(zip(docs, langs)) for s in window_starts(len(toks), body, cfg["max_chunks"][lang])]
    order = sorted(range(len(items)), key=lambda k: len(docs[items[k][0]]) - items[k][1])
    probs = np.zeros((len(items), len(LABELS)), np.float32)
    model.eval()
    for start in tqdm(range(0, len(order), cfg["eval_batch_size"]), desc="predict", mininterval=30):
        ks = order[start : start + cfg["eval_batch_size"]]
        ids, mask = batchify([docs[items[k][0]][items[k][1] : items[k][1] + body] for k in ks], tok, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs[ks] = logits.float().softmax(-1).cpu().numpy()
        if start // cfg["eval_batch_size"] % cfg["log_every"] == 0:
            log.info("predict %d/%d windows", start, len(items))
    return pd.DataFrame(probs, columns=LABELS).assign(doc=[d for d, _ in items], start=[s for _, s in items])


def window_stats(windows):
    stats = windows.groupby("doc")[LABELS].agg(["mean", "max", "min", "std", "first", "last"]).fillna(0)
    stats.columns = [label if stat == "mean" else f"{label}_{stat}" for label, stat in stats.columns]
    return stats.assign(n_chunks=windows.groupby("doc").size()).reset_index(drop=True)


def eval_parts(cfg):
    val = splits()[1]
    return {"val": subsample(val, cfg["eval_docs"], cfg["seed"]), "test": subsample(load_test(), cfg["eval_docs"], cfg["seed"])}


def predict_parts(cfg, model, tok, parts, device):
    for part, df in parts.items():
        windows = window_probs(cfg, model, tok, tokenize(tok, df.text), df.lang.to_numpy(), device)
        windows.assign(id=df.id.to_numpy()[windows.doc]).drop(columns="doc").to_parquet(art("features", f"{cfg['id']}_windows_{part}.parquet"), index=False)
        save_preds(cfg["id"], part, window_stats(windows).assign(id=df.id.to_numpy()))


def run(cfg):
    setup_logging(cfg["id"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = subsample(splits()[0], cfg["train_docs"], cfg["seed"])
    with track(cfg["id"], cfg) as info:
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
        ).to(device)
        info |= {"train_docs": len(train), "amp": str(amp_dtype(device))}
        fit(cfg, model, tok, tokenize(tok, train.text), train.label.map(LABEL2ID).to_numpy(), device)
        if cfg["save_model"]:
            model.save_pretrained(art("models", cfg["id"]))
            tok.save_pretrained(art("models", cfg["id"]))
        predict_parts(cfg, model, tok, eval_parts(cfg), device)
    return info


def repredict(cfg):
    setup_logging(cfg["id"] + "_repredict")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with track(cfg["id"] + "_repredict", cfg) as info:
        tok = AutoTokenizer.from_pretrained(art("models", cfg["id"]))
        model = AutoModelForSequenceClassification.from_pretrained(art("models", cfg["id"])).to(device)
        predict_parts(cfg, model, tok, eval_parts(cfg), device)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="encoder")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repredict", action="store_true")
    args = parser.parse_args()
    set_smoke(args.smoke)
    print((repredict if args.repredict else run)(load_config(args.config)))
