import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import art, load_config, seed_everything, set_smoke
from .data import LABELS, load_test, splits, subsample
from .encoder import amp_dtype, setup_logging
from .runs import save_preds, track


def token_stats(lo, lp, target):
    lo, lp = lo.float().log_softmax(-1), lp.float().log_softmax(-1)
    po = lo.exp()
    return torch.stack(
        [
            lo.gather(-1, target[:, None]).squeeze(-1),
            lp.gather(-1, target[:, None]).squeeze(-1),
            -(po * lo).sum(-1),
            -(po * lp).sum(-1),
        ]
    ).cpu().numpy()


def summarize(stats, segment):
    lp_obs, lp_perf, ent_obs, xent = stats
    seg = [slice(s, s + segment) for s in range(0, len(lp_perf), segment)]
    seg_lp = np.array([lp_perf[s].mean() for s in seg])
    seg_bino = np.array([-lp_perf[s].mean() / xent[s].mean() for s in seg])
    q10, q50, q90 = np.quantile(lp_obs, [0.1, 0.5, 0.9])
    return {
        "n_tokens": len(lp_obs),
        "lp_obs_mean": lp_obs.mean(),
        "lp_obs_std": lp_obs.std(),
        "lp_obs_q10": q10,
        "lp_obs_q50": q50,
        "lp_obs_q90": q90,
        "lp_perf_mean": lp_perf.mean(),
        "lp_gap_mean": (lp_perf - lp_obs).mean(),
        "ent_obs_mean": ent_obs.mean(),
        "xent_mean": xent.mean(),
        "binoculars": -lp_perf.mean() / xent.mean(),
        "seg_lp_std": seg_lp.std(),
        "seg_lp_min": seg_lp.min(),
        "seg_lp_max": seg_lp.max(),
        "seg_bino_std": seg_bino.std(),
        "seg_bino_min": seg_bino.min(),
        "seg_bino_max": seg_bino.max(),
    }


def crop(ids, n, where):
    start = max(len(ids) - n, 0) // 2 if where == "middle" else 0
    return ids[start : start + n]


@torch.no_grad()
def lm_features(cfg, df, tok, observer, performer, device):
    ids = [crop(tok(t, add_special_tokens=False)["input_ids"], cfg["max_tokens"][lang], cfg["crop"]) for t, lang in zip(df.text, df.lang)]
    order = np.argsort([len(x) for x in ids])
    rows, pad = [None] * len(ids), tok.pad_token_id or tok.eos_token_id
    for start in tqdm(range(0, len(order), cfg["batch_size"]), desc="lm features", mininterval=30):
        batch = order[start : start + cfg["batch_size"]]
        width = max(len(ids[i]) for i in batch)
        x = torch.full((len(batch), width), pad)
        mask = torch.zeros_like(x)
        for r, i in enumerate(batch):
            x[r, : len(ids[i])] = torch.tensor(ids[i])
            mask[r, : len(ids[i])] = 1
        x, mask = x.to(device), mask.to(device)
        lo, lp = observer(input_ids=x, attention_mask=mask).logits, performer(input_ids=x, attention_mask=mask).logits
        for r, i in enumerate(batch):
            n = len(ids[i])
            rows[i] = summarize(token_stats(lo[r, : n - 1], lp[r, : n - 1], x[r, 1:n]), cfg["segment"])
    return pd.DataFrame(rows).assign(id=df.id.to_numpy())


def run_features(cfg):
    setup_logging(cfg["id"] + "_features")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg["dtype"]) if cfg["dtype"] else amp_dtype(device) or torch.float32
    train, val = splits()
    parts = {
        "train": subsample(train, cfg["train_docs"], cfg["seed"]),
        "val": subsample(val, cfg["eval_docs"], cfg["seed"]),
        "test": subsample(load_test(), cfg["eval_docs"], cfg["seed"]),
    }
    with track(cfg["id"] + "_features", cfg) as info:
        tok = AutoTokenizer.from_pretrained(cfg["observer"])
        observer = AutoModelForCausalLM.from_pretrained(cfg["observer"], dtype=dtype).to(device).eval()
        performer = AutoModelForCausalLM.from_pretrained(cfg["performer"], dtype=dtype).to(device).eval()
        for part, df in parts.items():
            lm_features(cfg, df, tok, observer, performer, device).to_parquet(art("features", f"lm_{part}.parquet"), index=False)
            info[f"{part}_docs"] = len(df)
    return info


def lm_frame(part, df):
    return df[["id", "lang", "domain"]].merge(pd.read_parquet(art("features", f"lm_{part}.parquet")), on="id")


def run_classifier(cfg):
    train, val = splits()
    test = load_test()
    cats = {c: sorted(set(train[c])) for c in ["lang", "domain"]}
    frames = {p: lm_frame(p, d) for p, d in [("train", train), ("val", val), ("test", test)]}
    for f in frames.values():
        for c, values in cats.items():
            f[c] = pd.Categorical(f[c], categories=values)
    label = frames["train"][["id"]].merge(train[["id", "label"]], on="id").label
    with track(cfg["id"], cfg):
        model = HistGradientBoostingClassifier(
            **cfg["classifier"], categorical_features="from_dtype", class_weight="balanced", random_state=cfg["seed"]
        )
        model.fit(frames["train"].drop(columns="id"), label)
        for part in ["val", "test"]:
            proba = model.predict_proba(frames[part].drop(columns="id"))
            save_preds(cfg["id"], part, pd.DataFrame(proba, columns=model.classes_)[LABELS].assign(id=frames[part].id.to_numpy()))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lm_features")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    set_smoke(args.smoke)
    print(run_features(load_config(args.config)))
