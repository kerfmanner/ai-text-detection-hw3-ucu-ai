import subprocess
import zipfile

import pandas as pd

from .config import COMPETITION, ROOT, data_dir

LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SPLIT_PATH = ROOT / "splits" / "split.csv"


def download():
    target = data_dir()
    if (target / "train.csv").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(target), "-q"], check=True)
    with zipfile.ZipFile(target / f"{COMPETITION}.zip") as z:
        z.extractall(target)
    return target


def load_train():
    df = pd.read_csv(data_dir() / "train.csv")
    if SPLIT_PATH.exists():
        df = df.merge(pd.read_csv(SPLIT_PATH), on="id")
    return df


def load_test():
    return pd.read_csv(data_dir() / "test.csv")


def load_sample_submission():
    return pd.read_csv(data_dir() / "sample_submission.csv")


def splits():
    df = load_train()
    return df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)


def subsample(df, n, seed=42):
    if not n or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
