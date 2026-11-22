import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPETITION = "ai-generated-text-detection-hw3-ucu-ai-2026"
SMOKE = os.environ.get("HW3_SMOKE") == "1"


def set_smoke(flag):
    global SMOKE
    SMOKE = flag


def data_dir():
    if "HW3_DATA" in os.environ:
        return Path(os.environ["HW3_DATA"])
    mounted = [Path("/kaggle/input") / COMPETITION, Path("/kaggle/input/competitions") / COMPETITION]
    return next((p for p in mounted if p.exists()), ROOT / "data")


def art(*parts):
    base = Path(os.environ.get("HW3_ARTIFACTS", ROOT / "artifacts"))
    path = base.joinpath("smoke" if SMOKE else "", *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_config(name):
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    smoke = cfg.pop("smoke", {})
    if SMOKE:
        cfg.update(smoke)
    return cfg


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
