import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def mean_ci95(values) -> tuple:
    """Mean and 95% confidence half-width (normal approximation) over episodes."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return mean, ci


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
