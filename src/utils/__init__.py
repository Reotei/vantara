"""Shared helpers: configuration, logging and artifact I/O."""

import json
import logging
import os
import random
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from pythonjsonlogger.json import JsonFormatter

ROOT = Path(__file__).resolve().parents[2]


@cache
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config; ARTIFACTS_DIR / DATA_DIR / FIGURES_DIR env vars override paths."""
    cfg = yaml.safe_load(Path(path or os.getenv("VANTARA_CONFIG", ROOT / "config/config.yaml")).read_text())
    for key, env in [("artifacts", "ARTIFACTS_DIR"), ("figures", "FIGURES_DIR"), ("processed", "PROCESSED_DIR")]:
        cfg["paths"][key] = os.getenv(env, cfg["paths"][key])
    for key, value in cfg["paths"].items():
        if key != "raw_url" and not Path(value).is_absolute():
            cfg["paths"][key] = str(ROOT / value)
    return cfg


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-structured logger at LOG_LEVEL (default INFO)."""
    root = logging.getLogger()
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def artifact(name: str, cfg: dict[str, Any] | None = None) -> Path:
    """Path of a named file in the artifacts directory (created on demand)."""
    folder = Path((cfg or load_config())["paths"]["artifacts"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def save(obj: Any, name: str, cfg: dict[str, Any] | None = None) -> None:
    """Persist an object as JSON (for .json names) or joblib."""
    path = artifact(name, cfg)
    if name.endswith(".json"):
        path.write_text(json.dumps(obj, indent=2, default=float))
    else:
        joblib.dump(obj, path)


def load(name: str, cfg: dict[str, Any] | None = None) -> Any:
    """Load an artifact saved with :func:`save`."""
    path = artifact(name, cfg)
    return json.loads(path.read_text()) if name.endswith(".json") else joblib.load(path)
