from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Section2Config:
    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]


def load_section2_config(path: str | Path) -> Section2Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return Section2Config(
        data=cfg.get("data", {}),
        model=cfg.get("model", {}),
        train=cfg.get("train", {}),
    )
