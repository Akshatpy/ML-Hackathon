from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = PROJECT_ROOT / "configs" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = PROJECT_ROOT / rel
    return cfg
