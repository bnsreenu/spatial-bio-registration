"""Remembered settings between launches (task 19): a small JSON file, as plain functions.

The file location can be overridden with the environment variable SPATIAL_REGISTRATION_CONFIG (used by tests,
so they never touch the real settings)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


def config_path() -> Path:
    override = os.environ.get("SPATIAL_REGISTRATION_CONFIG")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "spatial_registration" / "settings.json"


def load_config() -> Dict:
    try:
        with open(config_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(values: Dict) -> bool:
    """Merge values into the stored settings. False if the file could not be written (the app still works)."""
    data = load_config()
    data.update(values)
    try:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def remembered_output_folder() -> Optional[str]:
    value = load_config().get("output_folder")
    return value if isinstance(value, str) and value else None


def remember_output_folder(folder: str) -> bool:
    return save_config({"output_folder": str(folder)})
