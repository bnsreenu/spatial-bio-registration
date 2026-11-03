"""Project configuration for the spatial_registration tool."""

import os
from pathlib import Path


def dataset_dir(required: bool = True):
    """Dataset root (read only), from the environment variable UNUMLOCALIA_DIR.

    The folder must contain one subfolder per core (core01_hcc, ...)."""
    value = os.environ.get("UNUMLOCALIA_DIR", "").strip()
    if not value:
        if required:
            raise SystemExit(
                "UNUMLOCALIA_DIR is not set. Set it to the folder that contains the core folders "
                "(core01_hcc, core02_non_tumour, ...) of the UnumLocalia dataset."
            )
        return None
    return Path(value).resolve()


# Core ids present in data/UnumLocalia (one subfolder per core).
CORE_IDS = [
    "core01_hcc",
    "core02_non_tumour",
    "core03_tonsil",
    "core04_hca",
]

# This project's root and output folder.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
