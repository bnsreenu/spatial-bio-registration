"""Task 11, item 2: sample discovery and linking.

discover_samples(root) follows the folder convention already used by
data/UnumLocalia: a sample folder directly under root, with subfolders
named "he", "comet", "xenium" (case-insensitive) holding that modality's
data. The folder name is the sample id.

Within an "he" or "comet" subfolder the actual modality file is not
assumed to be named after the subfolder; it is found by looking for a
single ".ome.zarr" directory, a single OME-TIFF/TIFF file, or a single
SVS/NDPI file inside it (in that order), since this is exactly what
data/UnumLocalia contains (e.g. he/core01_he.ome.zarr, not
he/he.ome.zarr). If a modality subfolder exists but no such file is found,
or more than one candidate is found, this is reported as an ambiguity
rather than guessed. The "xenium" subfolder is linked as a folder (not a
single file), since Xenium output is itself a folder of files.

If a sample folder is missing one of the three modality subfolders, a
fallback search looks anywhere under that sample folder for a file or
folder whose name contains the modality token (case-insensitive); again,
more than one candidate is reported as an ambiguity, not guessed.

discover_samples returns a flat list of record dicts, one per
(sample_id, modality) pair found:

    {"sample_id": ..., "modality": ..., "path": ... or None,
     "pixel_size_um": None, "channel": None, "warning": "", "candidates": []}

"candidates" lists the paths of an ambiguous modality (path is None then),
so a caller can offer them as a choice; it is empty otherwise.

This is the same shape as the manifest CSV (minus the "warning" column,
which load_manifest/save_manifest drop since it is discovery-only
information, not an input the pipeline needs). group_samples converts
this flat list into the nested, per-sample structure pipeline.run_sample
consumes.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

MODALITIES = ("he", "comet", "xenium")

MANIFEST_COLUMNS = ["sample_id", "modality", "path", "pixel_size_um", "channel"]


# ---------------------------------------------------------------------------
# discover_samples
# ---------------------------------------------------------------------------

def _image_candidates(folder: Path) -> List[Path]:
    """The .ome.zarr directories in folder if there are any, else the
    .tif/.tiff/.svs/.ndpi files."""
    zarr_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower().endswith(".ome.zarr"))
    if zarr_dirs:
        return zarr_dirs
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".svs", ".ndpi")
    )


def _find_single_file_in(folder: Path) -> "tuple[Optional[Path], Optional[str]]":
    """Look inside an he/comet subfolder for the one image to use: a
    .ome.zarr directory, else a single .ome.tif(f)/.tif(f) file, else a
    single .svs/.ndpi file. Returns (path_or_None, warning_or_None)."""
    found = _image_candidates(folder)
    if len(found) == 1:
        return found[0], None
    if len(found) > 1:
        kind = ".ome.zarr" if found[0].is_dir() else "image file"
        return None, f"ambiguous: {len(found)} {kind} candidates in {folder}: {[p.name for p in found]}"
    return None, f"no image file (.ome.zarr, .tif(f), .svs, .ndpi) found in {folder}"


def _fallback_candidates(sample_folder: Path, token: str) -> List[Path]:
    """Name-token matches under sample_folder, shallowest first."""
    candidates = [p for p in sample_folder.rglob("*") if token in p.name.lower()]
    candidates.sort(key=lambda p: len(p.relative_to(sample_folder).parts))
    return candidates


def _fallback_search(sample_folder: Path, token: str) -> "tuple[Optional[Path], Optional[str]]":
    """Search recursively under sample_folder for a file or directory whose
    name contains token (case-insensitive). Used only when the expected
    <token>/ subfolder convention is not found directly."""
    # Prefer top-level matches (shortest relative path) if there are several.
    candidates = _fallback_candidates(sample_folder, token)
    if not candidates:
        return None, f"no candidate found for modality {token!r} under {sample_folder}"
    if len(candidates) > 1 and len(candidates[0].relative_to(sample_folder).parts) == len(
        candidates[1].relative_to(sample_folder).parts
    ):
        return None, (
            f"ambiguous: {len(candidates)} candidates for modality {token!r} under {sample_folder}: "
            f"{[str(c.relative_to(sample_folder)) for c in candidates]}"
        )
    return candidates[0], None


def discover_samples(root: "str | Path") -> List[Dict]:
    """Discover samples under root by the he/comet/xenium subfolder
    convention, with a fallback name-token search. Returns a flat list of
    records (see module docstring). Ambiguities and missing modalities are
    reported via the "path": None plus a "warning" string, never guessed.
    """
    root = Path(root)
    records: List[Dict] = []

    for sample_folder in sorted(p for p in root.iterdir() if p.is_dir()):
        sample_id = sample_folder.name
        subfolders = {p.name.lower(): p for p in sample_folder.iterdir() if p.is_dir()}

        for modality in MODALITIES:
            path: Optional[Path] = None
            warning = ""
            candidates: List[str] = []

            if modality in subfolders:
                folder = subfolders[modality]
                if modality == "xenium":
                    path = folder
                else:
                    path, warn = _find_single_file_in(folder)
                    if warn:
                        warning = warn
                    if path is None and warn and warn.startswith("ambiguous"):
                        candidates = [str(p) for p in _image_candidates(folder)]
            else:
                path, warn = _fallback_search(sample_folder, modality)
                if warn:
                    warning = warn
                    if warn.startswith("ambiguous"):
                        found = _fallback_candidates(sample_folder, modality)
                        depth = len(found[0].relative_to(sample_folder).parts)
                        candidates = [str(p) for p in found if len(p.relative_to(sample_folder).parts) == depth]
                elif path is not None:
                    warning = f"found by fallback name-token search, not the {modality}/ subfolder convention"

            records.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "path": str(path) if path is not None else None,
                    "pixel_size_um": None,
                    "channel": None,
                    "warning": warning,
                    "candidates": candidates,
                }
            )

    return records


# ---------------------------------------------------------------------------
# group_samples: flat records -> nested per-sample structure
# ---------------------------------------------------------------------------

def group_samples(records: List[Dict]) -> List[Dict]:
    """Convert a flat list of (sample_id, modality) records (as returned by
    discover_samples or load_manifest) into a nested, per-sample structure:

        {"sample_id": ..., "layers": {modality: {"path", "pixel_size_um",
         "channel"}, ...}}

    "layers" preserves he, comet, xenium insertion order (the fixed order
    used throughout this module) regardless of the input record order, so
    pipeline.run_sample's "first listed layer" fallback reference choice is
    deterministic. Records with path=None are omitted from "layers" (that
    modality is simply absent for this sample).
    """
    by_sample: Dict[str, Dict[str, Dict]] = {}
    for rec in records:
        by_sample.setdefault(rec["sample_id"], {})
        if rec.get("path"):
            by_sample[rec["sample_id"]][rec["modality"]] = {
                "path": rec["path"],
                "pixel_size_um": rec.get("pixel_size_um"),
                "channel": rec.get("channel"),
            }

    samples = []
    for sample_id, layers_by_modality in by_sample.items():
        ordered_layers = {m: layers_by_modality[m] for m in MODALITIES if m in layers_by_modality}
        samples.append({"sample_id": sample_id, "layers": ordered_layers})
    return samples


# ---------------------------------------------------------------------------
# manifest I/O
# ---------------------------------------------------------------------------

def save_manifest(samples: List[Dict], csv_path: "str | Path") -> None:
    """Write a flat list of records (discover_samples's format; "warning"
    is dropped if present, since it is discovery-only, not a manifest
    input) to csv_path with columns sample_id, modality, path,
    pixel_size_um, channel. None values are written as empty strings."""
    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for rec in samples:
            writer.writerow(
                {
                    "sample_id": rec["sample_id"],
                    "modality": rec["modality"],
                    "path": rec.get("path") if rec.get("path") is not None else "",
                    "pixel_size_um": rec.get("pixel_size_um") if rec.get("pixel_size_um") is not None else "",
                    "channel": rec.get("channel") if rec.get("channel") is not None else "",
                }
            )


def load_manifest(csv_path: "str | Path") -> List[Dict]:
    """Read a manifest CSV back into the flat record list format. Empty
    strings are read back as None (sample_id/modality/path/pixel_size_um/
    channel), since otherwise pandas-free csv.DictReader would keep them as
    empty strings rather than round-tripping to the same None values
    save_manifest was given."""
    csv_path = Path(csv_path)
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path") or None
            pixel_size_raw = row.get("pixel_size_um") or None
            channel = row.get("channel") or None
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "modality": row["modality"],
                    "path": path,
                    "pixel_size_um": float(pixel_size_raw) if pixel_size_raw is not None else None,
                    "channel": channel,
                    "warning": "",
                }
            )
    return records
