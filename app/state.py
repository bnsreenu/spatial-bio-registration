"""Plain-data model of the Samples screen: no Qt imports.

A sample row is {"sample_id", "layers": {modality: {"path", "pixel_size_um",
"channel"}}, "proposed": {modality: note}}. "proposed" marks links that
samples.discover_samples (or a drop) suggested and the user has not yet
touched; editing a cell removes its entry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import samples as sm
import reference as reference_mod
from app import summary

MODALITY_COLUMNS = [("he", "H&E"), ("comet", "COMET"), ("xenium", "Xenium"), ("other", "Other")]
MODALITY_LABEL = dict(MODALITY_COLUMNS)
DEFAULT_TARGET_UM = 1.0


def new_state() -> Dict:
    return {"samples": [], "results": {}, "inputs": {}, "review": {}, "exports": {}, "project_dir": None,
            "settings": {"target_um": DEFAULT_TARGET_UM}}


def make_sample(sample_id: str) -> Dict:
    return {"sample_id": sample_id, "layers": {}, "proposed": {}, "reference": None, "candidates": {}}


def find_sample(state: Dict, sample_id: str) -> Optional[Dict]:
    return next((s for s in state["samples"] if s["sample_id"] == sample_id), None)


def unique_id(state: Dict, base: str) -> str:
    ids = {s["sample_id"] for s in state["samples"]}
    if base not in ids:
        return base
    i = 2
    while f"{base}_{i}" in ids:
        i += 1
    return f"{base}_{i}"


def set_layer(sample: Dict, modality: str, path: "str | Path", proposed_note: Optional[str] = None,
              pixel_size_um: Optional[float] = None, channel: Optional[str] = None) -> None:
    sample["layers"][modality] = {"path": str(path), "pixel_size_um": pixel_size_um, "channel": channel}
    sample.setdefault("candidates", {}).pop(modality, None)
    if proposed_note is None:
        sample["proposed"].pop(modality, None)
    else:
        sample["proposed"][modality] = proposed_note


def candidates_text(modality: str, count: int) -> str:
    """"3 H&E candidates found: pick one"."""
    return f"{count} {MODALITY_LABEL.get(modality, modality)} candidates found: pick one"


def clear_layer(sample: Dict, modality: str) -> None:
    sample["layers"].pop(modality, None)
    sample["proposed"].pop(modality, None)


def folder_proposal(root: "str | Path") -> "tuple[List[str], int]":
    """(one plain line per sample folder found, their count) for a chosen folder of samples; nothing is added."""
    records = sm.discover_samples(root)
    by_sample: Dict[str, List[Dict]] = {}
    for rec in records:
        by_sample.setdefault(rec["sample_id"], []).append(rec)
    lines = []
    for sid, recs in by_sample.items():
        subs = {p.name.lower() for p in (Path(root) / sid).iterdir() if p.is_dir()}
        found = [MODALITY_LABEL[r["modality"]] for r in recs if r["path"]]
        pick = [f"{MODALITY_LABEL[r['modality']]}: {len(r['candidates'])} candidates, you pick one"
                for r in recs if r.get("candidates")]
        missing = [MODALITY_LABEL[r["modality"]] for r in recs if not r["path"] and not r.get("candidates")]
        if not any(m in subs for m in sm.MODALITIES):
            extra = "; ".join(([", ".join(found) + " found by file name"] if found else []) + pick)
            lines.append(f"{sid}: no he, comet or xenium subfolders found" + (f" ({extra})" if extra else "")
                         + "; each sample folder needs these subfolders, see Help")
            continue
        parts = [", ".join(found) if found else "nothing found"] + pick
        if missing:
            parts.append("missing: " + ", ".join(missing))
        lines.append(f"{sid}: " + "; ".join(parts))
    return lines, len(by_sample)


def propose_from_folder(state: Dict, root: "str | Path") -> List[str]:
    """Add one row per sample proposed by samples.discover_samples(root).
    Existing rows with the same id are replaced. Returns the new ids."""
    records = sm.discover_samples(root)
    warnings = {(r["sample_id"], r["modality"]): r.get("warning") or "" for r in records}
    candidates = {(r["sample_id"], r["modality"]): r.get("candidates") or [] for r in records}
    ids = []
    for grouped in sm.group_samples(records):
        sample = make_sample(grouped["sample_id"])
        for modality in sm.MODALITIES:
            found = candidates.get((grouped["sample_id"], modality))
            if found:
                sample["candidates"][modality] = list(found)
        for modality, layer in grouped["layers"].items():
            note = warnings.get((grouped["sample_id"], modality)) or "proposed by folder convention"
            set_layer(sample, modality, layer["path"], proposed_note=note)
        old = find_sample(state, sample["sample_id"])
        if old is not None:
            state["samples"].remove(old)
            state["results"].pop(old["sample_id"], None)
            state["review"].pop(old["sample_id"], None)
        state["samples"].append(sample)
        ids.append(sample["sample_id"])
    return ids


def guess_modality(path: "str | Path") -> str:
    """Column for a file dropped outside any cell, from its content kind
    (never from its name): brightfield RGB -> he, multichannel
    fluorescence -> comet, Xenium folder -> xenium, anything else (or an
    unreadable file) -> other."""
    desc, _err = summary.describe({"path": str(path), "pixel_size_um": None})
    if desc is None:
        return "other"
    return {"rgb_brightfield": "he", "multichannel_fluorescence": "comet", "xenium": "xenium"}.get(desc["kind"], "other")


def blocked_reasons(sample: Dict) -> List[str]:
    """Why this sample cannot be run yet (empty list: it can)."""
    reasons: List[str] = []
    layers = sample["layers"]
    if len(layers) < 2:
        reasons.append("needs at least two linked layers")
    for modality, layer in layers.items():
        label = MODALITY_LABEL.get(modality, modality)
        desc, err = summary.describe(layer)
        if desc is None:
            reasons.append(f"{label}: cannot read ({err})")
        elif desc["kind"] == "xenium":
            if desc.get("xenium_source") == "rendered" and not summary.has_nucleus_outlines(desc["path"]):
                reasons.append(f"{label}: no cell nucleus outlines found")
        elif summary.pixel_size_missing(desc):
            reasons.append(f"{label}: pixel size missing, enter it")
    return reasons


REFERENCE_CHOICES = ["xenium", "comet", "he"]


def loaded_references(sample: Dict) -> List[str]:
    """Modalities that can be the reference (loaded ones only), in the order
    Xenium, COMET, H&E."""
    return [m for m in REFERENCE_CHOICES if m in sample["layers"]]


def chosen_reference(sample: Dict) -> "tuple[Optional[str], str]":
    """(reference modality, one-line reason). The user's choice if it is
    still loaded, else the default (Xenium, else COMET, else H&E)."""
    loaded = loaded_references(sample)
    default, reason = reference_mod.default_reference(loaded)
    choice = sample.get("reference")
    if choice in loaded and choice != default:
        return choice, f"You chose {summary.name(choice)} as the reference."
    return default, reason


def linked_text(sample: Dict) -> str:
    """"H&E, COMET and Xenium" for the layers that are linked."""
    names = [MODALITY_LABEL[m] for m, _l in MODALITY_COLUMNS if m in sample["layers"]]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def runnable_samples(state: Dict) -> List[Dict]:
    return [s for s in state["samples"] if not blocked_reasons(s)]


def to_run_dict(sample: Dict) -> Dict:
    """The dict pipeline.run_sample expects (UI-only keys dropped)."""
    return {
        "sample_id": sample["sample_id"],
        "layers": {m: {"path": l["path"], "pixel_size_um": l.get("pixel_size_um"), "channel": l.get("channel")}
                   for m, l in sample["layers"].items()},
    }


def pixel_sizes_used(sample: Dict) -> Dict:
    out = {}
    for modality, layer in sample["layers"].items():
        desc, _err = summary.describe(layer)
        out[modality] = None if desc is None else desc.get("pixel_size_um")
    return out
