"""Text and checks shown in the side panel and the review header: no Qt."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import engine
import deformation
import io_layers
import pipeline
from app import errors

LEVEL_NOMINAL_TOLERANCE = 2e-4  # 0.02 percent: technical line in Details
LEVEL_INFO_THRESHOLD = 1e-3  # 0.1 percent: neutral note on the card
MODALITY_NAMES = {"he": "H&E", "comet": "COMET", "xenium": "Xenium", "other": "Other"}
XENIUM_NO_MORPHOLOGY = "No Xenium morphology image found; nuclei were drawn from the cell nucleus outlines instead."

_cache: Dict[tuple, Tuple[Optional[Dict], str]] = {}
_raw_errors: Dict[tuple, str] = {}


def name(modality: str) -> str:
    return MODALITY_NAMES.get(modality, str(modality))


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe(layer: Dict) -> Tuple[Optional[Dict], str]:
    """(descriptor, "") or (None, plain message) for a sample layer, cached.
    Uses pipeline.describe_layer, so a user-entered pixel size is applied
    exactly as run_sample will apply it. The exception text is logged and
    available from technical_error()."""
    key = (str(layer["path"]), layer.get("pixel_size_um"))
    if key not in _cache:
        try:
            _cache[key] = (pipeline.describe_layer(layer), "")
        except Exception as exc:  # noqa: BLE001 - logged, plain message shown
            _raw_errors[key] = errors.log_exception(f"describe {layer['path']}", exc)
            _cache[key] = (None, errors.INPUT_UNREADABLE)
    return _cache[key]


def technical_error(layer: Dict) -> str:
    return _raw_errors.get((str(layer["path"]), layer.get("pixel_size_um")), "")


def clear_cache() -> None:
    _cache.clear()
    _raw_errors.clear()
    _thumbs.clear()


def pixel_size_missing(desc: Dict) -> bool:
    return not io_layers._plausible(desc.get("pixel_size_um"))


def _level_deviations(desc: Dict) -> List[Tuple[Dict, float, float]]:
    """(level, nominal um/px, relative deviation) for every level after level 0
    whose pixel size is known. Nominal = level 0 times the nearest power of two."""
    known = [l for l in (desc.get("shape_per_level") or []) if l.get("um_per_px")]
    if len(known) < 2:
        return []
    um0 = known[0]["um_per_px"]
    out = []
    for lvl in known[1:]:
        ratio = lvl["um_per_px"] / um0
        if ratio <= 0:
            continue
        nominal = um0 * 2.0 ** round(math.log2(ratio))
        out.append((lvl, nominal, abs(lvl["um_per_px"] / nominal - 1.0)))
    return out


def level_scale_note(desc: Optional[Dict]) -> Optional[str]:
    """A neutral info note, only when a level's derived pixel size is more than
    0.1 percent off the nominal value. Below that nothing is shown on the card
    (the exact figures stay in Details)."""
    if desc is None:
        return None
    worst = max((rel for _l, _n, rel in _level_deviations(desc)), default=0.0)
    if worst <= LEVEL_INFO_THRESHOLD:
        return None
    um = math.floor(worst * 1000.0 * 10.0 + 0.5) / 10.0
    return (f"The pixel size of a lower-resolution level differs slightly from the ideal "
            f"({100 * worst:.3g} percent, about {um:.1f} um over 1 mm). This is common in converted files; "
            "no action needed.")


def level_warnings(desc: Dict) -> List[str]:
    """Technical lines (Details only) for every pyramid level whose pixel size
    differs from the nominal power-of-two multiple of level 0 by more than
    0.02 percent."""
    return [
        f"level {lvl['level']}: derived pixel size {lvl['um_per_px']:.6g} um/px differs from the "
        f"nominal {nominal:.6g} by {rel * 100:.3f}% (limit 0.02%)"
        for lvl, nominal, rel in _level_deviations(desc) if rel > LEVEL_NOMINAL_TOLERANCE
    ]


def pixel_dims(desc: Dict) -> Optional[Tuple[int, int]]:
    """(width, height) in pixels of level 0, from its shape: the two largest
    axes in order (rows, columns), which skips channel and singleton axes."""
    levels = desc.get("shape_per_level")
    if not levels:
        return None
    shape = [int(s) for s in levels[0]["shape"]]
    big = sorted(range(len(shape)), key=lambda i: -shape[i])[:2]
    big.sort()
    if len(big) < 2:
        return None
    return shape[big[1]], shape[big[0]]


def describe_text(desc: Optional[Dict], err: str = "") -> str:
    """Short plain summary for the side panel (details go in details_text)."""
    if desc is None:
        return err or errors.INPUT_UNREADABLE
    lines = []
    dims = pixel_dims(desc)
    if dims:
        lines.append(f"Size: {dims[0]} x {dims[1]} pixels")
    px = desc.get("pixel_size_um")
    if desc["kind"] == "xenium":
        lines.append("Pixel size: not applicable (nuclei are drawn at the working resolution)")
    elif pixel_size_missing(desc):
        lines.append("Pixel size: MISSING" if px is None else f"Pixel size: {px:g} um/px (not plausible)")
    else:
        lines.append(f"Pixel size: {px:.4g} um/px")
    names = [n for n in (desc.get("channel_names") or []) if n]
    if names:
        lines.append(f"Channels ({len(names)}): " + ", ".join(str(n) for n in names))
    elif desc["kind"] == "rgb_brightfield":
        lines.append("Channels: RGB color")
    if desc.get("shape_per_level"):
        lines.append(f"Pyramid levels: {len(desc['shape_per_level'])}")
    notice = xenium_notice(desc)
    if notice:
        lines += ["", notice]
    return "\n".join(lines)


def details_text(desc: Optional[Dict], layer: Optional[Dict] = None) -> str:
    """Technical detail for a collapsed Details section."""
    if desc is None:
        return (technical_error(layer) if layer else "") or "No technical detail recorded."
    lines = [f"Path: {desc['path']}", f"Kind: {desc['kind']}   Format: {desc['format']}"]
    for lvl in desc.get("shape_per_level") or []:
        um = lvl.get("um_per_px")
        lines.append(f"  level {lvl['level']}: shape {tuple(lvl['shape'])}  "
                     + ("pixel size unknown" if um is None else f"{um:.8g} um/px"))
    for w in desc.get("warnings") or []:
        lines.append("note: " + w)
    return "\n".join(lines)


def xenium_notice(desc: Optional[Dict]) -> Optional[str]:
    if desc and desc.get("kind") == "xenium" and desc.get("xenium_source") == "rendered":
        return XENIUM_NO_MORPHOLOGY
    return None


XENIUM_NO_IMAGE_CARD = "No Xenium image found. That is fine: nuclei will be drawn from the cell nucleus outlines."
XENIUM_NO_OUTLINES = ("No cell nucleus outlines were found in this folder, so nuclei cannot be drawn. Choose the Xenium "
                      "output folder that contains them (nucleus_boundaries.parquet).")
PIXEL_SIZE_MISSING_CARD = "The pixel size is missing, so this sample cannot run yet. Enter it below. It is never guessed."
CARD_UNREADABLE = errors.INPUT_UNREADABLE + " Check that it still exists and is the right kind of image, or choose another."
XENIUM_FILE_LABELS = [
    ("cells.csv", "Cells"),
    ("cell_boundaries.parquet", "Cell outlines"),
    ("nucleus_boundaries.parquet", "Nucleus outlines"),
    ("cell_boundaries_comet_space.geojson", "Cell outlines in COMET space"),
    ("transcripts.parquet", "Transcripts (not used)"),
]


def xenium_files(path) -> List[str]:
    """Plain lines for the files found in a Xenium output folder, such as
    "Cells (cells.csv)". A morphology image, if present, is listed first."""
    folder = Path(path)
    if not folder.is_dir():
        return []
    names = {p.name.lower(): p.name for p in folder.iterdir() if not p.name.startswith(".")}
    lines = []
    for low, real in names.items():
        if low.startswith("morphology") and (".tif" in low or low.endswith(".zarr")):
            lines.append(f"Morphology image ({real})")
    for fname, label in XENIUM_FILE_LABELS:
        if fname in names:
            lines.append(f"{label} ({names[fname]})")
    return lines


def has_nucleus_outlines(path) -> bool:
    return (Path(path) / "nucleus_boundaries.parquet").exists()


def card_messages(desc: Optional[Dict], err: str, layer: Dict) -> Tuple[List[str], List[str]]:
    """(blocking, warnings): plain sentences for a modality card, with what to do."""
    if desc is None:
        return [CARD_UNREADABLE], []
    blocking: List[str] = []
    if desc["kind"] == "xenium":
        if desc.get("xenium_source") == "rendered" and not has_nucleus_outlines(desc["path"]):
            blocking.append(XENIUM_NO_OUTLINES)
    elif pixel_size_missing(desc):
        blocking.append(PIXEL_SIZE_MISSING_CARD)
    plain, _all = split_notes(descriptor_warnings(desc))
    warnings = [p for p in plain if p != XENIUM_NO_MORPHOLOGY]
    return blocking, warnings


def facts_line(desc: Dict) -> str:
    """One line: "4116 x 4088 pixels, 0.28 um/px, 16 channels, 6 levels"."""
    parts = []
    dims = pixel_dims(desc)
    if dims:
        parts.append(f"{dims[0]} x {dims[1]} pixels")
    if desc["kind"] != "xenium":
        px = desc.get("pixel_size_um")
        parts.append("pixel size missing" if pixel_size_missing(desc) else f"{px:.4g} um/px")
    names = [n for n in (desc.get("channel_names") or []) if n]
    if names:
        parts.append(plural(len(names), "channel"))
    elif desc["kind"] == "rgb_brightfield":
        parts.append("RGB color")
    if desc.get("shape_per_level"):
        parts.append(plural(len(desc["shape_per_level"]), "level"))
    return ", ".join(parts)


def level_table(desc: Optional[Dict]) -> str:
    """Path and level table for the card's Details section."""
    if desc is None:
        return ""
    lines = [f"Path: {desc['path']}"]
    for lvl in desc.get("shape_per_level") or []:
        um = lvl.get("um_per_px")
        lines.append(f"Level {lvl['level']}: {tuple(lvl['shape'])}, " + ("pixel size unknown" if um is None else f"{um:.6g} um/px"))
    lines += level_warnings(desc)
    return "\n".join(lines)


_thumbs: Dict[tuple, Optional[np.ndarray]] = {}


def thumbnail(desc: Optional[Dict], layer: Dict) -> Optional[np.ndarray]:
    """Small uint8 HxWx3 thumbnail read from the lowest-resolution pyramid
    level (None for Xenium folders, or when it cannot be read). H&E is shown
    in color for zarr and TIFF inputs, everything else in grayscale."""
    if desc is None or desc["kind"] == "xenium":
        return None
    key = (str(layer["path"]), layer.get("pixel_size_um"), layer.get("channel"))
    if key not in _thumbs:
        try:
            _thumbs[key] = _make_thumbnail(desc, layer)
        except Exception as exc:  # noqa: BLE001 - a missing thumbnail is not an error for the user
            errors.log_exception(f"thumbnail {layer['path']}", exc)
            _thumbs[key] = None
    return _thumbs[key]


def _make_thumbnail(desc: Dict, layer: Dict) -> Optional[np.ndarray]:
    import review_overlay as ro
    ums = [lvl["um_per_px"] for lvl in desc.get("shape_per_level") or [] if lvl.get("um_per_px")]
    if not ums:
        return None
    um = max(ums)
    rgb = ro._open_rgb(desc, um) if desc["kind"] == "rgb_brightfield" else None
    if rgb is not None:
        return rgb
    img, _ach = io_layers.open_layer(desc, um, channel=layer.get("channel"))
    gray = ro.to_uint8(img)
    return np.stack([gray] * 3, axis=-1)


def plain_note(text: str) -> Optional[str]:
    """Plain-language version of a known engine or reader note, or None when
    the note is technical (it then goes to Details only)."""
    low = text.lower()
    if "no morphology image" in low:
        return XENIUM_NO_MORPHOLOGY
    if "entered by the user" in low or "entered by you" in low:
        return "The pixel size was entered by you; it was not read from the file."
    if low.startswith("level ") and "nominal" in low:
        return None  # a small level-scale difference is shown as an info note on the card, details in Details
    return None


def split_notes(notes: List[str]) -> Tuple[List[str], List[str]]:
    """(plain notes without duplicates, all raw notes) for main text and Details."""
    plain: List[str] = []
    for n in notes:
        p = plain_note(n)
        if p and p not in plain:
            plain.append(p)
    return plain, list(dict.fromkeys(notes))


def descriptor_warnings(desc: Optional[Dict]) -> List[str]:
    if desc is None:
        return []
    return list(desc.get("warnings") or []) + level_warnings(desc)


def _level(step: Dict) -> str:
    return step.get("confidence") or "low"


def step_for(result: Dict, modality: str) -> Optional[Dict]:
    return next((d for d in result.get("diagnostics", []) if d["step"].startswith(f"{modality}_to_")), None)


def step_path(result: Dict, modality: str) -> List[Dict]:
    """Engine steps (not derived ones) that carry `modality` to the engine's
    own reference, in order."""
    base = result.get("base_reference", result.get("reference_modality"))
    steps = [d for d in result.get("diagnostics", []) if not d.get("derived")]
    path: List[Dict] = []
    current, seen = modality, set()
    while current != base and current not in seen:
        seen.add(current)
        step = next((d for d in steps if d["step"].startswith(f"{current}_to_")), None)
        if step is None:
            break
        path.append(step)
        current = step["step"].split("_to_", 1)[1]
    return path


def layer_steps(result: Dict, modality: str, reference: Optional[str]) -> List[Dict]:
    """The engine steps that matter for showing `modality` on `reference`: the
    steps from the layer to the engine's reference plus those from `reference`
    to it, with the shared tail left out (it cancels)."""
    mine = step_path(result, modality)
    theirs = step_path(result, reference) if reference else []
    mine_names, their_names = {s["step"] for s in mine}, {s["step"] for s in theirs}
    steps = [s for s in mine if s["step"] not in their_names] + [s for s in theirs if s["step"] not in mine_names]
    if not steps:
        own = step_for(result, modality)
        steps = [own] if own else []
    return steps


def steps_level(result: Dict, steps: List[Dict]) -> str:
    """The weakest confidence among the steps (low < medium < high)."""
    order = {"low": 0, "medium": 1, "high": 2}
    if not steps:
        return result.get("confidence") or "low"
    return min((_level(s) for s in steps), key=lambda lv: order.get(lv, 0))


def steps_text(steps: List[Dict]) -> str:
    """One plain line per step, e.g. "COMET to Xenium: 1591 matching points, 92 percent agree. ..."."""
    return "\n".join(step_sentence(s, s["step"].split("_to_", 1)[1]) for s in steps)


def deformation_level(steps: List[Dict]) -> Optional[str]:
    """The deformation check outcome for these steps (None when not available)."""
    return deformation.sample_check(steps) if steps else None


def capped_level(level: str, steps: List[Dict]) -> str:
    check = deformation_level(steps)
    return pipeline.cap_confidence(level, check) if check else level


def deformation_reason(level: str, capped: str, layer: str, reference: str) -> str:
    """Card headline: the usual one, or the capped wording when the check capped a high match."""
    if level == "high" and capped != level:
        return "High match confidence, but possible tissue deformation. Check the overlay."
    return layer_headline(capped, layer, reference)


def layer_headline(level: str, layer: str, reference: str) -> str:
    support = {"high": "is well supported.", "medium": "is reasonably supported; look at it closely.",
               "low": "is weakly supported."}.get(level, "has no result.")
    return f"Confidence: {level}. {name(layer)} onto {name(reference)} {support}"


def step_sentence(step: Dict, reference: str) -> str:
    """e.g. "COMET to Xenium: 1591 matching points, 92 percent agree. A second
    method differs by 0.06 um."."""
    src, dst = step["step"].split("_to_", 1)
    text = f"{name(src)} to {name(dst)}: {plural(int(step.get('n_inliers') or 0), 'matching point')}"
    ratio = step.get("inlier_ratio")
    if ratio is not None and ratio == ratio:
        text += f", {round(100 * ratio)} percent agree"
    text += "."
    if step.get("mirrored"):
        text += " The match looks flipped (mirrored)."
    diff = step.get("second_opinion_diff_mean_um")
    if diff is not None:
        text += f" A second method differs by {diff:.2f} um."
    if dst != reference:
        text += f" {name(src)} is placed onto {name(reference)} through {name(dst)}."
    return text


def layer_level(result: Dict, modality: str) -> str:
    step = step_for(result, modality)
    return _level(step) if step else (result.get("confidence") or "low")


def layer_text(result: Dict, modality: str) -> str:
    """Plain text for one layer: its confidence and why, plus advice if low."""
    if result.get("status") != "ok":
        return failure_text(result)
    step = step_for(result, modality)
    level = layer_level(result, modality)
    text = f"Confidence: {level}. " + (step_sentence(step, result["reference_modality"]) if step else "")
    if level == "low":
        text += " " + errors.LOW_ADVICE
    return text.strip()


def overall_text(result: Dict) -> str:
    if result.get("status") != "ok":
        return {"cancelled": "Overall: cancelled before this sample finished.",
                "failed": "Overall: registration failed."}.get(result.get("status"), "Overall: no result.")
    return f"Overall sample confidence: {result.get('confidence')}"


def failure_text(result: Dict) -> str:
    if result.get("status") == "cancelled":
        return "This sample was cancelled before it finished."
    return (errors.RUN_FAILED + " Check that the right images are linked and that the files still open, "
            "then run again. " + "Technical detail is in the Details section.")


def confidence_rule_text(result: Dict) -> str:
    """Technical detail: thresholds, step names, second-method figures."""
    lines = [f"status: {result.get('status')}   overall confidence: {result.get('confidence')}",
             f"reference: {result.get('reference_modality')}",
             f"rule: high needs at least {engine.HIGH_MIN_INLIERS} inliers and ratio {engine.HIGH_MIN_RATIO}; "
             f"low is under {engine.MEDIUM_MIN_INLIERS} inliers or ratio under {engine.LOW_MAX_RATIO}"]
    for d in result.get("diagnostics", []):
        lines.append(f"step {d['step']}: inliers {d.get('n_inliers')}, ratio {d.get('inlier_ratio')}, "
                     f"confidence {d.get('confidence')}, mirrored {d.get('mirrored')}, "
                     f"second method {d.get('second_opinion_method')} diff {d.get('second_opinion_diff_mean_um')} um")
        lines += ["  note: " + w for w in d.get("warnings") or []]
    lines += ["note: " + w for w in result.get("warnings") or []]
    if result.get("error"):
        lines += ["", str(result["error"])]
    return "\n".join(lines)

