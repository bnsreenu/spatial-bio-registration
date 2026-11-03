"""Task 11, item 4 (extended in task 12): saving/loading a session, and
exporting a pipeline result.

save_session/load_session persist everything needed to reopen a session:
the input descriptors used, pixel sizes, the pipeline.run_sample result,
any landmarks (if the caller supplies them, e.g. from src/landmarks.py),
and the software version string (SOFTWARE_VERSION below).

Storage choice for the matched inlier points (task 12): they are stored
inside the session JSON, under result["diagnostics"][i] as nested lists
(src_pts, dst_pts as [[x, y], ...], match_ids as [i, ...]), not in a
sidecar file. Python's json writes floats with their shortest exact
round-trip representation, so load_session restores the float64 arrays
bit for bit. A session stays a single self-contained file; the cost is
file size (a few hundred to a few thousand points per step, tens to a few
hundred KB per sample). load_session converts them back to numpy arrays;
sessions written before task 12 simply have no points (empty arrays).

export_result(result, folder) writes, for each registered layer in a
pipeline.run_sample result, a Xenium Explorer style
"{modality}_imagealignment.csv" (3x3, row-major, no header; reuses
export.write_imagealignment_csv), a "{modality}_keypoints.csv" (columns
fixedX, fixedY, alignmentX, alignmentY; about 40 points thinned with
landmarks.suggest_landmarks from that layer's matched inliers, fixed
computed as matrix @ alignment, the same convention as task 9's
export.export_core), plus a single params.json summarizing the run
(reference modality, reference pixel size, per-layer status, confidence,
warnings).

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
No UI imports are used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from export import write_imagealignment_csv, write_keypoints_csv
from landmarks import suggest_landmarks
from reference import apply_reference

from version import VERSION as SOFTWARE_VERSION

N_KEYPOINTS_EXPORT = 40


# ---------------------------------------------------------------------------
# JSON (de)serialization helpers
# ---------------------------------------------------------------------------

def _matrix_to_list(m: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if m is None:
        return None
    return np.asarray(m, dtype=float).tolist()


def _matrix_from_list(data: Optional[List[List[float]]]) -> Optional[np.ndarray]:
    if data is None:
        return None
    return np.array(data, dtype=float)


def _result_to_jsonable(result: Dict) -> Dict:
    """Convert a pipeline.run_sample result dict (numpy matrices, possibly
    NaN/inf floats in diagnostics) into a plain JSON-serializable dict."""
    out = dict(result)
    out["matrices"] = {k: _matrix_to_list(v) for k, v in result.get("matrices", {}).items()}
    out["diagnostics"] = json.loads(
        json.dumps(result.get("diagnostics", []), default=_json_default)
    )
    return out


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _result_from_jsonable(data: Dict) -> Dict:
    out = dict(data)
    out["matrices"] = {k: _matrix_from_list(v) for k, v in data.get("matrices", {}).items()}
    diagnostics = []
    for diag in data.get("diagnostics", []):
        diag = dict(diag)
        diag["src_pts"] = np.array(diag.get("src_pts") or [], dtype=np.float64).reshape(-1, 2)
        diag["dst_pts"] = np.array(diag.get("dst_pts") or [], dtype=np.float64).reshape(-1, 2)
        diag["match_ids"] = np.array(diag.get("match_ids") or [], dtype=np.int64)
        diagnostics.append(diag)
    out["diagnostics"] = diagnostics
    return out


# ---------------------------------------------------------------------------
# save_session / load_session
# ---------------------------------------------------------------------------

def save_session(
    path: "str | Path",
    inputs: Dict,
    pixel_sizes: Dict,
    result: Dict,
    landmarks: Optional[Dict] = None,
    reference: Optional[str] = None,
    review: Optional[Dict] = None,
    export: Optional[Dict] = None,
) -> None:
    """Write a session JSON file: inputs (e.g. sample["layers"]), pixel_sizes
    (per modality, as actually used), result (a pipeline.run_sample dict),
    landmarks (optional, e.g. a src/landmarks.py fit_from_landmarks result,
    or None), the chosen reference modality (task 16A; defaults to the
    result's own reference), and SOFTWARE_VERSION. The result is stored as
    the engine produced it; the chosen reference is applied on export. `export` ({"time", "folder"} or None)
    records when and where the sample was last exported (task 19)."""
    path = Path(path)
    session = {
        "software_version": SOFTWARE_VERSION,
        "inputs": inputs,
        "pixel_sizes": pixel_sizes,
        "reference": reference or result.get("reference_modality"),
        "result": _result_to_jsonable(result),
        "landmarks": landmarks,
        "review": review,
        "export": export,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, default=_json_default)


def load_session(path: "str | Path") -> Dict:
    """Read a session JSON file back, restoring "result"["matrices"] to
    numpy arrays. Returns the same shape save_session was given: {
    software_version, inputs, pixel_sizes, result, landmarks}."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        session = json.load(f)
    session["result"] = _result_from_jsonable(session["result"])
    session.setdefault("reference", session["result"].get("reference_modality"))
    session.setdefault("review", None)
    session.setdefault("export", None)
    return session


# ---------------------------------------------------------------------------
# export_result
# ---------------------------------------------------------------------------

def _with_reference(result: Dict, reference: Optional[str], pixel_sizes: Optional[Dict]) -> Dict:
    """result expressed against `reference` (None: as the engine produced it)."""
    return result if not reference else apply_reference(result, reference, pixel_sizes)


def _file_names(modality: str, reference: str) -> "tuple[str, str]":
    """(matrix file, keypoints file). The Xenium Explorer names are used only
    when Xenium is the reference, because Explorer expects fixed = Xenium DAPI."""
    if reference == "xenium":
        return f"{modality}_imagealignment.csv", f"{modality}_keypoints.csv"
    return f"{modality}_to_{reference}_matrix.csv", f"{modality}_to_{reference}_keypoints.csv"


def export_notes(result: Dict, reference: Optional[str] = None, pixel_sizes: Optional[Dict] = None) -> List[str]:
    """Plain-language notes to show next to the file list (task 16A)."""
    ref = (_with_reference(result, reference, pixel_sizes)).get("reference_modality")
    if ref == "xenium":
        return []
    return [f"No Xenium Explorer alignment file is written, because Xenium is not the reference "
            f"(Explorer needs the Xenium DAPI image as the fixed image). The matrix files map each "
            f"layer into the pixel frame of {ref}."]


def plan_export(result: Dict, folder: "str | Path", reference: Optional[str] = None,
                pixel_sizes: Optional[Dict] = None) -> List[Dict]:
    """Task 14: the files export_result(result, folder, reference) would write,
    without writing anything. Returns a list of {"file": Path, "exists": bool,
    "what": str} using the same rules as export_result (a matrix CSV per
    layer with a matrix, a keypoints CSV when the layer's step has matched
    points, and params.json)."""
    folder = Path(folder)
    result = _with_reference(result, reference, pixel_sizes)
    ref = result.get("reference_modality")
    steps = {d["step"]: d for d in result.get("diagnostics", [])}
    planned: List[Dict] = []
    for modality, matrix in result.get("matrices", {}).items():
        if matrix is None:
            continue
        matrix_name, keypoints_name = _file_names(modality, ref)
        planned.append({"file": folder / matrix_name, "what": f"{modality} alignment matrix"})
        step = next((d for name, d in steps.items() if name.startswith(f"{modality}_to_")), None)
        pts = None if step is None else step.get("src_pts")
        if pts is not None and len(pts) > 0:
            planned.append({"file": folder / keypoints_name, "what": f"{modality} keypoints"})
    planned.append({"file": folder / "params.json", "what": "run parameters, confidence and warnings"})
    for item in planned:
        item["exists"] = item["file"].exists()
    return planned


def export_result(result: Dict, folder: "str | Path", reference: Optional[str] = None,
                  pixel_sizes: Optional[Dict] = None) -> Dict[str, str]:
    """Write a Xenium Explorer style "{modality}_imagealignment.csv" and a
    "{modality}_keypoints.csv" for each layer in result["matrices"] that
    has a matrix, and a single params.json with the run's reference
    modality, reference pixel size, per-layer status/confidence/warnings,
    and SOFTWARE_VERSION.

    Keypoints: the inlier points of the step named "{modality}_to_*" are
    thinned to about N_KEYPOINTS_EXPORT with landmarks.suggest_landmarks
    (on the layer's own side), alignmentX/Y are those points in the
    layer's native pixels and fixedX/Y = matrix @ alignment (reference
    frame), as in export.write_keypoints_csv. If a result has no points
    for a layer (for example a session saved before task 12), only the
    alignment CSV is written and the status says so.

    Returns {modality: status_string}, e.g. "ok, exported, 48 keypoints" or
    "skipped: no matrix (registration failed)"."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = _with_reference(result, reference, pixel_sizes)
    ref = result.get("reference_modality")

    diagnostics_by_step = {d["step"]: d for d in result.get("diagnostics", [])}

    status: Dict[str, str] = {}
    for modality, matrix in result.get("matrices", {}).items():
        if matrix is None:
            status[modality] = "skipped: no matrix (registration failed)"
            continue
        matrix = np.asarray(matrix, dtype=float)
        matrix_name, keypoints_name = _file_names(modality, ref)
        write_imagealignment_csv(matrix, folder / matrix_name)

        step = next(
            (d for name, d in diagnostics_by_step.items() if name.startswith(f"{modality}_to_")), None
        )
        src = None if step is None else step.get("src_pts")
        dst = None if step is None else step.get("dst_pts")
        if src is None or dst is None or len(src) == 0:
            status[modality] = "ok, exported (no matched points available, keypoints CSV not written)"
            continue
        src_sel, _dst_sel, _idx = suggest_landmarks(src, dst, n=N_KEYPOINTS_EXPORT)
        write_keypoints_csv(src_sel, matrix, folder / keypoints_name)
        status[modality] = f"ok, exported, {len(src_sel)} keypoints"

    params = {
        "software_version": SOFTWARE_VERSION,
        "sample_id": result.get("sample_id"),
        "reference_modality": result.get("reference_modality"),
        "reference_pixel_size_um": result.get("reference_pixel_size_um"),
        "engine_reference_modality": result.get("base_reference", result.get("reference_modality")),
        "xenium_explorer_alignment_written": ref == "xenium",
        "overall_confidence": result.get("confidence"),
        "status": result.get("status"),
        "warnings": result.get("warnings", []),
        "steps": {
            step: {
                "status": diag.get("status"),
                "n_inliers": diag.get("n_inliers"),
                "inlier_ratio": diag.get("inlier_ratio"),
                "confidence": diag.get("confidence"),
                "mirrored": diag.get("mirrored"),
            }
            for step, diag in diagnostics_by_step.items()
        },
        "export_status": status,
        "note": (
            "keypoints are a spatially thinned subset of the automatic RANSAC inliers "
            "(landmarks.suggest_landmarks); fixedX/fixedY are the exported matrix applied "
            "to alignmentX/alignmentY. This export has not been tested by importing it "
            "into Xenium Explorer."
        ),
    }
    with open(folder / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=_json_default)

    return status


# ---------------------------------------------------------------------------
# batch_summary.csv (task 17)
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = ["sample", "confidence", "inliers_weakest_step", "inlier_ratio_weakest_step",
                   "sift_diff_um_worst_step", "warnings", "reference", "review_status", "export_time",
                   "tool_version"]


def not_run_row(sample_id: str, status_text: str, reference: Optional[str], tool_version: str) -> Dict[str, object]:
    """batch_summary.csv row for a sample with no result (blocked or never run): confidence blank,
    review status is the plain status text."""
    return {"sample": sample_id, "confidence": "", "inliers_weakest_step": "", "inlier_ratio_weakest_step": "",
            "sift_diff_um_worst_step": "", "warnings": "", "reference": reference or "",
            "review_status": status_text, "export_time": "", "tool_version": tool_version}


def summary_row(result: Dict, reference: Optional[str], review_status: str, export_time: str,
                tool_version: str) -> Dict[str, object]:
    """One batch_summary.csv row. The inlier columns are the smallest over the
    engine steps (the weakest step); sift_diff_um_worst_step is the largest
    second-method difference over the steps; blank when unavailable."""
    diags = [d for d in result.get("diagnostics", []) if d.get("n_inliers") is not None]
    ratios = [d["inlier_ratio"] for d in diags if d.get("inlier_ratio") is not None and d["inlier_ratio"] == d["inlier_ratio"]]
    diffs = [d["second_opinion_diff_mean_um"] for d in result.get("diagnostics", [])
             if d.get("second_opinion_diff_mean_um") is not None]
    warnings = " | ".join(" ".join(str(w).split()) for w in result.get("warnings") or [])
    return {
        "sample": result.get("sample_id"),
        "confidence": result.get("confidence") if result.get("status") == "ok" else result.get("status"),
        "inliers_weakest_step": min((d["n_inliers"] for d in diags), default=""),
        "inlier_ratio_weakest_step": round(min(ratios), 4) if ratios else "",
        "sift_diff_um_worst_step": round(max(diffs), 4) if diffs else "",
        "warnings": warnings,
        "reference": reference or result.get("reference_modality") or "",
        "review_status": review_status,
        "export_time": export_time,
        "tool_version": tool_version,
    }


def read_export_times(path: "str | Path") -> Dict[str, str]:
    """{sample: export_time} from an existing batch_summary.csv ({} if none)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["sample"]: r.get("export_time", "") for r in csv.DictReader(f)}


def write_batch_summary(rows: List[Dict], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
