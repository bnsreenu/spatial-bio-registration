"""Task 9, item 3: export per-core matrices and keypoints in the dataset's
own format (Xenium Explorer `{name}_imagealignment.csv`, 3x3 row-major, and
a keypoints CSV with columns fixedX, fixedY, alignmentX, alignmentY).

This reuses src/chain.py's automatic matrices (M_he_auto, M_comet_auto,
mapping H&E/COMET native pixels into the reference frame), and re-derives
the actual matched inlier point pairs behind each registration step
(auto_register.feature_based_similarity with return_points=True, which
src/engine.py's register_pair does not expose) so the exported keypoints
file reflects real matched features, not a synthetic grid.

Caveat, stated plainly in outputs/task9_notes.md: this export has not been
tested by importing it into Xenium Explorer.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import auto_register as ar
import data_access as da
import engine

N_EXPORT_POINTS = 40


def _register_points_native(
    img_fixed: np.ndarray,
    img_moving: np.ndarray,
    px_fixed_um: float,
    px_moving_um: float,
    method: str,
    mirrored: bool,
) -> Dict:
    """Reproduce engine.register_pair's resample + (optional) mirror step
    for a single, already-known-winning (method, mirrored) hypothesis, and
    return the actual matched inlier points, converted to each image's own
    native pixel grid (px_fixed_um / px_moving_um).

    This duplicates a small part of engine._common_resample /
    engine._match_with_mirror_test on purpose: register_pair's own
    contract (task 9 item 1) does not return point coordinates, only the
    fitted matrix and diagnostics, and item 3 needs the real matched
    points for the keypoints export. mirrored must be the value already
    returned by chain.py's register_pair call for this step, so the exact
    same winning hypothesis is re-extracted (not re-decided here).
    """
    import cv2

    fixed_ws, moving_ws, ws_fixed_um, ws_moving_um = engine._common_resample(
        img_fixed, img_moving, px_fixed_um, px_moving_um
    )

    moving_for_match = moving_ws
    if mirrored:
        moving_for_match = cv2.flip(np.asarray(moving_ws, dtype=np.float32), 1)

    res = ar.feature_based_similarity(moving_for_match, fixed_ws, method=method, return_points=True)
    if res["M"] is None or res["pts_src"] is None or res["inlier_mask"] is None:
        return {"status": res["status"], "pts_moving_native": None, "pts_fixed_native": None}

    inlier = res["inlier_mask"]
    pts_moving_for_match = res["pts_src"][inlier]
    pts_fixed_ws = res["pts_dst"][inlier]

    if mirrored:
        # pts_moving_for_match are in the flipped image's pixel space;
        # undo the flip (same construction as engine.mirror_matrix) to
        # recover coordinates in the original (unflipped) moving_ws image.
        width = moving_ws.shape[1]
        pts_moving_ws = pts_moving_for_match.copy()
        pts_moving_ws[:, 0] = (width - 1) - pts_moving_for_match[:, 0]
    else:
        pts_moving_ws = pts_moving_for_match

    scale_moving = ws_moving_um / px_moving_um
    scale_fixed = ws_fixed_um / px_fixed_um
    pts_moving_native = pts_moving_ws * scale_moving
    pts_fixed_native = pts_fixed_ws * scale_fixed

    return {"status": "ok", "pts_moving_native": pts_moving_native, "pts_fixed_native": pts_fixed_native}


def select_spatially_distributed(points: np.ndarray, n_target: int = N_EXPORT_POINTS) -> np.ndarray:
    """Grid-based selection of up to about n_target spatially well
    distributed points from a point cloud.

    Bins points into an n_cells x n_cells grid (n_cells = ceil(sqrt(n_target)))
    over the point cloud's own bounding box, and keeps, from each occupied
    cell, the single point closest to that cell's center. Returns the
    integer indices (into the input array) of the selected points.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)
    if n_points <= n_target:
        return np.arange(n_points)

    n_cells = max(1, math.ceil(math.sqrt(n_target)))
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)

    col = np.clip(((points[:, 0] - xmin) / width * n_cells).astype(int), 0, n_cells - 1)
    row = np.clip(((points[:, 1] - ymin) / height * n_cells).astype(int), 0, n_cells - 1)
    cell_id = row * n_cells + col

    selected = []
    for cid in np.unique(cell_id):
        idx_in_cell = np.where(cell_id == cid)[0]
        cx = xmin + (cid % n_cells + 0.5) * (width / n_cells)
        cy = ymin + (cid // n_cells + 0.5) * (height / n_cells)
        cell_pts = points[idx_in_cell]
        dist = np.hypot(cell_pts[:, 0] - cx, cell_pts[:, 1] - cy)
        selected.append(idx_in_cell[np.argmin(dist)])
    return np.array(selected, dtype=int)


def write_imagealignment_csv(m_native: np.ndarray, path: Path) -> None:
    """Write a 3x3 matrix, row-major, no header, no index (Xenium Explorer
    `{name}_imagealignment.csv` format)."""
    pd.DataFrame(m_native).to_csv(path, header=False, index=False)


def write_keypoints_csv(pts_alignment_native: np.ndarray, m_native: np.ndarray, path: Path) -> None:
    """Write a keypoints CSV with columns fixedX, fixedY, alignmentX,
    alignmentY, matching keypoints_he.csv/keypoints_comet.csv's format.

    fixedX/fixedY are m_native applied to each alignment point (reference
    frame coordinates), i.e. the same direction as the dataset's own
    provided matrices (fixed = M @ alignment).
    """
    pts_h = np.hstack([pts_alignment_native, np.ones((pts_alignment_native.shape[0], 1))])
    fixed = (pts_h @ m_native[:2, :].T)
    df = pd.DataFrame(
        {
            "fixedX": fixed[:, 0],
            "fixedY": fixed[:, 1],
            "alignmentX": pts_alignment_native[:, 0],
            "alignmentY": pts_alignment_native[:, 1],
        }
    )
    df.to_csv(path, index=False)


def export_core(data_dir: Path, core_id: str, chain_result: Dict, out_dir: Path) -> Dict:
    """Export matrix_he/matrix_comet equivalents and keypoint CSVs for one
    core, reusing chain_result (one entry of chain.run_all's output).

    Returns a dict of per-modality status strings, for reporting in
    outputs/task9_notes.md.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}

    images = chain_result["images"]
    m_comet_auto = chain_result["M_comet_auto"]
    m_he_auto = chain_result["M_he_auto"]

    # COMET: re-extract the Xenium-render-to-COMET inlier matches (the
    # same registration step used to build M_comet_auto), then express
    # them as COMET-native "alignment" points.
    if m_comet_auto is not None:
        xen_reg = chain_result["xenium_reg"]
        pts = _register_points_native(
            img_fixed=images["dapi_norm"],
            img_moving=chain_result["render_img"],
            px_fixed_um=images["comet_um"],
            px_moving_um=1.0,
            method=xen_reg["method"],
            mirrored=xen_reg["mirrored"],
        )
        if pts["status"] == "ok":
            # chain.py declared px_fixed_um=images["comet_um"] (working
            # resolution) to register_pair for this step, so
            # pts_fixed_native is at COMET's working grid, not its true
            # native grid; one more scale step (same pattern as
            # evaluation.to_working_matrix) is needed before export.
            comet_native_scale = images["comet_um"] / images["px_comet"]
            pts_comet_native = pts["pts_fixed_native"] * comet_native_scale
            idx = select_spatially_distributed(pts_comet_native)
            write_imagealignment_csv(m_comet_auto, out_dir / "comet_imagealignment.csv")
            write_keypoints_csv(pts_comet_native[idx], m_comet_auto, out_dir / "comet_keypoints.csv")
            status["comet"] = f"ok, {len(idx)} points exported"
        else:
            status["comet"] = f"failed to recover matched points: {pts['status']}"
    else:
        status["comet"] = "M_comet_auto not available (xenium-to-comet registration failed)"

    # H&E: re-extract the H&E-to-COMET inlier matches used to build T (and
    # hence M_he_auto), and express them as H&E-native "alignment" points.
    if m_he_auto is not None:
        he_reg = chain_result["he_reg"]
        pts = _register_points_native(
            img_fixed=images["dapi_norm"],
            img_moving=images["hema"],
            px_fixed_um=images["comet_um"],
            px_moving_um=images["he_um"],
            method=he_reg["method"],
            mirrored=he_reg["mirrored"],
        )
        if pts["status"] == "ok":
            # Same working-to-native conversion needed on the H&E side.
            he_native_scale = images["he_um"] / images["px_he"]
            pts_he_native = pts["pts_moving_native"] * he_native_scale
            idx = select_spatially_distributed(pts_he_native)
            write_imagealignment_csv(m_he_auto, out_dir / "he_imagealignment.csv")
            write_keypoints_csv(pts_he_native[idx], m_he_auto, out_dir / "he_keypoints.csv")
            status["he"] = f"ok, {len(idx)} points exported"
        else:
            status["he"] = f"failed to recover matched points: {pts['status']}"
    else:
        status["he"] = "M_he_auto not available (chain failed upstream)"

    return status


def export_all(data_dir: Path, chain_results, output_dir: Path) -> Dict[str, Dict]:
    """Run export_core for every core in chain_results (chain.run_all's
    output list). Returns {core_id: status_dict}."""
    all_status = {}
    for res in chain_results:
        core_id = res["core_id"]
        out_dir = output_dir / "export" / core_id
        all_status[core_id] = export_core(data_dir, core_id, res, out_dir)
    return all_status
