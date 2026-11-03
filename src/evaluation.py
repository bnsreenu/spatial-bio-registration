"""Evaluation harness: a native-pixel target transform, and metrics against it.

This module does NOT implement a registration method. It builds a target
H&E-to-COMET transform from the matrices already provided with the dataset
(matrix_he.csv, matrix_comet.csv), so that any future candidate transform can
be scored against this target in physical units (micrometers).

Both matrix_he.csv and matrix_comet.csv map their own modality's native
pixels into the same shared fixed/canvas frame:

    fixed = M_he_raw @ he_native
    fixed = M_comet_raw @ comet_native

so a direct H&E-native to COMET-native transform is:

    comet_native = inv(M_comet_raw) @ M_he_raw @ he_native

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from fit_transforms import (
    apply_transform,
    fit_similarity,
    load_keypoints,
    load_provided_matrix,
)


# ---------------------------------------------------------------------------
# Target transform
# ---------------------------------------------------------------------------

def _matrix_path(data_dir: Path, core_id: str, modality: str) -> Path:
    return data_dir / core_id / modality / f"matrix_{modality}.csv"


def target_he_to_comet(data_dir: Path, core_id: str) -> np.ndarray:
    """Build the native-pixel H&E-to-COMET target transform for one core.

    Loads matrix_he.csv and matrix_comet.csv (both map native source pixels
    into the shared fixed frame) and returns:

        M = inv(M_comet_raw) @ M_he_raw

    which maps H&E native pixel coordinates directly to COMET native pixel
    coordinates. This is the transform that a working registration method
    should reproduce; it is derived only from the matrices shipped with the
    dataset, not fit from scratch.
    """
    m_he_raw = load_provided_matrix(_matrix_path(data_dir, core_id, "he"))
    m_comet_raw = load_provided_matrix(_matrix_path(data_dir, core_id, "comet"))
    return np.linalg.inv(m_comet_raw) @ m_he_raw


# ---------------------------------------------------------------------------
# Noise floor: how much does the target itself move under leave-one-out
# ---------------------------------------------------------------------------

def _keypoints_path(data_dir: Path, core_id: str, modality: str) -> Path:
    return data_dir / core_id / modality / f"keypoints_{modality}.csv"


def noise_floor_he_to_comet_diffs(
    data_dir: Path,
    core_id: str,
    grid_points_he_native: np.ndarray,
    comet_px_um: float = 0.28,
    comet_valid_mask: np.ndarray = None,
    comet_valid_mask_um_per_px: float = None,
) -> Dict:
    """How much the H&E-to-COMET target transform itself moves under
    leave-one-out refitting of either side's keypoints, as a noise floor.

    For each keypoint left out of the H&E fit (with the COMET fit held
    fixed at its full-data value), and separately for each keypoint left
    out of the COMET fit (with the H&E fit held fixed), rebuilds the
    composed H&E-to-COMET transform and measures its difference from the
    full-data target with evaluation.transform_difference_um. These
    per-left-out-point differences describe how much noise is already
    present in the provided matrices themselves: an automatic method
    scoring within this range is "equal within noise", not demonstrably
    worse (or better) than the reference.

    Requires at least 4 keypoints on a given side to compute its
    leave-one-out set (same minimum as fit_transforms.leave_one_out_similarity);
    a side with too few keypoints is skipped and noted in "notes".

    Returns a dict: status, mean_um, median_um, max_um (over all pooled
    per-left-out-point differences from both sides), n_values, notes.
    """
    notes: List[str] = []
    kp_he_path = _keypoints_path(data_dir, core_id, "he")
    kp_comet_path = _keypoints_path(data_dir, core_id, "comet")
    mx_he_path = _matrix_path(data_dir, core_id, "he")
    mx_comet_path = _matrix_path(data_dir, core_id, "comet")

    for p in (kp_he_path, kp_comet_path, mx_he_path, mx_comet_path):
        if not p.exists():
            return {
                "status": "missing_file",
                "mean_um": None,
                "median_um": None,
                "max_um": None,
                "n_values": 0,
                "notes": [f"{core_id}: required file not found: {p}"],
            }

    src_he, dst_he = load_keypoints(kp_he_path)
    src_comet, dst_comet = load_keypoints(kp_comet_path)
    m_he_full = load_provided_matrix(mx_he_path)
    m_comet_full = load_provided_matrix(mx_comet_path)
    m_target = np.linalg.inv(m_comet_full) @ m_he_full

    diffs_um: List[float] = []

    n_he = src_he.shape[0]
    if n_he >= 4:
        mask = np.ones(n_he, dtype=bool)
        for i in range(n_he):
            mask[:] = True
            mask[i] = False
            m_he_loo = fit_similarity(src_he[mask], dst_he[mask])
            m_prime = np.linalg.inv(m_comet_full) @ m_he_loo
            mean_um, _, _, n_used = transform_difference_um(
                m_prime,
                m_target,
                grid_points_he_native,
                comet_px_um=comet_px_um,
                comet_valid_mask=comet_valid_mask,
                comet_valid_mask_um_per_px=comet_valid_mask_um_per_px,
            )
            if n_used > 0:
                diffs_um.append(mean_um)
    else:
        notes.append(
            f"{core_id}: only {n_he} H&E keypoints, need >= 4 for leave-one-out noise floor on this side."
        )

    n_comet = src_comet.shape[0]
    if n_comet >= 4:
        mask = np.ones(n_comet, dtype=bool)
        for j in range(n_comet):
            mask[:] = True
            mask[j] = False
            m_comet_loo = fit_similarity(src_comet[mask], dst_comet[mask])
            m_prime = np.linalg.inv(m_comet_loo) @ m_he_full
            mean_um, _, _, n_used = transform_difference_um(
                m_prime,
                m_target,
                grid_points_he_native,
                comet_px_um=comet_px_um,
                comet_valid_mask=comet_valid_mask,
                comet_valid_mask_um_per_px=comet_valid_mask_um_per_px,
            )
            if n_used > 0:
                diffs_um.append(mean_um)
    else:
        notes.append(
            f"{core_id}: only {n_comet} COMET keypoints, need >= 4 for leave-one-out noise floor on this side."
        )

    if not diffs_um:
        return {
            "status": "no_values",
            "mean_um": None,
            "median_um": None,
            "max_um": None,
            "n_values": 0,
            "notes": notes,
        }

    diffs_arr = np.array(diffs_um)
    return {
        "status": "ok",
        "mean_um": float(np.mean(diffs_arr)),
        "median_um": float(np.median(diffs_arr)),
        "max_um": float(np.max(diffs_arr)),
        "n_values": int(diffs_arr.shape[0]),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Resolution conversion
# ---------------------------------------------------------------------------

def to_working_matrix(
    m_native: np.ndarray,
    px_src_um: float,
    px_dst_um: float,
    ws_src_um: float,
    ws_dst_um: float,
) -> np.ndarray:
    """Convert a native-pixel transform to a working-resolution transform.

    m_native maps native-pixel source coordinates to native-pixel
    destination coordinates. px_src_um/px_dst_um are each modality's native
    pixel size in micrometers; ws_src_um/ws_dst_um are the working-resolution
    pixel sizes actually used for the source/destination images (these can
    differ between modalities, e.g. H&E and COMET landing on different
    pyramid levels for the same nominal 2 um/px request).

    Derivation: a working-resolution source pixel coordinate p_src_ws
    corresponds to native coordinate p_src_native = p_src_ws * (ws_src_um /
    px_src_um). Applying m_native and converting the result back to working
    resolution in the destination frame gives:

        p_dst_ws = (px_dst_um / ws_dst_um) * m_native[:2, :2] @
                   ((ws_src_um / px_src_um) * p_src_ws)
                   + (px_dst_um / ws_dst_um) * m_native[:2, 2]

    i.e. M_working = S_dst_inv @ M_native @ S_src, where S_src and S_dst are
    the diagonal (plus homogeneous 1) scale matrices converting working-
    resolution pixels to native pixels in the source and destination frames
    respectively.

    Calling this function again with the four scale arguments in swapped
    order, i.e. to_working_matrix(m_working, ws_src_um, ws_dst_um,
    px_src_um, px_dst_um), recovers m_native, since S_src and S_dst are
    then simply inverted.
    """
    scale_src_to_native = ws_src_um / px_src_um
    scale_dst_to_native = ws_dst_um / px_dst_um

    s_src = np.diag([scale_src_to_native, scale_src_to_native, 1.0])
    s_dst_inv = np.diag([1.0 / scale_dst_to_native, 1.0 / scale_dst_to_native, 1.0])

    return s_dst_inv @ m_native @ s_src


def _roundtrip_self_test(tol: float = 1e-9) -> None:
    """Unit test: converting to working resolution and back recovers m_native.

    Uses arbitrary but fixed native/working pixel sizes and an arbitrary
    affine matrix; asserts max abs error below tol. Run via
    `python evaluation.py` or imported and called explicitly; not run on
    import so importing this module has no side effects.
    """
    rng = np.random.default_rng(0)
    m_native = np.eye(3)
    m_native[:2, :2] = rng.normal(size=(2, 2)) * 0.1 + np.eye(2)
    m_native[:2, 2] = rng.normal(size=2) * 50.0

    px_src_um, px_dst_um = 0.2031, 0.28
    ws_src_um, ws_dst_um = 1.625, 2.24

    m_working = to_working_matrix(m_native, px_src_um, px_dst_um, ws_src_um, ws_dst_um)
    m_recovered = to_working_matrix(m_working, ws_src_um, ws_dst_um, px_src_um, px_dst_um)

    max_err = float(np.max(np.abs(m_recovered - m_native)))
    if max_err >= tol:
        raise AssertionError(
            f"to_working_matrix round trip error {max_err:.3e} exceeds tolerance {tol:.1e}"
        )
    print(f"to_working_matrix round trip max abs error: {max_err:.3e} (tolerance {tol:.1e}): OK")


# ---------------------------------------------------------------------------
# Grid sampling and difference metric
# ---------------------------------------------------------------------------

def sample_grid_in_mask(
    mask: np.ndarray, mask_um_per_px: float, native_um_per_px: float, step_px_mask: int = 10
) -> np.ndarray:
    """Sample a regular grid of points inside a working-resolution tissue mask.

    mask is a boolean array at mask_um_per_px resolution (e.g. the H&E
    tissue mask from prep.tissue_mask at the working resolution used by
    data_access). Points are placed every step_px_mask pixels in the mask's
    own grid, kept only where the mask is True, and converted to native
    H&E pixel coordinates (the coordinate frame matrix_he.csv was built in)
    by multiplying by (mask_um_per_px / native_um_per_px).

    Returns an (N, 2) float array of (x, y) native-pixel coordinates.
    """
    ys, xs = np.mgrid[0 : mask.shape[0] : step_px_mask, 0 : mask.shape[1] : step_px_mask]
    ys = ys.ravel()
    xs = xs.ravel()
    inside = mask[ys, xs]
    xs = xs[inside]
    ys = ys[inside]
    scale = mask_um_per_px / native_um_per_px
    points_native = np.stack([xs, ys], axis=1).astype(np.float64) * scale
    return points_native


def transform_difference_um(
    m_candidate: np.ndarray,
    m_target: np.ndarray,
    grid_points_he_native: np.ndarray,
    comet_px_um: float = 0.28,
    comet_valid_mask: np.ndarray = None,
    comet_valid_mask_um_per_px: float = None,
) -> Tuple[float, float, float, int]:
    """Compare two H&E-native-to-COMET-native transforms on a point grid.

    Applies both m_candidate and m_target to grid_points_he_native (native
    H&E pixel coordinates), measures the Euclidean distance between the two
    predicted COMET-native pixel positions, and converts to micrometers
    using comet_px_um (the COMET native pixel size, 0.28 um/px per
    outputs/findings.md, confirmed by data_access.get_pixel_size_um).

    If comet_valid_mask is given (a boolean array marking COMET pixels that
    actually have data, e.g. non-zero DAPI, at comet_valid_mask_um_per_px
    resolution), grid points are additionally dropped if their
    m_target-mapped COMET native coordinate falls outside the mask array
    bounds or lands on a False pixel. comet_valid_mask_um_per_px must be
    given together with comet_valid_mask. If comet_valid_mask is None
    (default) this check is skipped, e.g. for self-tests that compare a
    matrix against itself without a real COMET image.

    Returns (mean_um, median_um, max_um, n_points_used). If no points
    survive filtering, returns (nan, nan, nan, 0) rather than raising, so
    callers can report the degenerate case plainly.
    """
    pred_target = apply_transform(m_target, grid_points_he_native)
    src_points = grid_points_he_native

    if comet_valid_mask is not None:
        if comet_valid_mask_um_per_px is None:
            raise ValueError(
                "comet_valid_mask_um_per_px is required when comet_valid_mask is given."
            )
        scale = comet_px_um / comet_valid_mask_um_per_px
        mask_xy = np.round(pred_target * scale).astype(int)
        h, w = comet_valid_mask.shape
        in_bounds = (
            (mask_xy[:, 0] >= 0)
            & (mask_xy[:, 0] < w)
            & (mask_xy[:, 1] >= 0)
            & (mask_xy[:, 1] < h)
        )
        has_data = np.zeros(len(src_points), dtype=bool)
        has_data[in_bounds] = comet_valid_mask[
            mask_xy[in_bounds, 1], mask_xy[in_bounds, 0]
        ]
        src_points = src_points[has_data]
        pred_target = pred_target[has_data]

    n_used = int(src_points.shape[0])
    if n_used == 0:
        return float("nan"), float("nan"), float("nan"), 0

    pred_candidate = apply_transform(m_candidate, src_points)
    dist_px = np.linalg.norm(pred_candidate - pred_target, axis=1)
    dist_um = dist_px * comet_px_um
    return float(np.mean(dist_um)), float(np.median(dist_um)), float(np.max(dist_um)), n_used


if __name__ == "__main__":
    _roundtrip_self_test()
