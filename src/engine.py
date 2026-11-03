"""Task 9, item 1: a reusable image-to-image registration engine.

register_pair(img_fixed, img_moving, px_fixed_um, px_moving_um, method,
mirror_test) wraps auto_register.feature_based_similarity (task 5
settings, unchanged defaults) with:

  - resampling only when needed (task 12): if the two declared pixel sizes
    agree within 0.1 percent the images are used as given; otherwise both
    are resampled once to the coarser (larger) of the two (same
    convention as auto_baselines.load_core_images), then the fitted
    similarity is converted back out to the pixel grids the caller
    actually declared for each image (evaluation.to_working_matrix, run in
    its documented "working to native" direction);
  - a mirror test: also match the horizontally flipped moving image, and
    keep whichever hypothesis (original or flipped) has more inliers. If
    the flipped hypothesis wins, the returned matrix is composed with the
    flip (so it carries a negative determinant) and mirrored=True;
  - a confidence label from n_inliers / inlier_ratio;
  - a second-opinion run with the other feature method (AKAZE if SIFT is
    primary, and vice versa), including transform_difference_um between
    the two candidate matrices as a simple agreement check;
  - a plain-text list of warnings.

px_fixed_um and px_moving_um must be the actual pixel size of the arrays
passed in (this can be a dataset's true native resolution, or any working
resolution the caller chooses: the function does not know or assume
which). The returned matrix maps moving-image pixel coordinates (at
px_moving_um) to fixed-image pixel coordinates (at px_fixed_um), in
whatever grids those two values represent for the caller; see
src/chain.py for an example that treats register_pair's own output as an
intermediate "working" result and performs a further explicit conversion
to a dataset's true native pixel grid.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

import auto_register as ar
import data_access as da
import evaluation as ev

METHODS = ("sift", "akaze")

# Confidence thresholds, fixed and documented, not tuned per core or dataset
# (task 9 wording): high needs both conditions; low is checked first since
# it takes priority over the "30 to 100 inliers" medium range.
HIGH_MIN_INLIERS = 100
HIGH_MIN_RATIO = 0.7
MEDIUM_MIN_INLIERS = 30
LOW_MAX_RATIO = 0.4

# Mean disagreement between the primary and second-opinion matrices above
# which a warning is raised; same order of magnitude as
# robustness.SUCCESS_THRESHOLD_UM (5 um), which already separates
# "recovered correctly" from "wildly wrong" in this project's other checks.
AGREEMENT_WARN_UM = 5.0

# Task 10, item 0a: a lighter-weight notice raised below AGREEMENT_WARN_UM,
# for disagreement that is still larger than the sub-micron accuracy chain.py
# (task 9) found the engine capable of on real cores, so it is worth
# surfacing even when it does not reach the "wildly wrong" threshold above.
AGREEMENT_NOTICE_UM = 1.0

# Task 12, item 2: if the two declared (effective) pixel sizes agree within
# this relative tolerance, register_pair does not resample at all; the
# small scale difference is absorbed exactly by the fitted similarity and
# by the pixel-size-aware conversion of the returned matrix. Otherwise it
# resamples once, in _prepare_working_pair.
PIXEL_SIZE_AGREE_RTOL = 1e-3


def other_method(method: str) -> str:
    """The other of the two supported feature methods."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")
    return "sift" if method == "akaze" else "akaze"


def mirror_matrix(width: int) -> np.ndarray:
    """Horizontal flip about the image center, as a 3x3 matrix (det = -1).

    Maps a point at pixel column x to column (width - 1 - x); y unchanged.
    Same construction as robustness._flip_matrix_about_center.
    """
    cx = (width - 1) / 2.0
    t1 = np.eye(3)
    t1[0, 2] = -cx
    flip = np.diag([-1.0, 1.0, 1.0])
    t2 = np.eye(3)
    t2[0, 2] = cx
    return t2 @ flip @ t1


def _match_with_mirror_test(fixed_ws: np.ndarray, moving_ws: np.ndarray, method: str, mirror_test: bool) -> Dict:
    """Run feature_based_similarity, and (if mirror_test) also on the
    horizontally flipped moving image; keep whichever has more inliers.

    Returns the same keys as feature_based_similarity plus "mirrored"
    (bool). If the flipped hypothesis wins, M (moving_ws -> fixed_ws) is
    already composed with the flip matrix, so it carries a reflection.
    """
    res = ar.feature_based_similarity(moving_ws, fixed_ws, method=method, return_points=True)
    best = dict(res)
    best["mirrored"] = False

    if mirror_test:
        moving_mirror = cv2.flip(np.asarray(moving_ws, dtype=np.float32), 1)
        res_mirror = ar.feature_based_similarity(moving_mirror, fixed_ws, method=method, return_points=True)
        if res_mirror["M"] is not None and res_mirror["n_inliers"] > best["n_inliers"]:
            flip_mat = mirror_matrix(moving_ws.shape[1])
            best = dict(res_mirror)
            best["M"] = res_mirror["M"] @ flip_mat
            best["mirrored"] = True

    return best


def _common_resample(img_fixed: np.ndarray, img_moving: np.ndarray, px_fixed_um: float, px_moving_um: float):
    """Resample both images to the coarser (larger um/px) of the two
    declared pixel sizes, matching auto_baselines.load_core_images.
    Returns (fixed_ws, moving_ws, ws_fixed_um, ws_moving_um)."""
    common_um = max(px_fixed_um, px_moving_um)
    fixed_ws, ws_fixed_um = da.resample_to_um(img_fixed, px_fixed_um, common_um)
    moving_ws, ws_moving_um = da.resample_to_um(img_moving, px_moving_um, common_um)
    return fixed_ws, moving_ws, ws_fixed_um, ws_moving_um


def _prepare_working_pair(img_fixed: np.ndarray, img_moving: np.ndarray, px_fixed_um: float, px_moving_um: float):
    """Task 12, item 2: the single place where register_pair may resample.

    If the two declared pixel sizes agree within PIXEL_SIZE_AGREE_RTOL
    (0.1 percent), the images are used as given (no resampling) and the
    working pixel sizes are exactly the declared ones. Otherwise both are
    resampled once to the coarser pixel size (_common_resample).
    Returns (fixed_ws, moving_ws, ws_fixed_um, ws_moving_um, resampled).
    """
    if abs(px_fixed_um / px_moving_um - 1.0) <= PIXEL_SIZE_AGREE_RTOL:
        return img_fixed, img_moving, float(px_fixed_um), float(px_moving_um), False
    fixed_ws, moving_ws, ws_fixed_um, ws_moving_um = _common_resample(
        img_fixed, img_moving, px_fixed_um, px_moving_um
    )
    return fixed_ws, moving_ws, ws_fixed_um, ws_moving_um, True


def _inlier_points_native(
    match: Dict, moving_ws_width: int, ws_moving_um: float, ws_fixed_um: float, px_moving_um: float, px_fixed_um: float
) -> Dict:
    """Matched RANSAC inlier points of one (method, mirrored) hypothesis,
    in the pixel grids of the arrays passed to register_pair (px_moving_um
    / px_fixed_um), i.e. converted back from the working grid with the same
    scale factors used for the matrix (evaluation.to_working_matrix). If
    the winning hypothesis was the mirrored one, the x coordinates of the
    moving points are un-flipped first (same construction as
    mirror_matrix), so src_pts always refer to the original moving image.

    Returns src_pts, dst_pts (float64, shape (n, 2), x then y) and
    match_ids (int64, position of each inlier among all ratio-test matches
    of that hypothesis); all empty if no inlier mask exists.
    """
    empty = {
        "src_pts": np.zeros((0, 2), dtype=np.float64),
        "dst_pts": np.zeros((0, 2), dtype=np.float64),
        "match_ids": np.zeros((0,), dtype=np.int64),
    }
    mask = match.get("inlier_mask")
    pts_src = match.get("pts_src")
    pts_dst = match.get("pts_dst")
    if mask is None or pts_src is None or pts_dst is None:
        return empty

    ids = np.nonzero(mask)[0].astype(np.int64)
    src = np.asarray(pts_src, dtype=np.float64)[ids].copy()
    dst = np.asarray(pts_dst, dtype=np.float64)[ids].copy()
    if match.get("mirrored"):
        src[:, 0] = (moving_ws_width - 1) - src[:, 0]
    src *= ws_moving_um / px_moving_um
    dst *= ws_fixed_um / px_fixed_um
    return {"src_pts": src, "dst_pts": dst, "match_ids": ids}


def _native_matrix(m_ws: np.ndarray, ws_moving_um: float, ws_fixed_um: float, px_moving_um: float, px_fixed_um: float) -> np.ndarray:
    """Convert a moving_ws -> fixed_ws matrix to moving(px_moving_um) -> fixed(px_fixed_um).

    evaluation.to_working_matrix(m_native, px_src, px_dst, ws_src, ws_dst)
    builds a working-resolution matrix from a native one; calling it with
    the native/working argument pairs swapped (documented in
    evaluation.py) recovers the native matrix from a working one, which is
    exactly the conversion needed here: m_ws plays the role of "working",
    (ws_moving_um, ws_fixed_um) are its own pixel sizes, and
    (px_moving_um, px_fixed_um) are the pixel sizes the caller declared
    for the original (pre-resample) images.
    """
    return ev.to_working_matrix(m_ws, ws_moving_um, ws_fixed_um, px_moving_um, px_fixed_um)


def _confidence(n_inliers, inlier_ratio) -> str:
    """high: n_inliers >= 100 and inlier_ratio >= 0.7.
    low: n_inliers < 30 or inlier_ratio < 0.4 (checked first, takes
    priority over the medium range).
    medium: everything else (30 to 100 inliers, or >= 100 inliers with
    inlier_ratio between 0.4 and 0.7)."""
    if n_inliers is None or inlier_ratio is None or not np.isfinite(inlier_ratio):
        return "low"
    if n_inliers < MEDIUM_MIN_INLIERS or inlier_ratio < LOW_MAX_RATIO:
        return "low"
    if n_inliers >= HIGH_MIN_INLIERS and inlier_ratio >= HIGH_MIN_RATIO:
        return "high"
    return "medium"


def register_pair(
    img_fixed: np.ndarray,
    img_moving: np.ndarray,
    px_fixed_um: float,
    px_moving_um: float,
    method: str = "akaze",
    mirror_test: bool = True,
) -> Dict:
    """Register img_moving onto img_fixed. See the module docstring for
    the full contract. Returns a dict:

      M                3x3, moving (px_moving_um) -> fixed (px_fixed_um),
                       or None if the primary method failed
      status           "ok" or a failure reason from feature_based_similarity
      n_matches, n_inliers, inlier_ratio   from the winning (primary) hypothesis
      mirrored         bool, True if the flipped-moving hypothesis won
      confidence       "high" / "medium" / "low"
      method           the primary method actually used
      second_opinion   dict: method, status, M, n_inliers, inlier_ratio,
                       mirrored, diff_um (mean/median/max/n_points_used vs
                       the primary M, or None if either failed)
      warnings         list of plain-text strings
      src_pts, dst_pts float64 (n, 2) matched RANSAC inliers of the primary
                       hypothesis, x then y, in the pixel grids of the
                       arrays passed in (px_moving_um / px_fixed_um), i.e.
                       converted back from any internal working grid;
                       M @ [src_pts, 1] is approximately dst_pts. If the
                       mirrored hypothesis won, src_pts refer to the
                       original (unflipped) moving image. Empty (0, 2) if
                       no inliers exist.
      match_ids        int64 (n,), position of each inlier among all
                       ratio-test matches of the primary hypothesis
      resampled        True if the two pixel sizes differed by more than
                       PIXEL_SIZE_AGREE_RTOL and register_pair resampled
                       (once); False if the images were used as given

    The two pixel sizes should be the EFFECTIVE pixel sizes of the arrays
    (for example the value open_layer returns). If they agree within 0.1
    percent no resampling happens at all; otherwise both images are
    resampled once to the coarser size, in _prepare_working_pair only.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")

    fixed_ws, moving_ws, ws_fixed_um, ws_moving_um, resampled = _prepare_working_pair(
        img_fixed, img_moving, px_fixed_um, px_moving_um
    )

    primary = _match_with_mirror_test(fixed_ws, moving_ws, method, mirror_test)
    secondary = _match_with_mirror_test(fixed_ws, moving_ws, other_method(method), mirror_test)

    warnings: List[str] = []
    if resampled:
        warnings.append(
            f"pixel sizes differ by more than {PIXEL_SIZE_AGREE_RTOL * 100:g} percent "
            f"({px_fixed_um:.6g} vs {px_moving_um:.6g} um/px); both images were resampled once to "
            f"{max(px_fixed_um, px_moving_um):.6g} um/px inside register_pair."
        )

    m_native = None
    if primary["M"] is not None:
        m_native = _native_matrix(primary["M"], ws_moving_um, ws_fixed_um, px_moving_um, px_fixed_um)
    else:
        warnings.append(f"primary method {method!r} failed: status={primary['status']!r}")

    m_native_secondary = None
    if secondary["M"] is not None:
        m_native_secondary = _native_matrix(secondary["M"], ws_moving_um, ws_fixed_um, px_moving_um, px_fixed_um)
    else:
        warnings.append(f"second-opinion method {other_method(method)!r} failed: status={secondary['status']!r}")

    diff = None
    if m_native is not None and m_native_secondary is not None:
        grid = ev.sample_grid_in_mask(
            np.ones(moving_ws.shape, dtype=bool), ws_moving_um, px_moving_um, step_px_mask=20
        )
        mean_um, median_um, max_um, n_used = ev.transform_difference_um(
            m_native_secondary, m_native, grid, comet_px_um=px_fixed_um
        )
        diff = {"mean_um": mean_um, "median_um": median_um, "max_um": max_um, "n_points_used": n_used}
        if n_used > 0 and mean_um > AGREEMENT_WARN_UM:
            warnings.append(
                f"second-opinion ({other_method(method)}) disagrees with primary ({method}) "
                f"by {mean_um:.2f} um on average (> {AGREEMENT_WARN_UM:g} um)."
            )
        elif n_used > 0 and mean_um > AGREEMENT_NOTICE_UM:
            warnings.append(
                f"second-opinion ({other_method(method)}) differs from primary ({method}) "
                f"by {mean_um:.2f} um on average (> {AGREEMENT_NOTICE_UM:g} um)."
            )

    if primary.get("mirrored"):
        warnings.append(
            "mirror hypothesis had more inliers than the non-mirrored one; "
            "the returned matrix includes a reflection (negative determinant)."
        )

    confidence = _confidence(primary.get("n_inliers"), primary.get("inlier_ratio"))
    if confidence == "low" and primary["M"] is not None:
        warnings.append(
            f"low confidence: n_inliers={primary['n_inliers']}, "
            f"inlier_ratio={primary['inlier_ratio']:.2f}"
        )

    points = _inlier_points_native(
        primary, moving_ws.shape[1], ws_moving_um, ws_fixed_um, px_moving_um, px_fixed_um
    )

    return {
        "M": m_native,
        "status": primary["status"],
        "n_matches": primary["n_matches"],
        "n_inliers": primary["n_inliers"],
        "inlier_ratio": primary["inlier_ratio"],
        "mirrored": bool(primary.get("mirrored", False)),
        "confidence": confidence,
        "method": method,
        "second_opinion": {
            "method": other_method(method),
            "status": secondary["status"],
            "M": m_native_secondary,
            "n_inliers": secondary["n_inliers"],
            "inlier_ratio": secondary["inlier_ratio"],
            "mirrored": bool(secondary.get("mirrored", False)),
            "diff_um": diff,
        },
        "warnings": warnings,
        "src_pts": points["src_pts"],
        "dst_pts": points["dst_pts"],
        "match_ids": points["match_ids"],
        "resampled": resampled,
    }
