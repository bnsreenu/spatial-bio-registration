"""Task 10, item 1: a reusable landmark-fitting engine (similarity or affine,
optionally robust), and a landmark-suggestion helper that reuses the
grid-based spatial thinning already written for task 9's export.

fit_from_landmarks wraps fit_transforms.fit_similarity / fit_affine (and
weighted variants of each, written here) with:

  - optional per-landmark weights (weighted least squares; ignored for the
    leave-one-out pass, which always uses an unweighted fit of the
    remaining points, matching fit_transforms.leave_one_out_similarity /
    leave_one_out_affine's existing convention);
  - robust=True: RANSAC (cv2.estimateAffinePartial2D for similarity,
    cv2.estimateAffine2D for affine), not iterative single-point MAD
    trimming (that scheme already exists in src/robust_refit.py and task 10
    explicitly asks for a different one here). If RANSAC's inlier fraction
    falls below min_inlier_fraction (default 0.7), the fit is considered
    unreliable and falls back to the plain (weighted, non-robust) fit over
    all points, with status "low_inlier_fraction_fallback";
  - per-landmark residuals in both pixels and micrometers (when
    dst_um_per_px is given);
  - a leave-one-out residual per landmark, and a boolean "suspect" flag:
    True when a landmark's own leave-one-out residual exceeds both 3 times
    the median leave-one-out residual AND 2 pixels (both conditions must
    hold, per task wording).

suggest_landmarks thins a set of automatic inlier matches down to about n
spatially well distributed points, reusing export.select_spatially_distributed
exactly (same grid-cell-nearest-to-center rule), so "automatically suggested
landmarks" and "task 9's exported keypoints" are built the same way.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from export import select_spatially_distributed
from fit_transforms import apply_transform, fit_affine, fit_similarity, residuals

MIN_INLIER_FRACTION = 0.7
DEFAULT_RANSAC_THRESH_PX = 5.0  # fixed, documented default; not tuned per experiment.
SUSPECT_MEDIAN_FACTOR = 3.0
SUSPECT_MIN_PX = 2.0

MIN_POINTS = {"similarity": 3, "affine": 3}
MIN_POINTS_LOO = {"similarity": 4, "affine": 7}


# ---------------------------------------------------------------------------
# Weighted (non-robust) fits
# ---------------------------------------------------------------------------

def weighted_fit_similarity(src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted Umeyama similarity fit. weights=None is equivalent to
    fit_transforms.fit_similarity (uniform weights)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    if weights is None:
        return fit_similarity(src, dst)

    w = np.asarray(weights, dtype=np.float64)
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")

    src_mean = (w[:, None] * src).sum(axis=0) / total_w
    dst_mean = (w[:, None] * dst).sum(axis=0) / total_w
    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = (w * (src_c ** 2).sum(axis=1)).sum() / total_w
    cov = (dst_c * w[:, None]).T @ src_c / total_w

    u, s, vt = np.linalg.svd(cov)
    d = np.ones(dim)
    if np.linalg.det(cov) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt

    scale = float(np.sum(s * d) / var_src)
    t = dst_mean - scale * (r @ src_mean)

    m = np.eye(3, dtype=np.float64)
    m[:2, :2] = scale * r
    m[:2, 2] = t
    return m


def weighted_fit_affine(src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted ordinary least squares affine fit. weights=None is
    equivalent to fit_transforms.fit_affine (uniform weights)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if weights is None:
        return fit_affine(src, dst)

    w = np.asarray(weights, dtype=np.float64)
    sqrt_w = np.sqrt(w)

    x = np.zeros((2 * n, 6), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    for i in range(n):
        x0, y0 = src[i]
        u, v = dst[i]
        x[2 * i] = [x0, y0, 1.0, 0.0, 0.0, 0.0]
        x[2 * i + 1] = [0.0, 0.0, 0.0, x0, y0, 1.0]
        y[2 * i] = u
        y[2 * i + 1] = v
        x[2 * i] *= sqrt_w[i]
        x[2 * i + 1] *= sqrt_w[i]
        y[2 * i] *= sqrt_w[i]
        y[2 * i + 1] *= sqrt_w[i]

    params, *_ = np.linalg.lstsq(x, y, rcond=None)
    m = np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return m


def _weighted_fit(src: np.ndarray, dst: np.ndarray, model: str, weights: Optional[np.ndarray]) -> np.ndarray:
    if model == "similarity":
        return weighted_fit_similarity(src, dst, weights)
    if model == "affine":
        return weighted_fit_affine(src, dst, weights)
    raise ValueError(f"Unknown model {model!r}, expected 'similarity' or 'affine'.")


# ---------------------------------------------------------------------------
# Robust (RANSAC) fit
# ---------------------------------------------------------------------------

def _ransac_fit(src: np.ndarray, dst: np.ndarray, model: str, ransac_thresh_px: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run cv2's RANSAC similarity/affine estimator. Returns (M, inlier_mask),
    both None if RANSAC failed to find a model."""
    src32 = src.astype(np.float32)
    dst32 = dst.astype(np.float32)
    if model == "similarity":
        m2x3, inliers = cv2.estimateAffinePartial2D(
            src32, dst32, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px
        )
    elif model == "affine":
        m2x3, inliers = cv2.estimateAffine2D(
            src32, dst32, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px
        )
    else:
        raise ValueError(f"Unknown model {model!r}, expected 'similarity' or 'affine'.")
    if m2x3 is None:
        return None, None
    m = np.vstack([m2x3, [0.0, 0.0, 1.0]])
    inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(src), dtype=bool)
    return m, inlier_mask


# ---------------------------------------------------------------------------
# Leave-one-out (always a plain, unweighted fit of the remaining points,
# matching fit_transforms.leave_one_out_similarity / leave_one_out_affine)
# ---------------------------------------------------------------------------

def _leave_one_out(src: np.ndarray, dst: np.ndarray, model: str) -> Optional[np.ndarray]:
    n = src.shape[0]
    if n < MIN_POINTS_LOO[model]:
        return None
    errors = np.zeros(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[:] = True
        mask[i] = False
        m = _weighted_fit(src[mask], dst[mask], model, weights=None)
        pred = apply_transform(m, src[i : i + 1])
        errors[i] = np.linalg.norm(pred[0] - dst[i])
    return errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fit_from_landmarks(
    src: np.ndarray,
    dst: np.ndarray,
    model: str = "similarity",
    robust: bool = False,
    weights: Optional[np.ndarray] = None,
    dst_um_per_px: Optional[float] = None,
    ransac_thresh_px: float = DEFAULT_RANSAC_THRESH_PX,
    min_inlier_fraction: float = MIN_INLIER_FRACTION,
) -> Dict:
    """Fit a similarity or affine transform from matched landmarks.

    src, dst: (N, 2) arrays, same convention as fit_transforms (dst = M @ src).
    model: "similarity" or "affine".
    robust: if True, fit with RANSAC (cv2.estimateAffinePartial2D for
      similarity, cv2.estimateAffine2D for affine); if the resulting inlier
      fraction is below min_inlier_fraction, falls back to the plain
      (weighted) fit over all points and reports status
      "low_inlier_fraction_fallback" rather than trusting a RANSAC result
      built from too few points.
    weights: optional per-landmark weights for the non-robust fit (and the
      robust fallback fit); ignored for leave-one-out (always unweighted,
      matching fit_transforms' existing convention).
    dst_um_per_px: if given, residuals are also reported in micrometers
      (residual_px * dst_um_per_px); this assumes dst is already at the
      pixel size this scale factor describes, documented by the caller.

    Returns a dict:
      M                 3x3 fitted matrix
      status            "ok", "low_inlier_fraction_fallback", or
                        "ransac_failed_fallback"
      inlier_mask       bool array (RANSAC inliers), all True if robust=False
                        or the fallback was used
      inlier_fraction   float
      residuals_px, residuals_um (None if dst_um_per_px not given)
      loo_residuals_px, loo_residuals_um (None if too few points for LOO)
      suspect           bool array, same length as src: True if a
                        landmark's own leave-one-out residual exceeds both
                        3x the median leave-one-out residual and 2 px (all
                        False if LOO could not be computed)
    """
    if model not in ("similarity", "affine"):
        raise ValueError(f"Unknown model {model!r}, expected 'similarity' or 'affine'.")
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < MIN_POINTS[model]:
        raise ValueError(f"Need >= {MIN_POINTS[model]} landmarks for model {model!r}, have {n}")

    status = "ok"
    inlier_mask = np.ones(n, dtype=bool)

    if robust:
        m_ransac, ransac_inliers = _ransac_fit(src, dst, model, ransac_thresh_px)
        if m_ransac is None:
            status = "ransac_failed_fallback"
            m = _weighted_fit(src, dst, model, weights)
        else:
            inlier_fraction_ransac = float(ransac_inliers.mean())
            if inlier_fraction_ransac < min_inlier_fraction:
                status = "low_inlier_fraction_fallback"
                m = _weighted_fit(src, dst, model, weights)
            else:
                m = m_ransac
                inlier_mask = ransac_inliers
    else:
        m = _weighted_fit(src, dst, model, weights)

    residuals_px = residuals(m, src, dst)
    residuals_um = residuals_px * dst_um_per_px if dst_um_per_px is not None else None

    loo_residuals_px = _leave_one_out(src, dst, model)
    loo_residuals_um = None
    suspect = np.zeros(n, dtype=bool)
    if loo_residuals_px is not None:
        if dst_um_per_px is not None:
            loo_residuals_um = loo_residuals_px * dst_um_per_px
        median_loo = float(np.median(loo_residuals_px))
        threshold_px = max(SUSPECT_MEDIAN_FACTOR * median_loo, 0.0)
        suspect = (loo_residuals_px > threshold_px) & (loo_residuals_px > SUSPECT_MIN_PX)

    return {
        "M": m,
        "status": status,
        "inlier_mask": inlier_mask,
        "inlier_fraction": float(inlier_mask.mean()),
        "residuals_px": residuals_px,
        "residuals_um": residuals_um,
        "loo_residuals_px": loo_residuals_px,
        "loo_residuals_um": loo_residuals_um,
        "suspect": suspect,
    }


# ---------------------------------------------------------------------------
# Landmark suggestion
# ---------------------------------------------------------------------------

def suggest_landmarks(src_inliers: np.ndarray, dst_inliers: np.ndarray, n: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin a set of automatic inlier matches to about n spatially well
    distributed landmarks, using the same grid-based rule already written
    for task 9's export (export.select_spatially_distributed), applied to
    the src (moving) side's own spatial spread.

    Returns (src_selected, dst_selected, selected_idx).
    """
    src_inliers = np.asarray(src_inliers, dtype=np.float64)
    dst_inliers = np.asarray(dst_inliers, dtype=np.float64)
    idx = select_spatially_distributed(src_inliers, n_target=n)
    return src_inliers[idx], dst_inliers[idx], idx
