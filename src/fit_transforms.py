"""Fit and evaluate landmark-based transforms for the UnumLocalia dataset.

This module reproduces the similarity matrices shipped with the dataset
(matrix_he.csv, matrix_comet.csv) from the matched keypoint files
(keypoints_he.csv, keypoints_comet.csv), and checks whether an unconstrained
affine fit does meaningfully better than a similarity fit.

Keypoint file columns are fixedX, fixedY, alignmentX, alignmentY.
"alignment" points are in the native pixel grid of the source image (H&E or
COMET). "fixed" points are in the shared canvas frame. The provided matrices
map alignment (source) pixels into the fixed frame:

    [fixedX, fixedY, 1]^T = M @ [alignmentX, alignmentY, 1]^T

Only functions are used here, no classes, per project convention
(see spatial_registration/CONTEXT.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_keypoints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a keypoints CSV and return (src_xy, dst_xy) as float arrays.

    src is the "alignment" (source image) point, dst is the "fixed" (shared
    canvas) point, matching the direction of the provided matrices.
    """
    df = pd.read_csv(path)
    src = df[["alignmentX", "alignmentY"]].to_numpy(dtype=np.float64)
    dst = df[["fixedX", "fixedY"]].to_numpy(dtype=np.float64)
    return src, dst


def load_provided_matrix(path: Path) -> np.ndarray:
    """Load a provided 3x3 (or 2x3) affine matrix CSV with no header."""
    raw = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if raw.shape == (3, 3):
        return raw
    if raw.shape == (2, 3):
        return np.vstack([raw, [0.0, 0.0, 1.0]])
    raise ValueError(f"Unexpected matrix shape {raw.shape} in {path}")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit a similarity transform (uniform scale, rotation, translation).

    Uses the Umeyama (1991) closed-form least squares solution. Returns a
    3x3 matrix M such that dst_hat = M @ [src, 1]^T minimizes the sum of
    squared Euclidean residuals.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n

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


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit an unconstrained 2D affine transform by ordinary least squares."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    x = np.zeros((2 * n, 6), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    for i in range(n):
        x0, y0 = src[i]
        u, v = dst[i]
        x[2 * i] = [x0, y0, 1.0, 0.0, 0.0, 0.0]
        x[2 * i + 1] = [0.0, 0.0, 0.0, x0, y0, 1.0]
        y[2 * i] = u
        y[2 * i + 1] = v

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


# ---------------------------------------------------------------------------
# Applying and evaluating
# ---------------------------------------------------------------------------

def apply_transform(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    return (h @ m[:2, :].T)


def residuals(m: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return the per-point Euclidean distance (pixels) between M(src) and dst."""
    pred = apply_transform(m, src)
    return np.linalg.norm(pred - np.asarray(dst, dtype=np.float64), axis=1)


def decompose_similarity(m: np.ndarray) -> Tuple[float, float, float, float]:
    """Decompose a similarity-like 3x3 matrix into (scale, rotation_deg, tx, ty).

    rotation_deg is the counter-clockwise rotation angle in image coordinates
    (x right, y down), read from the upper-left 2x2 block. If M carries shear
    this only reports the rotation implied by its first column.
    """
    a = float(m[0, 0])
    b = float(m[1, 0])
    scale = float(np.hypot(a, b))
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    tx = float(m[0, 2])
    ty = float(m[1, 2])
    return scale, rotation_deg, tx, ty


def leave_one_out_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Leave-one-out cross-validation error (pixels) for the similarity fit.

    For each point, fit a similarity transform on all other points and
    measure the distance between the predicted and true held-out point.
    Returns one residual per point. Requires at least 4 points (3 to fit,
    1 held out) to be meaningful; raises ValueError otherwise.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"Need >= 4 keypoints for leave-one-out, have {n}")

    errors = np.zeros(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[:] = True
        mask[i] = False
        m = fit_similarity(src[mask], dst[mask])
        pred = apply_transform(m, src[i : i + 1])
        errors[i] = np.linalg.norm(pred[0] - dst[i])
    return errors


def leave_one_out_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Leave-one-out cross-validation error (pixels) for the affine fit.

    Same scheme as leave_one_out_similarity, but fits an unconstrained
    affine transform. Requires at least 7 points (6 to fit, 1 held out).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 7:
        raise ValueError(f"Need >= 7 keypoints for leave-one-out, have {n}")

    errors = np.zeros(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[:] = True
        mask[i] = False
        m = fit_affine(src[mask], dst[mask])
        pred = apply_transform(m, src[i : i + 1])
        errors[i] = np.linalg.norm(pred[0] - dst[i])
    return errors
