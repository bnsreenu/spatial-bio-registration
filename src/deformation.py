"""Task 27: tissue deformation check and optional affine, per registration step.

For a step already registered by engine.register_pair (similarity, unchanged),
analyze_step re-reads the same raw ratio-test matches (the same
auto_register.feature_based_similarity call the engine makes for the
winning hypothesis) and fits a full affine with cv2.estimateAffine2D
(RANSAC, 3 px). It also measures how much the inlier residuals of the
similarity are structured. Nothing here changes the similarity result.

Thresholds live in this one place (DEFORMATION_CONFIG). They come from the
Task 25 stress test on two cores and are not validated on real data.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

import auto_register as ar
import engine
import evaluation as ev
import prep

DEFORMATION_CONFIG = {
    "r2_threshold": 0.05,            # adjusted R squared of a quadratic surface fitted to inlier residuals
    "linear_threshold_um": 1.0,      # mean similarity-versus-affine difference over the tissue grid
    "affine_ransac_px": 3.0,
    "grid_step_px": 8,
}

NONE, LINEAR, NONLINEAR = "none", "linear", "nonlinear"
LABELS = {
    NONE: "none detected",
    LINEAR: "stretch or shear likely (an affine fit would correct this)",
    NONLINEAR: "non-linear deformation possible (an affine fit would not correct this)",
}
SHORT_LABELS = {NONE: "none detected", LINEAR: "stretch or shear likely", NONLINEAR: "non-linear deformation possible"}
ORDER = {NONE: 0, NONLINEAR: 1, LINEAR: 2}


def _apply(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ m[:2, :2].T + m[:2, 2]


def quad_r2(pos: np.ndarray, r: np.ndarray):
    """(R squared, adjusted R squared) of a quadratic surface (6 terms per
    component, normalized coordinates) fitted to residual vectors r at pos.
    Same implementation as src/task25_stress_test.py."""
    p = (pos - pos.mean(0)) / max(np.ptp(pos, axis=0).max() / 2, 1e-9)
    x, y = p[:, 0], p[:, 1]
    f = np.column_stack([np.ones(len(p)), x, y, x * x, x * y, y * y])
    ssr = sst = 0.0
    for k in range(2):
        c = np.linalg.lstsq(f, r[:, k], rcond=None)[0]
        ssr += ((r[:, k] - f @ c) ** 2).sum()
        sst += ((r[:, k] - r[:, k].mean()) ** 2).sum()
    obs = 1 - ssr / max(sst, 1e-12)
    n = len(p)
    return float(obs), float(1 - (1 - obs) * (n - 1) / max(n - 6, 1))


def classify(linear_mean_um: Optional[float], quad_r2_adj: Optional[float], config: Optional[Dict] = None) -> Optional[str]:
    """NONE, LINEAR or NONLINEAR; None when the numbers are unavailable."""
    cfg = config or DEFORMATION_CONFIG
    if linear_mean_um is None or quad_r2_adj is None:
        return None
    if linear_mean_um != linear_mean_um or quad_r2_adj != quad_r2_adj:
        return None
    if linear_mean_um >= cfg["linear_threshold_um"]:
        return LINEAR
    if quad_r2_adj >= cfg["r2_threshold"]:
        return NONLINEAR
    return NONE


def step_check(step: Dict) -> Optional[str]:
    return classify(step.get("linear_deformation_mean_um"), step.get("quad_r2_adj"))


def sample_check(steps: List[Dict]) -> Optional[str]:
    """The most serious check among steps (linear, then non-linear, then none); None if no step has one."""
    found = [c for c in (step_check(s) for s in steps) if c]
    return max(found, key=lambda c: ORDER[c]) if found else None


def analyze_step(fixed_img, moving_img, reg: Dict, px_fixed_um: float, px_moving_um: float, method: str) -> Dict:
    """Extra per-step numbers for a register_pair result `reg` (same arguments
    as the call that produced it). Returns {"affine_reg": 3x3 or None,
    "linear_deformation_mean_um", "linear_deformation_max_um", "quad_r2_adj",
    "inlier_rms_um"}; entries are None when they cannot be computed. The
    affine is in the same frame as reg["M"] (moving px at px_moving_um to
    fixed px at px_fixed_um); the caller converts it like the similarity."""
    out = {"affine_reg": None, "linear_deformation_mean_um": None, "linear_deformation_max_um": None,
           "quad_r2_adj": None, "inlier_rms_um": None}
    if reg.get("M") is None:
        return out
    m = np.asarray(reg["M"], dtype=float)
    src, dst = reg.get("src_pts"), reg.get("dst_pts")
    if src is not None and len(src) >= 10:
        res = (np.asarray(dst) - _apply(m, np.asarray(src))) * px_fixed_um
        out["inlier_rms_um"] = float(np.sqrt((res ** 2).sum(1).mean()))
        out["quad_r2_adj"] = quad_r2(np.asarray(src), res)[1]

    fixed_ws, moving_ws, ws_fixed_um, ws_moving_um, _res = engine._prepare_working_pair(
        fixed_img, moving_img, px_fixed_um, px_moving_um)
    mov = cv2.flip(np.asarray(moving_ws, dtype=np.float32), 1) if reg.get("mirrored") else moving_ws
    raw = ar.feature_based_similarity(mov, fixed_ws, method=method, return_points=True)
    if raw.get("pts_src") is None or len(raw["pts_src"]) < 4:
        return out
    a2, _inl = cv2.estimateAffine2D(raw["pts_src"], raw["pts_dst"], method=cv2.RANSAC,
                                    ransacReprojThreshold=DEFORMATION_CONFIG["affine_ransac_px"])
    if a2 is None:
        return out
    a_ws = np.vstack([a2, [0.0, 0.0, 1.0]])
    if reg.get("mirrored"):
        a_ws = a_ws @ engine.mirror_matrix(moving_ws.shape[1])
    a_reg = engine._native_matrix(a_ws, ws_moving_um, ws_fixed_um, px_moving_um, px_fixed_um)
    out["affine_reg"] = a_reg

    mask = prep.tissue_mask(np.asarray(moving_ws, dtype=float), ws_moving_um)
    grid = ev.sample_grid_in_mask(mask, ws_moving_um, px_moving_um, step_px_mask=DEFORMATION_CONFIG["grid_step_px"])
    mean_um, _median, max_um, n_used = ev.transform_difference_um(a_reg, m, grid, comet_px_um=px_fixed_um)
    if n_used > 0:
        out["linear_deformation_mean_um"], out["linear_deformation_max_um"] = float(mean_um), float(max_um)
    return out


def check_text(step_or_steps) -> str:
    """Plain-language line for the confidence card, with the numbers."""
    steps = [step_or_steps] if isinstance(step_or_steps, dict) else list(step_or_steps)
    check = sample_check(steps)
    if check is None:
        return "Deformation check: not available for this result."
    if check == NONE:
        return "Deformation check: none detected."
    if check == LINEAR:
        worst = max((s["linear_deformation_mean_um"] for s in steps if step_check(s) == LINEAR))
        return (f"Deformation check: possible stretch or shear of about {worst:.1f} um across the tissue. "
                "An affine fit would correct this.")
    r2 = max((s["quad_r2_adj"] for s in steps if step_check(s) == NONLINEAR))
    return (f"Deformation check: possible non-linear deformation (residual pattern R squared {r2:.2f}). "
            "An affine fit would not correct this.")
