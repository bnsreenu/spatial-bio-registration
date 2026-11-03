"""Task 9, item 2: reproduce matrix_he.csv / matrix_comet.csv equivalents,
with the Xenium pixel grid as the reference frame, using only automatic
registration (src/engine.py's register_pair; task 5 / task 8 settings and
parameters unchanged, no per-core tuning).

Reference frame: R = Xenium micrometers / 0.2125, i.e. a point at Xenium
coordinate (x, y) um has reference-frame pixel coordinate (x / 0.2125,
y / 0.2125). Task 8 item 0 found that this dataset's own shared "fixed"
canvas is, for practical purposes, exactly this frame (see
outputs/task8_notes.md): composing the provided matrix_comet.csv with the
provided Xenium-to-COMET affine gives almost exactly a pure scale of
1 / 0.2125 px/um with negligible rotation, shear, or translation on every
core. R is therefore a reasonable, independently defined stand-in for the
dataset's own fixed canvas, usable even when nothing about that canvas is
assumed in advance.

  D = diag(1/0.2125, 1/0.2125, 1)                   Xenium um -> reference px

  B: Xenium um -> COMET native px                    from register_pair on
                                                      (render_nuclei image, COMET DAPI)
  M_comet_auto = D @ inv(B)                          COMET native px -> reference px

  T_he_to_comet: H&E native px -> COMET native px     from register_pair on
                                                      (H&E hematoxylin, COMET DAPI)
  M_he_auto = M_comet_auto @ T_he_to_comet            H&E native px -> reference px

Both register_pair calls are run at 1.0 um/px (task 5's working
resolution), with the fixed primary method "akaze" and mirror_test=True
(task 9 default); register_pair's own output is then explicitly converted
from that working resolution to each modality's true native pixel size
with evaluation.to_working_matrix, the same pattern already used in
src/xenium_register.py and src/auto_baselines.py.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import auto_baselines as ab
import engine
import evaluation as ev
import prep
import render
from fit_transforms import apply_transform, load_provided_matrix

REFERENCE_UM_PER_PX = 0.2125  # Xenium Explorer DAPI pixel grid; see xenium_transform.XENIUM_EXPLORER_UM_PER_PX
D_XENIUM_TO_REFERENCE = np.diag([1.0 / REFERENCE_UM_PER_PX, 1.0 / REFERENCE_UM_PER_PX, 1.0])

WORKING_UM_PER_PX = 1.0
METHOD = "akaze"
GRID_STEP_PX_MASK = 8


def _matrix_path(data_dir: Path, core_id: str, modality: str) -> Path:
    return data_dir / core_id / modality / f"matrix_{modality}.csv"


# ---------------------------------------------------------------------------
# The two register_pair steps
# ---------------------------------------------------------------------------

def register_xenium_to_comet(data_dir: Path, core_id: str, comet: Dict) -> Dict:
    """register_pair(render_nuclei image, COMET DAPI); build B = Xenium um -> COMET native px.

    render_nuclei is rendered at exactly 1.0 um/px, so its pixel grid has
    unit scale relative to Xenium micrometers, but render.py's S matrix
    (render pixel -> Xenium um) carries a nonzero translation (the
    bounding-box origin minx, miny used to keep the rendered canvas
    small): register_pair's output (render pixel -> COMET working px)
    must therefore be composed with inv(S) to get Xenium um -> COMET
    working px, before the COMET side is converted to COMET's true
    native pixel size.
    """
    render_img, s_mat = render.render_nuclei(data_dir, core_id, px_um=WORKING_UM_PER_PX)
    reg = engine.register_pair(
        img_fixed=comet["dapi_norm"],
        img_moving=render_img,
        px_fixed_um=comet["um_per_px"],
        px_moving_um=WORKING_UM_PER_PX,
        method=METHOD,
        mirror_test=True,
    )
    b_mat = None
    if reg["M"] is not None:
        render_px_to_comet_working = reg["M"]
        xenium_um_to_comet_working = render_px_to_comet_working @ np.linalg.inv(s_mat)
        b_mat = ev.to_working_matrix(
            xenium_um_to_comet_working, 1.0, comet["um_per_px"], 1.0, comet["px_comet_native"]
        )
    return {"reg": reg, "B": b_mat, "S_render": s_mat, "render_img": render_img}


def register_he_to_comet(data_dir: Path, core_id: str, comet: Dict, he: Dict) -> Dict:
    """register_pair(H&E hematoxylin, COMET DAPI); build T_he_to_comet =
    H&E native px -> COMET native px."""
    reg = engine.register_pair(
        img_fixed=comet["dapi_norm"],
        img_moving=he["hema"],
        px_fixed_um=comet["um_per_px"],
        px_moving_um=he["he_um"],
        method=METHOD,
        mirror_test=True,
    )
    t_mat = None
    if reg["M"] is not None:
        t_mat = ev.to_working_matrix(
            reg["M"], he["he_um"], comet["um_per_px"], he["px_he"], comet["px_comet_native"]
        )
    return {"reg": reg, "T": t_mat}


# ---------------------------------------------------------------------------
# Scoring against the provided matrices
# ---------------------------------------------------------------------------

def _signed_mean_offset_um(m_candidate: np.ndarray, m_target: np.ndarray, grid_points: np.ndarray, um_per_px: float):
    """Task 10, item 0b: signed mean (dx, dy) offset in um between
    m_candidate and m_target applied to the same points, in whatever frame
    um_per_px converts to. This is a plain diagnostic (candidate - target,
    averaged per axis over the grid), not an attempt to interpret the
    direction of any bias; returns (nan, nan) if grid_points is empty.
    """
    if grid_points.shape[0] == 0:
        return float("nan"), float("nan")
    pred_candidate = apply_transform(m_candidate, grid_points)
    pred_target = apply_transform(m_target, grid_points)
    diff_px = pred_candidate - pred_target
    dx_um = float(np.mean(diff_px[:, 0])) * um_per_px
    dy_um = float(np.mean(diff_px[:, 1])) * um_per_px
    return dx_um, dy_um


def compare_to_provided_in_reference_frame(
    data_dir: Path, core_id: str, m_auto: np.ndarray, modality: str, mask: np.ndarray, mask_um: float, native_um: float
) -> Dict:
    """transform_difference_um between m_auto and the provided matrix_{modality}.csv,
    on a grid inside that modality's own tissue mask, with distances
    measured in the reference frame (reference px * 0.2125 = um)."""
    m_provided = load_provided_matrix(_matrix_path(data_dir, core_id, modality))
    grid_native = ev.sample_grid_in_mask(mask, mask_um, native_um, step_px_mask=GRID_STEP_PX_MASK)
    mean_um, median_um, max_um, n_used = ev.transform_difference_um(
        m_auto, m_provided, grid_native, comet_px_um=REFERENCE_UM_PER_PX
    )
    dx_um, dy_um = _signed_mean_offset_um(m_auto, m_provided, grid_native, REFERENCE_UM_PER_PX)
    return {
        "mean_um": mean_um,
        "median_um": median_um,
        "max_um": max_um,
        "n_points_used": n_used,
        "dx_um": dx_um,
        "dy_um": dy_um,
    }


def compare_he_to_comet_alone(
    data_dir: Path, core_id: str, t_auto: np.ndarray, mask_he: np.ndarray, he_um: float, px_he: float, px_comet: float
) -> Dict:
    """transform_difference_um between the automatic T_he_to_comet and the
    provided-matrix-derived H&E-to-COMET target, in COMET native pixels."""
    m_target = ev.target_he_to_comet(data_dir, core_id)
    grid_he_native = ev.sample_grid_in_mask(mask_he, he_um, px_he, step_px_mask=GRID_STEP_PX_MASK)
    mean_um, median_um, max_um, n_used = ev.transform_difference_um(
        t_auto, m_target, grid_he_native, comet_px_um=px_comet
    )
    return {"mean_um": mean_um, "median_um": median_um, "max_um": max_um, "n_points_used": n_used}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_core(data_dir: Path, core_id: str) -> Dict:
    """Run the full chain for one core: Xenium-to-COMET and H&E-to-COMET
    automatic registration, compose M_comet_auto and M_he_auto, and score
    both against the provided matrices."""
    images = ab.load_core_images(data_dir, core_id, WORKING_UM_PER_PX)
    comet = {
        "dapi_norm": images["dapi_norm"],
        "um_per_px": images["comet_um"],
        "px_comet_native": images["px_comet"],
    }
    he = {"hema": images["hema"], "he_um": images["he_um"], "px_he": images["px_he"]}
    mask_comet = prep.tissue_mask(images["dapi_norm"], images["comet_um"])

    xen = register_xenium_to_comet(data_dir, core_id, comet)
    he_reg = register_he_to_comet(data_dir, core_id, comet, he)

    result = {
        "core_id": core_id,
        "images": images,
        "mask_comet": mask_comet,
        "xenium_reg": xen["reg"],
        "he_reg": he_reg["reg"],
        "B": xen["B"],
        "T": he_reg["T"],
        "render_img": xen["render_img"],
        "S_render": xen["S_render"],
        "M_comet_auto": None,
        "M_he_auto": None,
        "comet_vs_provided": None,
        "he_vs_provided": None,
        "he_to_comet_alone": None,
    }

    if xen["B"] is not None:
        try:
            b_inv = np.linalg.inv(xen["B"])
        except np.linalg.LinAlgError:
            result["comet_vs_provided"] = {"status": "singular_B", "mean_um": None, "median_um": None, "max_um": None, "n_points_used": 0}
        else:
            m_comet_auto = D_XENIUM_TO_REFERENCE @ b_inv
            result["M_comet_auto"] = m_comet_auto
            result["comet_vs_provided"] = compare_to_provided_in_reference_frame(
                data_dir, core_id, m_comet_auto, "comet", mask_comet, images["comet_um"], images["px_comet"]
            )

    if result["M_comet_auto"] is not None and he_reg["T"] is not None:
        m_he_auto = result["M_comet_auto"] @ he_reg["T"]
        result["M_he_auto"] = m_he_auto
        result["he_vs_provided"] = compare_to_provided_in_reference_frame(
            data_dir, core_id, m_he_auto, "he", images["mask_he"], images["he_um"], images["px_he"]
        )

    if he_reg["T"] is not None:
        result["he_to_comet_alone"] = compare_he_to_comet_alone(
            data_dir, core_id, he_reg["T"], images["mask_he"], images["he_um"], images["px_he"], images["px_comet"]
        )

    return result


def run_all(data_dir: Path, core_ids: List[str]) -> List[Dict]:
    return [run_core(data_dir, core_id) for core_id in core_ids]


# ---------------------------------------------------------------------------
# Flat results table
# ---------------------------------------------------------------------------

def _step_columns(prefix: str, reg: Dict) -> Dict:
    diff = reg["second_opinion"]["diff_um"]
    return {
        f"{prefix}_status": reg["status"],
        f"{prefix}_n_matches": reg["n_matches"],
        f"{prefix}_n_inliers": reg["n_inliers"],
        f"{prefix}_inlier_ratio": reg["inlier_ratio"],
        f"{prefix}_mirrored": reg["mirrored"],
        f"{prefix}_confidence": reg["confidence"],
        f"{prefix}_warnings": "; ".join(reg["warnings"]) if reg["warnings"] else "",
        f"{prefix}_second_opinion_method": reg["second_opinion"]["method"],
        f"{prefix}_second_opinion_status": reg["second_opinion"]["status"],
        f"{prefix}_second_opinion_diff_mean_um": diff["mean_um"] if diff else None,
    }


def _compare_columns(prefix: str, comp: Dict, has_offset: bool = False) -> Dict:
    if comp is None:
        cols = {
            f"{prefix}_mean_um": None,
            f"{prefix}_median_um": None,
            f"{prefix}_max_um": None,
            f"{prefix}_n_points_used": 0,
        }
        if has_offset:
            cols[f"{prefix}_dx_um"] = None
            cols[f"{prefix}_dy_um"] = None
        return cols
    cols = {
        f"{prefix}_mean_um": comp.get("mean_um"),
        f"{prefix}_median_um": comp.get("median_um"),
        f"{prefix}_max_um": comp.get("max_um"),
        f"{prefix}_n_points_used": comp.get("n_points_used", 0),
    }
    # Task 10, item 0b: signed mean (dx, dy) offset, reported only where
    # compare_to_provided_in_reference_frame computed it (comet_vs_provided,
    # he_vs_provided); not interpreted, just reported.
    if has_offset:
        cols[f"{prefix}_dx_um"] = comp.get("dx_um")
        cols[f"{prefix}_dy_um"] = comp.get("dy_um")
    return cols


def build_results_table(results: List[Dict]) -> pd.DataFrame:
    """One row per core, flattening both register_pair diagnostics and the
    three transform_difference_um comparisons."""
    rows = []
    for res in results:
        row = {"core_id": res["core_id"]}
        row.update(_step_columns("xenium_to_comet", res["xenium_reg"]))
        row.update(_step_columns("he_to_comet", res["he_reg"]))
        row.update(_compare_columns("comet_vs_provided", res["comet_vs_provided"], has_offset=True))
        row.update(_compare_columns("he_vs_provided", res["he_vs_provided"], has_offset=True))
        row.update(_compare_columns("he_to_comet_alone", res["he_to_comet_alone"]))
        rows.append(row)
    return pd.DataFrame(rows)

