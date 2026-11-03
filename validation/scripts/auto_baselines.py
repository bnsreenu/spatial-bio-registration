"""Task 5 driver: score Method A and Method B baselines against the target.

Runs, for every core and every working resolution, the same fixed set of
methods (no per-core tuning):

  sift_intensity, akaze_intensity   - Method A on normalized images
  sift_density, akaze_density       - Method A on nucleus-density maps
  coarse_to_fine                    - Method B (scale/rotation search +
                                       phase correlation + ECC refinement)

plus one noise_floor_loocv row per core (resolution-independent, built from
the provided keypoints, see evaluation.noise_floor_he_to_comet_diffs).

Every candidate's working-resolution similarity matrix is converted back to
native H&E-to-COMET pixels and scored with evaluation.transform_difference_um
against the provided-matrix target, restricted to H&E tissue mask points
that also land on real COMET data (task 5, item 0b).

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

import auto_register as ar
import data_access as da
import evaluation as ev
import prep
from fit_transforms import decompose_similarity

MATRIX_COLS = ["m00", "m01", "m02", "m10", "m11", "m12"]


def _matrix_to_row_cols(m: np.ndarray) -> Dict:
    return {
        "m00": float(m[0, 0]),
        "m01": float(m[0, 1]),
        "m02": float(m[0, 2]),
        "m10": float(m[1, 0]),
        "m11": float(m[1, 1]),
        "m12": float(m[1, 2]),
    }


def _empty_row(core_id: str, resolution_um, method: str) -> Dict:
    row = {
        "core_id": core_id,
        "resolution_um": resolution_um,
        "method": method,
        "status": None,
        "n_matches": None,
        "n_inliers": None,
        "inlier_ratio": None,
        "n_blobs_src": None,
        "n_blobs_dst": None,
        "coarse_scale": None,
        "coarse_rotation_deg": None,
        "ecc_status": None,
        "ecc_correlation": None,
        "mean_um": None,
        "median_um": None,
        "max_um": None,
        "n_points_used": None,
        "scale": None,
        "rotation_deg": None,
        "tx_native_px": None,
        "ty_native_px": None,
        "notes": "",
    }
    for c in MATRIX_COLS:
        row[c] = None
    return row


def load_core_images(data_dir: Path, core_id: str, working_um_per_px: float) -> Dict:
    """Load H&E hematoxylin and COMET DAPI at a common working resolution.

    Returns a dict with hema, dapi_norm (normalized 0..1 images, resampled
    to the same pixel size with data_access.resample_to_um), dapi_raw
    (resampled, NOT normalized, for the "has real data" valid mask), the
    common achieved um_per_px, native pixel sizes, and the H&E tissue mask
    at this resolution.
    """
    zp_he = list((data_dir / core_id / "he").glob("*.ome.zarr"))[0]
    zp_comet = list((data_dir / core_id / "comet").glob("*.ome.zarr"))[0]

    he_rgb, he_um, _ = da.load_he_rgb(zp_he, working_um_per_px)
    dapi, dapi_um, _, dapi_idx = da.load_comet_dapi(zp_comet, working_um_per_px)

    common_um = max(he_um, dapi_um)
    he_rs, he_rs_um = da.resample_to_um(he_rgb, he_um, common_um)
    dapi_rs, dapi_rs_um = da.resample_to_um(dapi, dapi_um, common_um)

    hema = prep.he_to_hematoxylin(he_rs)
    dapi_norm = prep.normalize_dapi(dapi_rs)
    mask_he = prep.tissue_mask(hema, he_rs_um)

    px_he = da.get_pixel_size_um(zp_he)
    px_comet = da.get_pixel_size_um(zp_comet)

    return {
        "hema": hema,
        "dapi_norm": dapi_norm,
        "dapi_raw": dapi_rs,
        "he_um": he_rs_um,
        "comet_um": dapi_rs_um,
        "px_he": px_he,
        "px_comet": px_comet,
        "mask_he": mask_he,
        "dapi_channel_index": dapi_idx,
        "zp_he": zp_he,
        "zp_comet": zp_comet,
    }


def score_candidate(
    m_working: np.ndarray,
    images: Dict,
    m_target_native: np.ndarray,
    grid_he_native: np.ndarray,
    step_grid_um: float = None,
) -> Dict:
    """Convert a working-resolution src->dst matrix to native and score it."""
    m_native = ev.to_working_matrix(
        m_working, images["he_um"], images["comet_um"], images["px_he"], images["px_comet"]
    )
    comet_valid = images["dapi_raw"] > 0
    mean_um, median_um, max_um, n_used = ev.transform_difference_um(
        m_native,
        m_target_native,
        grid_he_native,
        comet_px_um=images["px_comet"],
        comet_valid_mask=comet_valid,
        comet_valid_mask_um_per_px=images["comet_um"],
    )
    scale, rotation_deg, tx, ty = decompose_similarity(m_native)
    return {
        "m_native": m_native,
        "mean_um": mean_um,
        "median_um": median_um,
        "max_um": max_um,
        "n_points_used": n_used,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "tx_native_px": tx,
        "ty_native_px": ty,
    }


def run_one_core_resolution(
    data_dir: Path, core_id: str, working_um_per_px: float
) -> List[Dict]:
    """Run all five fixed methods for one core at one working resolution."""
    rows: List[Dict] = []
    images = load_core_images(data_dir, core_id, working_um_per_px)
    m_target_native = ev.target_he_to_comet(data_dir, core_id)
    grid_he_native = ev.sample_grid_in_mask(
        images["mask_he"], images["he_um"], images["px_he"], step_px_mask=8
    )

    def finish(row, scored):
        row.update(
            {
                "mean_um": scored["mean_um"],
                "median_um": scored["median_um"],
                "max_um": scored["max_um"],
                "n_points_used": scored["n_points_used"],
                "scale": scored["scale"],
                "rotation_deg": scored["rotation_deg"],
                "tx_native_px": scored["tx_native_px"],
                "ty_native_px": scored["ty_native_px"],
            }
        )
        row.update(_matrix_to_row_cols(scored["m_native"]))
        return row, scored["m_native"]

    matrices: Dict[str, np.ndarray] = {}

    # Method A on normalized intensity images.
    for method in ("sift", "akaze"):
        name = f"{method}_intensity"
        row = _empty_row(core_id, working_um_per_px, name)
        res = ar.feature_based_similarity(images["hema"], images["dapi_norm"], method=method)
        row.update(
            status=res["status"],
            n_matches=res["n_matches"],
            n_inliers=res["n_inliers"],
            inlier_ratio=res["inlier_ratio"],
        )
        if res["M"] is not None:
            scored = score_candidate(res["M"], images, m_target_native, grid_he_native)
            row, matrices[name] = finish(row, scored)
        rows.append(row)

    # Method A on nucleus-density maps.
    hema_density, n_blobs_he = ar.nucleus_density_map(images["hema"], images["he_um"])
    dapi_density, n_blobs_comet = ar.nucleus_density_map(images["dapi_norm"], images["comet_um"])
    for method in ("sift", "akaze"):
        name = f"{method}_density"
        row = _empty_row(core_id, working_um_per_px, name)
        row["n_blobs_src"] = n_blobs_he
        row["n_blobs_dst"] = n_blobs_comet
        res = ar.feature_based_similarity(hema_density, dapi_density, method=method)
        row.update(
            status=res["status"],
            n_matches=res["n_matches"],
            n_inliers=res["n_inliers"],
            inlier_ratio=res["inlier_ratio"],
        )
        if res["M"] is not None:
            scored = score_candidate(res["M"], images, m_target_native, grid_he_native)
            row, matrices[name] = finish(row, scored)
        rows.append(row)

    # Method B: coarse-to-fine.
    name = "coarse_to_fine"
    row = _empty_row(core_id, working_um_per_px, name)
    resb = ar.coarse_to_fine_register(images["hema"], images["dapi_norm"], images["he_um"])
    row.update(
        status=resb["status"],
        n_blobs_src=resb["n_blobs_src"],
        n_blobs_dst=resb["n_blobs_dst"],
        coarse_scale=resb["coarse_scale"],
        coarse_rotation_deg=resb["coarse_rotation_deg"],
        ecc_status=resb["ecc_status"],
        ecc_correlation=resb["ecc_correlation"],
    )
    if resb["M"] is not None:
        scored = score_candidate(resb["M"], images, m_target_native, grid_he_native)
        row, matrices[name] = finish(row, scored)
    rows.append(row)

    return rows, matrices, images, m_target_native, grid_he_native


def noise_floor_row(data_dir: Path, core_id: str) -> Dict:
    """Build the resolution-independent noise-floor reference row for one core."""
    row = _empty_row(core_id, None, "noise_floor_loocv")
    # A generous native-pixel grid covering the full H&E field; the noise
    # floor is about the target matrices themselves, not about any one
    # image's tissue mask, so a plain dense grid is used here (not the
    # H&E tissue mask used for method scoring).
    zp_he = list((data_dir / core_id / "he").glob("*.ome.zarr"))[0]
    px_he = da.get_pixel_size_um(zp_he)
    array_native, _eff_um, _level = da.open_level(zp_he, px_he)
    h_native, w_native = array_native.shape[-2], array_native.shape[-1]
    xs, ys = np.meshgrid(
        np.linspace(0, w_native, 40), np.linspace(0, h_native, 40)
    )
    grid = np.column_stack([xs.ravel(), ys.ravel()])

    result = ev.noise_floor_he_to_comet_diffs(data_dir, core_id, grid, comet_px_um=0.28)
    row.update(
        status=result["status"],
        mean_um=result["mean_um"],
        median_um=result["median_um"],
        max_um=result["max_um"],
        n_points_used=result["n_values"],
    )
    row["notes"] = "; ".join(result["notes"]) if result["notes"] else ""
    return row


def run_all(data_dir: Path, core_ids: List[str], resolutions_um=(2.0, 1.0)) -> pd.DataFrame:
    """Run every method on every core and resolution, plus noise-floor rows.

    Returns a DataFrame with one row per (core, resolution, method) and one
    noise-floor row per core. Does not raise on a single core/resolution
    failure; records status/error text in that row instead and continues.
    """
    all_rows: List[Dict] = []
    for core_id in core_ids:
        for ws in resolutions_um:
            try:
                rows, _matrices, _images, _target, _grid = run_one_core_resolution(
                    data_dir, core_id, ws
                )
                all_rows.extend(rows)
            except Exception as exc:  # noqa: BLE001 - record and continue
                row = _empty_row(core_id, ws, "run_failed")
                row["status"] = f"error: {exc}"
                all_rows.append(row)
        try:
            all_rows.append(noise_floor_row(data_dir, core_id))
        except Exception as exc:  # noqa: BLE001
            row = _empty_row(core_id, None, "noise_floor_loocv")
            row["status"] = f"error: {exc}"
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METHOD_ORDER = ["sift_intensity", "akaze_intensity", "sift_density", "akaze_density", "coarse_to_fine"]
METHOD_COLORS = {
    "sift_intensity": "#1f77b4",
    "akaze_intensity": "#ff7f0e",
    "sift_density": "#2ca02c",
    "akaze_density": "#d62728",
    "coarse_to_fine": "#9467bd",
}


def build_auto_baselines_figure(df: pd.DataFrame, core_ids: List[str], out_path: Path) -> None:
    """Save outputs/auto_baselines.png: mean_um per method/resolution/core.

    One panel per core, bars grouped by method, one bar pair per working
    resolution (2.0 and 1.0 um/px), log-scaled y-axis (errors span about
    0.2 to 400 um), with the per-core noise-floor mean_um drawn as a
    dashed horizontal line for scale.
    """
    import matplotlib

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.pyplot as plt

    resolutions = sorted(df.loc[df["method"] != "noise_floor_loocv", "resolution_um"].dropna().unique())
    n_methods = len(METHOD_ORDER)
    width = 0.8 / max(1, len(resolutions))

    fig, axs = plt.subplots(1, len(core_ids), figsize=(5.0 * len(core_ids), 5.0), sharey=True)
    if len(core_ids) == 1:
        axs = [axs]

    for ax, core_id in zip(axs, core_ids):
        sub = df[df["core_id"] == core_id]
        for r_i, res in enumerate(resolutions):
            means = []
            for method in METHOD_ORDER:
                match = sub[(sub["method"] == method) & (sub["resolution_um"] == res)]
                val = match["mean_um"].iloc[0] if len(match) and pd.notna(match["mean_um"].iloc[0]) else np.nan
                means.append(val)
            xs = np.arange(n_methods) + (r_i - (len(resolutions) - 1) / 2.0) * width
            ax.bar(
                xs,
                means,
                width=width,
                label=f"{res:g} um/px" if ax is axs[0] else None,
                color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                alpha=0.5 + 0.5 * (r_i / max(1, len(resolutions) - 1)),
                edgecolor="black",
                linewidth=0.5,
            )

        noise = sub[sub["method"] == "noise_floor_loocv"]
        if len(noise) and pd.notna(noise["mean_um"].iloc[0]):
            ax.axhline(noise["mean_um"].iloc[0], color="black", linestyle="--", linewidth=1.0, label="noise floor")

        ax.set_yscale("log")
        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right")
        ax.set_title(f"{core_id}\nmean residual vs target (um, log scale)")
        ax.set_ylabel("mean residual (um)")
        ax.grid(True, which="both", axis="y", alpha=0.3)

    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Automatic H&E-to-COMET baselines\nsame fixed parameters for all cores", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def best_method_per_core(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the row with the lowest mean_um per core among actual methods
    (excludes noise_floor_loocv and failed/NaN rows)."""
    candidates = df[(df["method"] != "noise_floor_loocv") & df["mean_um"].notna()]
    idx = candidates.groupby("core_id")["mean_um"].idxmin()
    return candidates.loc[idx].reset_index(drop=True)


def build_best_overlay_figure(
    data_dir: Path, core_id: str, m_native_candidate: np.ndarray, out_path: Path, working_um_per_px: float = 2.0
) -> None:
    """Save a red/green overlay of H&E vs COMET DAPI warped with the best
    automatic candidate transform for one core (QC only, not a metric)."""
    import matplotlib

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.pyplot as plt

    images = load_core_images(data_dir, core_id, working_um_per_px)
    m_working = ev.to_working_matrix(
        m_native_candidate, images["px_he"], images["px_comet"], images["he_um"], images["comet_um"]
    )
    m_comet_to_he_working = np.linalg.inv(m_working)
    he_shape_xy = (images["hema"].shape[1], images["hema"].shape[0])
    dapi_warped = cv2.warpAffine(
        images["dapi_norm"].astype(np.float32),
        m_comet_to_he_working[:2, :],
        he_shape_xy,
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    overlay = np.zeros((*images["hema"].shape, 3), dtype=np.float64)
    overlay[..., 0] = images["hema"]
    overlay[..., 1] = dapi_warped

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    axs[0].imshow(images["hema"], cmap="gray")
    axs[0].set_title(f"H&E hematoxylin\n{core_id}")
    axs[1].imshow(images["dapi_norm"], cmap="gray")
    axs[1].set_title(f"COMET DAPI (unregistered)\n{core_id}")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay, best automatic method\nH&E=red, COMET DAPI=green")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Best automatic baseline overlay: {core_id}", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
