"""Task 6, item 1: independent nuclei-based accuracy check.

H&E and COMET come from the same physical tissue section, so individual
nuclei correspond one to one between the two modalities. This module
detects nuclei independently in both images (same blob-detector settings,
defined in micrometers) and uses mutual nearest-neighbor matching between
transformed H&E nuclei and COMET nuclei as a check of registration accuracy
that does NOT use the provided matrix as ground truth.

This is a check, not a registration method: it only evaluates transforms
already produced elsewhere (the provided target, and the task 5
sift_intensity / akaze_intensity results at 1.0 um/px).

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import auto_baselines as ab
import auto_register as ar
import evaluation as ev
from fit_transforms import apply_transform

# Nuclear diameter about 6 to 10 um (task wording). skimage.feature.blob_log
# detects a blob of radius ~= sigma * sqrt(2), so sigma = (diameter / 2) /
# sqrt(2) = diameter / (2 * sqrt(2)).
_DIAM_MIN_UM = 6.0
_DIAM_MAX_UM = 10.0
SIGMA_MIN_UM = _DIAM_MIN_UM / (2.0 * np.sqrt(2.0))
SIGMA_MAX_UM = _DIAM_MAX_UM / (2.0 * np.sqrt(2.0))

MATCH_RADIUS_UM = 4.0
N_BOOTSTRAP = 2000
BOOTSTRAP_CI = 95.0


def detect_nuclei_centroids_working(img01: np.ndarray, um_per_px: float) -> np.ndarray:
    """Detect nuclei centroids with a fixed, documented blob-detector setting.

    Same settings (sigma range derived from a 6 to 10 um nuclear diameter,
    num_sigma=5, threshold=0.05) are used for H&E hematoxylin and COMET
    DAPI; only um_per_px (the image's own working resolution) differs,
    since sigma is specified in physical units. Returns an (N, 2) array of
    (x, y) centroids in the image's own working-resolution pixel units.
    """
    blobs = ar.detect_nuclei_blobs(
        img01,
        um_per_px,
        min_sigma_um=SIGMA_MIN_UM,
        max_sigma_um=SIGMA_MAX_UM,
        num_sigma=5,
        threshold=0.05,
    )
    if blobs.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([blobs[:, 1], blobs[:, 0]], axis=1)  # (y, x) -> (x, y)


def _to_native(xy_ws: np.ndarray, ws_um_per_px: float, native_um_per_px: float) -> np.ndarray:
    return xy_ws * (ws_um_per_px / native_um_per_px)


def mutual_nearest_neighbors(
    pred_xy: np.ndarray, target_xy: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual nearest-neighbor matching within a distance radius.

    A pair (i, j) is kept only if target_xy[j] is pred_xy[i]'s nearest
    neighbor AND pred_xy[i] is target_xy[j]'s nearest neighbor, and their
    distance is <= radius. Returns (idx_pred, idx_target, distances), all
    aligned, empty arrays if either input is empty.
    """
    if pred_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    tree_target = cKDTree(target_xy)
    tree_pred = cKDTree(pred_xy)

    dist_p2t, nn_p2t = tree_target.query(pred_xy, k=1)
    _dist_t2p, nn_t2p = tree_pred.query(target_xy, k=1)

    idx_pred = []
    idx_target = []
    dists = []
    for i, j in enumerate(nn_p2t):
        if nn_t2p[j] == i and dist_p2t[i] <= radius:
            idx_pred.append(i)
            idx_target.append(j)
            dists.append(dist_p2t[i])
    return np.array(idx_pred, dtype=int), np.array(idx_target, dtype=int), np.array(dists, dtype=float)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = BOOTSTRAP_CI, seed: int = 0):
    """Bootstrap the mean of values, returns (mean, ci_lo, ci_hi)."""
    if values.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    return float(values.mean()), float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))


def load_best_task5_matrices(data_dir: Path, core_id: str, auto_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Load the sift_intensity and akaze_intensity native matrices for one
    core at 1.0 um/px from the task 5 outputs/auto_baselines.csv table."""
    matrices = {}
    sub = auto_df[(auto_df["core_id"] == core_id) & (auto_df["resolution_um"] == 1.0)]
    for method in ("sift_intensity", "akaze_intensity"):
        row = sub[sub["method"] == method]
        if len(row) == 0 or pd.isna(row["m00"].iloc[0]):
            matrices[method] = None
            continue
        r = row.iloc[0]
        matrices[method] = np.array(
            [[r["m00"], r["m01"], r["m02"]], [r["m10"], r["m11"], r["m12"]], [0.0, 0.0, 1.0]]
        )
    return matrices


def run_nuclei_check_core(data_dir: Path, core_id: str, auto_df: pd.DataFrame) -> Dict:
    """Run the full nuclei-based accuracy check for one core at 1.0 um/px.

    Returns a dict with per-transform match statistics (for "target",
    "sift_intensity", "akaze_intensity") and the raw matched point pairs
    (native pixel coordinates, both sides, plus residual vectors in um)
    for the quiver figure.
    """
    images = ab.load_core_images(data_dir, core_id, 1.0)
    px_he = images["px_he"]
    px_comet = images["px_comet"]

    he_centroids_ws = detect_nuclei_centroids_working(images["hema"], images["he_um"])
    comet_centroids_ws = detect_nuclei_centroids_working(images["dapi_norm"], images["comet_um"])

    he_centroids_native = _to_native(he_centroids_ws, images["he_um"], px_he)
    comet_centroids_native = _to_native(comet_centroids_ws, images["comet_um"], px_comet)

    m_target_native = ev.target_he_to_comet(data_dir, core_id)
    matrices = {"target": m_target_native}
    matrices.update(load_best_task5_matrices(data_dir, core_id, auto_df))

    radius_px = MATCH_RADIUS_UM / px_comet

    per_transform: Dict[str, Dict] = {}
    for name, m in matrices.items():
        if m is None:
            per_transform[name] = {
                "status": "no_matrix",
                "n_pairs": 0,
                "fraction_matched": float("nan"),
                "mean_um": float("nan"),
                "median_um": float("nan"),
                "ci_lo_um": float("nan"),
                "ci_hi_um": float("nan"),
                "pred_pairs_native": None,
                "target_pairs_native": None,
            }
            continue

        pred_native = apply_transform(m, he_centroids_native)
        idx_pred, idx_target, dist_px = mutual_nearest_neighbors(pred_native, comet_centroids_native, radius_px)
        dist_um = dist_px * px_comet

        n_possible = min(he_centroids_native.shape[0], comet_centroids_native.shape[0])
        fraction_matched = float(dist_um.shape[0] / n_possible) if n_possible > 0 else float("nan")
        mean_um, ci_lo, ci_hi = bootstrap_mean_ci(dist_um)
        median_um = float(np.median(dist_um)) if dist_um.shape[0] > 0 else float("nan")

        per_transform[name] = {
            "status": "ok",
            "n_pairs": int(dist_um.shape[0]),
            "fraction_matched": fraction_matched,
            "mean_um": mean_um,
            "median_um": median_um,
            "ci_lo_um": ci_lo,
            "ci_hi_um": ci_hi,
            "pred_pairs_native": pred_native[idx_pred] if idx_pred.shape[0] else np.zeros((0, 2)),
            "target_pairs_native": comet_centroids_native[idx_target] if idx_target.shape[0] else np.zeros((0, 2)),
        }

    return {
        "core_id": core_id,
        "n_he_detected": int(he_centroids_native.shape[0]),
        "n_comet_detected": int(comet_centroids_native.shape[0]),
        "px_comet": px_comet,
        "per_transform": per_transform,
    }


def build_nuclei_check_table(results: List[Dict]) -> pd.DataFrame:
    """Flatten run_nuclei_check_core results into one row per core/transform."""
    rows = []
    for res in results:
        for name, stats in res["per_transform"].items():
            rows.append(
                {
                    "core_id": res["core_id"],
                    "transform": name,
                    "n_he_detected": res["n_he_detected"],
                    "n_comet_detected": res["n_comet_detected"],
                    "status": stats["status"],
                    "n_pairs": stats["n_pairs"],
                    "fraction_matched": stats["fraction_matched"],
                    "mean_um": stats["mean_um"],
                    "median_um": stats["median_um"],
                    "ci95_lo_um": stats["ci_lo_um"],
                    "ci95_hi_um": stats["ci_hi_um"],
                }
            )
    return pd.DataFrame(rows)


def best_automatic_name(core_id: str, auto_df: pd.DataFrame) -> str:
    """Pick sift_intensity or akaze_intensity, whichever has lower mean_um
    in the task 5 outputs/auto_baselines.csv table at 1.0 um/px, for use as
    the single "best automatic" transform shown in the quiver figure."""
    sub = auto_df[
        (auto_df["core_id"] == core_id)
        & (auto_df["resolution_um"] == 1.0)
        & (auto_df["method"].isin(["sift_intensity", "akaze_intensity"]))
        & auto_df["mean_um"].notna()
    ]
    if len(sub) == 0:
        return "sift_intensity"
    return sub.loc[sub["mean_um"].idxmin(), "method"]


def build_nuclei_check_figure(results: List[Dict], core_ids: List[str], auto_df: pd.DataFrame, out_path: Path) -> None:
    """Save outputs/nuclei_check.png: per-core quiver of residual vectors
    (H&E-mapped minus COMET nuclei) for the provided target transform and
    for the best automatic transform (task 5's lower-mean_um of
    sift_intensity/akaze_intensity at 1.0 um/px for that core)."""
    import matplotlib

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(core_ids), 2, figsize=(10, 4.5 * len(core_ids)))
    if len(core_ids) == 1:
        axs = axs[np.newaxis, :]

    by_core = {r["core_id"]: r for r in results}

    for i, core_id in enumerate(core_ids):
        res = by_core[core_id]
        px_comet = res["px_comet"]
        best_name = best_automatic_name(core_id, auto_df)

        for j, name in enumerate(["target", best_name]):
            ax = axs[i, j]
            stats = res["per_transform"].get(name)
            if stats is None or stats["status"] != "ok" or stats["n_pairs"] == 0:
                ax.set_title(f"{core_id}\n{name}: no pairs")
                ax.axis("off")
                continue
            pred = stats["pred_pairs_native"]
            tgt = stats["target_pairs_native"]
            residual_um = (pred - tgt) * px_comet

            step = max(1, pred.shape[0] // 150)
            ax.quiver(
                tgt[::step, 0],
                tgt[::step, 1],
                residual_um[::step, 0],
                residual_um[::step, 1],
                angles="xy",
                scale_units="xy",
                scale=0.02,
                color="tab:red",
                width=0.004,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(
                f"{core_id}, {name}\nresidual vectors (um), n={stats['n_pairs']}, "
                f"mean={stats['mean_um']:.2f} um"
            )
            ax.set_xlabel("COMET x (native px)")
            ax.set_ylabel("COMET y (native px)")

    fig.suptitle(
        "Nuclei-matched residual vectors: H&E-mapped minus COMET nuclei\n"
        "(independent accuracy check, not using the provided matrix as ground truth)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run_all(data_dir: Path, core_ids: List[str], auto_baselines_csv: Path) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run the nuclei check for every core; returns (table, raw results)."""
    auto_df = pd.read_csv(auto_baselines_csv)
    results = [run_nuclei_check_core(data_dir, core_id, auto_df) for core_id in core_ids]
    table = build_nuclei_check_table(results)
    return table, results
