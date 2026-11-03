"""Task 12, items 2 and 3: compare pipeline.run_sample matrices with the
provided matrices and with src/chain.py, in the Xenium reference frame.

Every comparison uses the same function and the same tissue grids as
task 9 (chain.compare_to_provided_in_reference_frame, with the COMET mask
from prep.tissue_mask on the DAPI image and the H&E mask from
auto_baselines.load_core_images), so the numbers are directly comparable
with outputs/chain_results.csv. Distances are in micrometers measured in
the reference frame (reference pixel * 0.2125).

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import auto_baselines as ab
import chain
import evaluation as ev
import prep

REFERENCE_UM_PER_PX = chain.REFERENCE_UM_PER_PX
WORKING_UM_PER_PX = chain.WORKING_UM_PER_PX


def core_grids(data_dir: Path, core_id: str) -> Dict:
    """Tissue grids (native pixel coordinates) used for every comparison
    of one core, built exactly as chain.run_core builds its masks."""
    images = ab.load_core_images(data_dir, core_id, WORKING_UM_PER_PX)
    mask_comet = prep.tissue_mask(images["dapi_norm"], images["comet_um"])
    return {
        "comet": ev.sample_grid_in_mask(
            mask_comet, images["comet_um"], images["px_comet"], step_px_mask=chain.GRID_STEP_PX_MASK
        ),
        "he": ev.sample_grid_in_mask(
            images["mask_he"], images["he_um"], images["px_he"], step_px_mask=chain.GRID_STEP_PX_MASK
        ),
        "mask_comet": mask_comet,
        "images": images,
    }


def difference_um(m_a: Optional[np.ndarray], m_b: Optional[np.ndarray], grid_native: np.ndarray) -> Dict:
    """Mean/median/max distance (um, reference frame) between two
    native-to-reference matrices applied to the same grid."""
    if m_a is None or m_b is None:
        return {"mean_um": np.nan, "median_um": np.nan, "max_um": np.nan, "n_points": 0}
    mean_um, median_um, max_um, n_used = ev.transform_difference_um(
        m_a, m_b, grid_native, comet_px_um=REFERENCE_UM_PER_PX
    )
    return {"mean_um": mean_um, "median_um": median_um, "max_um": max_um, "n_points": n_used}


def provided_matrix(data_dir: Path, core_id: str, modality: str) -> np.ndarray:
    return chain.load_provided_matrix(chain._matrix_path(data_dir, core_id, modality))


def versus_provided_table(
    data_dir: Path, core_id: str, grids: Dict, matrices_by_source: Dict[str, Dict[str, Optional[np.ndarray]]]
) -> pd.DataFrame:
    """One row per (source, modality): difference of that source's matrix
    from the provided matrix_{modality}.csv. matrices_by_source =
    {"new_pipeline": {"comet": M, "he": M}, "old_pipeline": ..., "chain": ...}."""
    rows = []
    for source, mats in matrices_by_source.items():
        for modality in ("comet", "he"):
            m_provided = provided_matrix(data_dir, core_id, modality)
            d = difference_um(mats.get(modality), m_provided, grids[modality])
            rows.append({"core_id": core_id, "modality": modality, "source": source, **d})
    return pd.DataFrame(rows)


def pairwise_table(core_id: str, grids: Dict, matrices_by_source: Dict[str, Dict[str, Optional[np.ndarray]]], pairs: List[tuple]) -> pd.DataFrame:
    """One row per (pair, modality): difference between two sources'
    matrices (for example new_pipeline vs chain)."""
    rows = []
    for a, b in pairs:
        for modality in ("comet", "he"):
            d = difference_um(matrices_by_source[a].get(modality), matrices_by_source[b].get(modality), grids[modality])
            rows.append({"core_id": core_id, "modality": modality, "pair": f"{a} vs {b}", **d})
    return pd.DataFrame(rows)
