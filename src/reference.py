"""Task 16A: choose which modality is the reference, without changing the
engine.

pipeline.run_sample registers every layer onto one base reference (Xenium
when present). Its result holds, for each other layer X, a matrix
M_X mapping X's native pixels into the base reference's pixel frame. The
base reference itself has the identity matrix (it is not in
result["matrices"]).

For a chosen reference R, the matrix of any layer X into R's pixel frame is

    M_X_to_R = inv(M_R) @ M_X          (M_base = identity)

so everything is a composition of the existing chain matrices and their
inverses. When R is the base reference the result is returned unchanged,
so the current outputs are reproduced exactly (no arithmetic is done).

Pixel frames: a modality's own frame is its native pixel grid, except
Xenium without a raster image, whose frame is Xenium micrometers / 0.2125
(the Xenium Explorer DAPI grid), as documented in pipeline.py.

Only functions are used here. No UI imports.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

XENIUM_FRAME_UM = 0.2125


def modalities_in(result: Dict) -> List[str]:
    """Every modality in the result, the base reference first."""
    base = result.get("reference_modality")
    return ([base] if base else []) + [m for m in result.get("matrices", {}) if m != base]


def reference_matrices(result: Dict, reference: str) -> Dict[str, Optional[np.ndarray]]:
    """{modality: 3x3 matrix mapping that modality's pixels into the pixel
    frame of `reference`}, for every modality except `reference`. A value
    is None when it cannot be computed (the layer, or the reference, has no
    matrix, or a matrix is singular). Raises ValueError if `reference` is
    not a modality of this result."""
    base = result.get("reference_modality")
    matrices = result.get("matrices", {})
    if reference not in modalities_in(result):
        raise ValueError(f"{reference!r} is not part of this result")
    if reference == base:
        return {m: (None if v is None else np.asarray(v, dtype=float)) for m, v in matrices.items()}

    def to_base(modality: str) -> Optional[np.ndarray]:
        if modality == base:
            return np.eye(3)
        m = matrices.get(modality)
        return None if m is None else np.asarray(m, dtype=float)

    out: Dict[str, Optional[np.ndarray]] = {}
    ref_m = to_base(reference)
    inv_ref = None
    if ref_m is not None:
        try:
            inv_ref = np.linalg.inv(ref_m)
        except np.linalg.LinAlgError:
            inv_ref = None
    for modality in modalities_in(result):
        if modality == reference:
            continue
        m = to_base(modality)
        out[modality] = None if (m is None or inv_ref is None) else inv_ref @ m
    return out


def _reversed_step(step: Dict) -> Dict:
    src, dst = step["step"].split("_to_", 1)
    rev = dict(step)
    rev["step"] = f"{dst}_to_{src}"
    rev["src_pts"], rev["dst_pts"] = step.get("dst_pts"), step.get("src_pts")
    rev["src_frame"], rev["dst_frame"] = step.get("dst_frame"), step.get("src_frame")
    rev["derived"] = True
    return rev


def apply_reference(result: Dict, reference: str, pixel_sizes: Optional[Dict] = None) -> Dict:
    """A copy of the pipeline result expressed against `reference`. If it is
    already the result's reference, `result` itself is returned.

    Otherwise: matrices come from reference_matrices; reference_modality and
    reference_pixel_size_um are those of the chosen reference
    (pixel_sizes[reference], or None if unknown; Xenium without a raster
    image uses 0.2125); "base_reference" records the modality the engine
    registered onto. For a layer that has no "{layer}_to_*" step (for
    example Xenium, the engine's own reference), the reverse of a step that
    touches it is added, marked derived, so keypoints can be exported
    (points are exchanged between src and dst, nothing is re-matched)."""
    if reference == result.get("reference_modality"):
        return result
    matrices = reference_matrices(result, reference)
    pixel_sizes = pixel_sizes or {}
    out = dict(result)
    out["matrices"] = matrices
    out["base_reference"] = result.get("reference_modality")
    out["base_reference_pixel_size_um"] = result.get("reference_pixel_size_um")
    out["reference_modality"] = reference
    px = pixel_sizes.get(reference)
    if px is None and reference == "xenium":
        px = XENIUM_FRAME_UM
    out["reference_pixel_size_um"] = px
    diagnostics = copy.deepcopy(list(result.get("diagnostics", [])))
    have_source = {d["step"].split("_to_", 1)[0] for d in diagnostics}
    for modality in matrices:
        if modality in have_source:
            continue
        touching = next((d for d in diagnostics if d["step"].endswith(f"_to_{modality}")), None)
        if touching is not None:
            diagnostics.append(_reversed_step(touching))
    out["diagnostics"] = diagnostics
    return out


def default_reference(modalities) -> "tuple[Optional[str], str]":
    """(modality, one-line reason) for the default reference among the loaded
    modalities: Xenium, else COMET, else H&E."""
    loaded = set(modalities)
    if "xenium" in loaded:
        return "xenium", "Xenium is the usual reference for Xenium Explorer."
    if "comet" in loaded:
        return "comet", "No Xenium data is loaded, so COMET is the reference."
    if "he" in loaded:
        return "he", "Only H&E is loaded, so H&E is the reference."
    return None, ""
