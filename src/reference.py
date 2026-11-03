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


# ---------------------------------------------------------------------------
# Task 27: per-step similarity or affine, recomposed through the chain
#
# Each engine step "X_to_Y" in a result carries "step_matrix" (similarity)
# and "affine_matrix" (or None): X native pixels to Y native pixels (to the
# reference frame for a step onto the base reference). The matrix of layer X
# into the base reference frame is  M_X = M_Y @ step_X  (M_base = identity),
# which is what pipeline.run_sample builds. A choice is {source modality:
# "affine"}; every step not named stays similarity. These functions work on
# the engine's own (base) result, before apply_reference.
# ---------------------------------------------------------------------------

def _engine_steps(result: Dict) -> Dict[str, Dict]:
    return {d["step"].split("_to_", 1)[0]: d for d in result.get("diagnostics", []) if not d.get("derived")}


def _step_target(step: Dict) -> str:
    return step["step"].split("_to_", 1)[1]


def affine_available(result: Dict) -> List[str]:
    """Source modalities of steps whose deformation check says stretch or shear is likely and that have an affine."""
    from deformation import LINEAR, step_check
    return [m for m, d in _engine_steps(result).items()
            if d.get("affine_matrix") is not None and d.get("step_matrix") is not None and step_check(d) == LINEAR]


def chain_steps(result: Dict, modality: str) -> List[str]:
    """Source modalities of the engine steps that carry `modality` to the base reference, nearest first."""
    steps, base = _engine_steps(result), result.get("reference_modality")
    path, current = [], modality
    while current != base and current in steps and current not in path:
        path.append(current)
        current = _step_target(steps[current])
    return path


def compose_with_choice(result: Dict, choice: Dict[str, str]) -> Dict[str, Optional[np.ndarray]]:
    """{modality: matrix into the base reference} with the steps in `choice` ("affine") using their affine
    matrix and all other steps their similarity matrix, composed M_X = M_Y @ step_X. A layer is None when a step
    on its chain has no matrix."""
    steps, base = _engine_steps(result), result.get("reference_modality")
    memo: Dict[str, Optional[np.ndarray]] = {base: np.eye(3)}

    def matrix(modality: str) -> Optional[np.ndarray]:
        if modality in memo:
            return memo[modality]
        memo[modality] = None
        step = steps.get(modality)
        if step is not None and step.get("step_matrix") is not None:
            use = step.get("affine_matrix") if choice.get(modality) == "affine" else None
            step_m = np.asarray(use if use is not None else step["step_matrix"], dtype=float)
            parent = matrix(_step_target(step))
            memo[modality] = None if parent is None else parent @ step_m
        return memo[modality]

    return {m: matrix(m) for m in result.get("matrices", {})}


def with_transform_choice(result: Dict, choice: Dict[str, str]) -> Dict:
    """A copy of the engine result using `choice` ({source modality: "affine"}); entries that are not available
    are dropped. With an empty choice the stored similarity matrices are returned exactly as the engine made them."""
    usable = set(affine_available(result))
    choice = {m: "affine" for m, v in choice.items() if v == "affine" and m in usable}
    out = dict(result)
    out["transform_choice"] = choice
    similarity = result.get("matrices_similarity") or result.get("matrices", {})
    out["matrices_similarity"] = similarity
    if not choice:
        out["matrices"] = dict(similarity)
    else:
        out["matrices"] = compose_with_choice(result, choice)
    return out


def layer_transform_type(result: Dict, modality: str) -> str:
    choice = result.get("transform_choice") or {}
    return "affine" if any(choice.get(m) == "affine" for m in chain_steps(result, modality)) else "similarity"


def layer_matrix_variants(result: Dict, reference: str, pixel_sizes: Optional[Dict] = None) -> Dict[str, Dict]:
    """{modality: {"transform_type", "matrix_similarity", "matrix_affine"}} against `reference`. matrix_affine uses
    the affine for the steps where the deformation check offers it (the option's meaning), and is None for a layer
    whose chain has no such step (it would equal the similarity matrix)."""
    similarity = with_transform_choice({**result, "transform_choice": {}}, {})
    offered = affine_available(result)
    affine_matrices = compose_with_choice(result, {m: "affine" for m in offered})
    affine_result = dict(result)
    affine_result["matrices"] = affine_matrices
    sim_ref = apply_reference(similarity, reference, pixel_sizes)["matrices"]
    aff_ref = apply_reference(affine_result, reference, pixel_sizes)["matrices"]
    cur_ref = apply_reference(result, reference, pixel_sizes)["matrices"]
    out = {}
    for m in result.get("matrices", {}):
        if m == reference:
            continue
        a, s, c = aff_ref.get(m), sim_ref.get(m), cur_ref.get(m)
        differs = a is not None and s is not None and not np.array_equal(a, s)
        used = c is not None and s is not None and not np.array_equal(c, s)
        out[m] = {"transform_type": "affine" if used else "similarity",
                  "matrix_similarity": s, "matrix_affine": a if differs else None}
    return out
