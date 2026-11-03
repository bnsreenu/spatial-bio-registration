"""Task 11, item 3: a one-call registration pipeline built on engine.py.

run_sample(sample, reference="auto", ...) registers every other layer of a
sample (a group_samples-shaped dict: {"sample_id", "layers": {modality:
{"path", "pixel_size_um", "channel"}}}) onto one reference layer, using
engine.register_pair (task 9/10 settings, unchanged: method="akaze",
mirror_test=True) and io_layers.describe_input/open_layer for reading each
layer at a common working resolution (WORKING_UM_PER_PX, task 5's 1.0
um/px default).

Reference frame:
  - If the reference layer is a real raster image (H&E, fluorescence, or a
    real Xenium morphology image), the reference frame is that image's own
    native pixel grid (descriptor["pixel_size_um"]), and every returned
    matrix maps a layer's native pixels to that grid.
  - If the reference layer is Xenium with no morphology image (true for
    every core in data/UnumLocalia), there is no native raster pixel size
    to use. The reference frame is then Xenium micrometers / 0.2125 (the
    Xenium Explorer DAPI pixel grid convention; REFERENCE_UM_PER_PX below),
    exactly matching src/chain.py's convention, so this pipeline's results
    can be checked against outputs/chain_results.csv (task 11, item 5).
    This is a deliberate, documented simplification, not a claim that
    0.2125 um/px is this dataset's only possible Xenium reference frame.

Chaining: when the reference is a rendered Xenium image and a sample also
has a "comet" layer, "comet" is registered directly to the rendered Xenium
image (a nucleus-density match, like src/chain.py's
register_xenium_to_comet), and any other layer (e.g. "he") is registered
to "comet" (a more reliable histology-to-fluorescence match) and then
composed through the comet-to-reference matrix, i.e. a two-step chain, the
same path src/chain.py takes for H&E. If "comet" is absent or its match to
the reference fails, other layers fall back to a direct match against the
rendered Xenium image, with a warning that this is a less reliable path
(this fallback is implemented but not exercised by data/UnumLocalia, which
always has all three layers).

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
No UI imports are used.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import engine
import evaluation as ev
import io_layers
import render

WORKING_UM_PER_PX = 1.0
REFERENCE_UM_PER_PX = 0.2125  # Xenium Explorer DAPI pixel grid; see chain.py
METHOD = "akaze"

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# Single registration steps
# ---------------------------------------------------------------------------

def describe_layer(layer: Dict) -> Dict:
    """describe_input for a sample layer ({"path", "pixel_size_um",
    "channel"}). Task 14: if the file carries no usable pixel size and the
    layer holds one entered by the user (layer["pixel_size_um"]), that value
    is used as the layer's native pixel size and a note is added to the
    descriptor's warnings. A pixel size that the file does provide is never
    overridden, and nothing is ever guessed: with no pixel size from either
    source the descriptor is returned as is (pixel_size_um None)."""
    descriptor = io_layers.describe_input(Path(layer["path"]))
    entered = layer.get("pixel_size_um")
    if entered is not None and descriptor["kind"] != "xenium" and not io_layers._plausible(descriptor.get("pixel_size_um")):
        descriptor["pixel_size_um"] = float(entered)
        descriptor["warnings"] = list(descriptor.get("warnings") or []) + [
            f"pixel size {float(entered):g} um/px was entered by the user; the file has none."
        ]
    return descriptor


def _open_working(descriptor: Dict, channel=None):
    img, achieved_um = io_layers.open_layer(descriptor, WORKING_UM_PER_PX, channel=channel)
    return img, achieved_um


def _step_diagnostics(reg: Dict, step_name: str) -> Dict:
    diff = reg["second_opinion"]["diff_um"]
    return {
        "step": step_name,
        "status": reg["status"],
        "n_matches": reg["n_matches"],
        "n_inliers": reg["n_inliers"],
        "inlier_ratio": reg["inlier_ratio"],
        "mirrored": reg["mirrored"],
        "confidence": reg["confidence"],
        "second_opinion_method": reg["second_opinion"]["method"],
        "second_opinion_status": reg["second_opinion"]["status"],
        "second_opinion_diff_mean_um": diff["mean_um"] if diff else None,
        "engine_resampled": bool(reg.get("resampled", False)),
        "warnings": list(reg["warnings"]),
        # Matched inlier points of this step (task 12). Filled in by the
        # caller, in this step's own pixel frames (see src_frame/dst_frame).
        "src_pts": np.zeros((0, 2)),
        "dst_pts": np.zeros((0, 2)),
        "match_ids": np.zeros((0,), dtype=np.int64),
        "src_frame": None,
        "dst_frame": None,
    }


def _register_direct(
    fixed_img: np.ndarray, fixed_ws_um: float, fixed_native_um: Optional[float],
    moving_img: np.ndarray, moving_ws_um: float, moving_native_um: Optional[float],
    step_name: str,
) -> "tuple[Optional[np.ndarray], Dict]":
    """Register moving_img onto fixed_img at working resolution, then
    convert the result to moving_native -> fixed_native (native pixel
    coordinates on both sides), via evaluation.to_working_matrix (the same
    conversion engine.register_pair itself uses internally, applied here a
    second time to go from the working grid actually matched to each
    descriptor's own native pixel grid; see chain.py's register_he_to_comet
    for the same pattern). The inlier points are converted the same way
    and stored in the step's diagnostics (src_pts in the moving layer's
    native pixels, dst_pts in the fixed layer's native pixels)."""
    reg = engine.register_pair(
        img_fixed=fixed_img, img_moving=moving_img,
        px_fixed_um=fixed_ws_um, px_moving_um=moving_ws_um,
        method=METHOD, mirror_test=True,
    )
    diag = _step_diagnostics(reg, step_name)
    m_native = None
    if reg["M"] is not None:
        if fixed_native_um is not None and moving_native_um is not None:
            m_native = ev.to_working_matrix(reg["M"], moving_ws_um, fixed_ws_um, moving_native_um, fixed_native_um)
            diag["src_pts"] = reg["src_pts"] * (moving_ws_um / moving_native_um)
            diag["dst_pts"] = reg["dst_pts"] * (fixed_ws_um / fixed_native_um)
            diag["src_frame"] = "moving layer native pixels"
            diag["dst_frame"] = "fixed layer native pixels"
        else:
            diag["warnings"].append(
                f"{step_name}: native pixel size unknown for one side; "
                "the returned matrix is left in the working-resolution grid "
                f"({WORKING_UM_PER_PX:g} um/px) instead of a true native grid."
            )
            m_native = reg["M"]
            diag["src_pts"] = reg["src_pts"]
            diag["dst_pts"] = reg["dst_pts"]
            diag["src_frame"] = "moving working-resolution pixels"
            diag["dst_frame"] = "fixed working-resolution pixels"
    diag["match_ids"] = reg["match_ids"]
    return m_native, diag


def _register_to_rendered_xenium_reference(
    xenium_folder: Path, layer_img: np.ndarray, layer_ws_um: float, layer_native_um: Optional[float],
    step_name: str,
) -> "tuple[Optional[np.ndarray], Dict]":
    """Register layer_img (the fixed image) against a rendered Xenium
    nucleus image (the moving image), and compose the result into a
    layer_native -> reference(Xenium um / 0.2125) matrix, following the
    same construction as chain.py's register_xenium_to_comet /
    M_comet_auto = D @ inv(B)."""
    render_img, s_mat = render.render_nuclei_in_folder(xenium_folder, px_um=WORKING_UM_PER_PX)
    reg = engine.register_pair(
        img_fixed=layer_img, img_moving=render_img,
        px_fixed_um=layer_ws_um, px_moving_um=WORKING_UM_PER_PX,
        method=METHOD, mirror_test=True,
    )
    diag = _step_diagnostics(reg, step_name)
    m_native_to_ref = None
    if reg["M"] is not None and layer_native_um is not None:
        render_px_to_layer_ws = reg["M"]
        xenium_um_to_layer_ws = render_px_to_layer_ws @ np.linalg.inv(s_mat)
        b_mat = ev.to_working_matrix(xenium_um_to_layer_ws, 1.0, layer_ws_um, 1.0, layer_native_um)
        try:
            b_inv = np.linalg.inv(b_mat)
        except np.linalg.LinAlgError:
            diag["warnings"].append(f"{step_name}: Xenium-to-layer matrix B is singular, cannot invert.")
        else:
            d_mat = np.diag([1.0 / REFERENCE_UM_PER_PX, 1.0 / REFERENCE_UM_PER_PX, 1.0])
            m_native_to_ref = d_mat @ b_inv
            # Points: the layer (fixed image here) side in layer native
            # pixels, the render side (moving) via S into Xenium um, then
            # into reference pixels, so src_pts -> dst_pts is the same
            # direction as m_native_to_ref.
            diag["src_pts"] = reg["dst_pts"] * (layer_ws_um / layer_native_um)
            xen_um = reg["src_pts"] @ s_mat[:2, :2].T + s_mat[:2, 2]
            diag["dst_pts"] = xen_um / REFERENCE_UM_PER_PX
            diag["match_ids"] = reg["match_ids"]
            diag["src_frame"] = "layer native pixels"
            diag["dst_frame"] = "reference pixels (Xenium um / 0.2125)"
    elif reg["M"] is not None and layer_native_um is None:
        diag["warnings"].append(f"{step_name}: layer has no native pixel size, cannot build a native-to-reference matrix.")
    return m_native_to_ref, diag


# ---------------------------------------------------------------------------
# run_sample
# ---------------------------------------------------------------------------

def run_sample(
    sample: Dict,
    reference: str = "auto",
    progress: Optional[Callable[[float, str], None]] = None,
    cancel=None,
) -> Dict:
    """Register every non-reference layer of sample onto the reference
    layer. sample = {"sample_id", "layers": {modality: {"path",
    "pixel_size_um", "channel"}}} (samples.group_samples's output).

    reference="auto" picks "xenium" if present, else the first modality in
    sample["layers"] (insertion order, fixed he/comet/xenium by
    samples.group_samples).

    progress(fraction, message), if given, is called once before each
    registration step (fraction in [0, 1)) and once more at the end
    (fraction = 1.0). cancel, if given, is a threading.Event checked
    before each step; if set, returns immediately with status
    "cancelled" and whatever matrices/diagnostics were already computed.

    Returns a dict: sample_id, status ("ok" / "cancelled" / "failed"),
    reference_modality, reference_pixel_size_um, matrices (dict modality
    -> 3x3 array or None, native layer px -> reference px), diagnostics
    (list of per-step dicts; since task 12 each also carries the matched
    inlier points of that step: src_pts, dst_pts as float64 (n, 2) arrays,
    match_ids, and src_frame/dst_frame naming the pixel frames; for a
    step "{modality}_to_*", src_pts are in that modality's native pixels),
    confidence (the lowest per-step confidence,
    or "low" if nothing succeeded), warnings (list of str), runtime_s,
    error (str or None).
    """
    t0 = time.time()
    sample_id = sample.get("sample_id", "<unknown>")
    layers = sample.get("layers", {})
    warnings: List[str] = []
    diagnostics: List[Dict] = []
    matrices: Dict[str, Optional[np.ndarray]] = {}

    try:
        if not layers:
            return _finish(sample_id, "failed", None, None, {}, [], warnings + ["sample has no layers"], t0, "no layers")

        reference_modality = reference
        if reference == "auto":
            reference_modality = "xenium" if "xenium" in layers else next(iter(layers))
        if reference_modality not in layers:
            return _finish(
                sample_id, "failed", reference_modality, None, {}, [],
                warnings + [f"reference modality {reference_modality!r} not present in this sample"],
                t0, "reference modality missing",
            )

        other_modalities = [m for m in layers if m != reference_modality]
        n_steps = max(len(other_modalities), 1)
        step_i = 0

        def _report(message: str) -> None:
            if progress is not None:
                progress(step_i / n_steps, message)

        def _check_cancel() -> bool:
            return cancel is not None and cancel.is_set()

        ref_layer = layers[reference_modality]
        ref_descriptor = describe_layer(ref_layer)
        ref_rendered_xenium = (
            ref_descriptor["kind"] == "xenium" and ref_descriptor.get("xenium_source") == "rendered"
        )
        if ref_rendered_xenium and "comet" in other_modalities:
            # Process comet first so later layers (e.g. he) can chain
            # through comet_to_ref_matrix instead of falling back to a
            # direct (less reliable) match against the rendered Xenium image.
            other_modalities = ["comet"] + [m for m in other_modalities if m != "comet"]

        if ref_rendered_xenium:
            reference_native_um = REFERENCE_UM_PER_PX
            warnings.append(
                "reference is a rendered Xenium image with no native raster pixel size; "
                f"the reference frame is Xenium micrometers / {REFERENCE_UM_PER_PX:g} "
                "(the Xenium Explorer DAPI grid convention), matching chain.py."
            )
            ref_working_img = None  # built lazily per non-reference layer via render_nuclei_in_folder
            ref_working_um = None
        else:
            reference_native_um = ref_descriptor.get("pixel_size_um")
            ref_working_img, ref_working_um = _open_working(ref_descriptor, channel=ref_layer.get("channel"))

        # When chaining H&E through COMET, cache comet's working image/matrix.
        comet_to_ref_matrix = None
        comet_working_img = None
        comet_working_um = None
        comet_native_um = None

        for modality in other_modalities:
            if _check_cancel():
                return _finish(
                    sample_id, "cancelled", reference_modality, reference_native_um,
                    matrices, diagnostics, warnings, t0, None,
                )
            _report(f"registering {modality} to {reference_modality}")

            layer = layers[modality]
            layer_descriptor = describe_layer(layer)

            if ref_rendered_xenium:
                if modality == "comet":
                    comet_native_um = layer_descriptor.get("pixel_size_um")
                    comet_working_img, comet_working_um = _open_working(layer_descriptor, channel=layer.get("channel"))
                    m_native, diag = _register_to_rendered_xenium_reference(
                        Path(ref_layer["path"]), comet_working_img, comet_working_um, comet_native_um,
                        step_name="comet_to_xenium",
                    )
                    comet_to_ref_matrix = m_native
                    matrices[modality] = m_native
                    diagnostics.append(diag)
                    warnings.extend(diag["warnings"])
                elif comet_to_ref_matrix is not None:
                    layer_native_um = layer_descriptor.get("pixel_size_um")
                    layer_working_img, layer_working_um = _open_working(layer_descriptor, channel=layer.get("channel"))
                    m_layer_to_comet, diag = _register_direct(
                        comet_working_img, comet_working_um, comet_native_um,
                        layer_working_img, layer_working_um, layer_native_um,
                        step_name=f"{modality}_to_comet",
                    )
                    diagnostics.append(diag)
                    warnings.extend(diag["warnings"])
                    if m_layer_to_comet is not None:
                        matrices[modality] = comet_to_ref_matrix @ m_layer_to_comet
                    else:
                        matrices[modality] = None
                else:
                    # Fallback: comet absent or its match to the reference
                    # failed; register this layer directly to the rendered
                    # Xenium image. Not exercised by data/UnumLocalia.
                    warnings.append(
                        f"{modality}: registering directly to the rendered Xenium reference "
                        "(comet is absent or failed to register to it); this path is less "
                        "reliable and is not exercised by data/UnumLocalia."
                    )
                    layer_native_um = layer_descriptor.get("pixel_size_um")
                    layer_working_img, layer_working_um = _open_working(layer_descriptor, channel=layer.get("channel"))
                    m_native, diag = _register_to_rendered_xenium_reference(
                        Path(ref_layer["path"]), layer_working_img, layer_working_um, layer_native_um,
                        step_name=f"{modality}_to_xenium",
                    )
                    matrices[modality] = m_native
                    diagnostics.append(diag)
                    warnings.extend(diag["warnings"])
            elif layer_descriptor["kind"] == "xenium" and layer_descriptor.get("xenium_source") == "rendered":
                # Reference is a raster image; this layer is Xenium with no
                # morphology image. Render it at the working resolution and
                # register directly; since it has no native pixel size, the
                # render pixel grid (at WORKING_UM_PER_PX) is used in place
                # of a native grid, documented as a limitation.
                render_img, _s_mat = render.render_nuclei_in_folder(Path(layer["path"]), px_um=WORKING_UM_PER_PX)
                m_native, diag = _register_direct(
                    ref_working_img, ref_working_um, reference_native_um,
                    render_img, WORKING_UM_PER_PX, WORKING_UM_PER_PX,
                    step_name=f"xenium_to_{reference_modality}",
                )
                diag["warnings"].append(
                    "xenium has no native raster pixel size; its matrix maps rendered pixels "
                    f"at {WORKING_UM_PER_PX:g} um/px, not a true native Xenium grid."
                )
                matrices[modality] = m_native
                diagnostics.append(diag)
                warnings.extend(diag["warnings"])
            else:
                layer_native_um = layer_descriptor.get("pixel_size_um")
                layer_working_img, layer_working_um = _open_working(layer_descriptor, channel=layer.get("channel"))
                m_native, diag = _register_direct(
                    ref_working_img, ref_working_um, reference_native_um,
                    layer_working_img, layer_working_um, layer_native_um,
                    step_name=f"{modality}_to_{reference_modality}",
                )
                matrices[modality] = m_native
                diagnostics.append(diag)
                warnings.extend(diag["warnings"])

            step_i += 1

        if progress is not None:
            progress(1.0, "done")

        overall_confidence = _overall_confidence(diagnostics, matrices)
        status = "ok" if any(m is not None for m in matrices.values()) else "failed"

        return _finish(
            sample_id, status, reference_modality, reference_native_um,
            matrices, diagnostics, warnings, t0,
            None if status == "ok" else "all registration steps failed",
            confidence=overall_confidence,
        )

    except Exception as exc:  # noqa: BLE001 - run_batch must continue after a per-sample failure
        warnings.append(f"unhandled exception: {exc}")
        return _finish(
            sample_id, "failed", None, None, matrices, diagnostics, warnings, t0,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _overall_confidence(diagnostics: List[Dict], matrices: Dict) -> str:
    """The lowest confidence among steps that actually produced a matrix;
    "low" if nothing succeeded."""
    levels = []
    for diag in diagnostics:
        modality = diag["step"].split("_to_")[0]
        if matrices.get(modality) is not None:
            levels.append(diag["confidence"])
    if not levels:
        return "low"
    return min(levels, key=lambda c: CONFIDENCE_ORDER.get(c, 0))


def _finish(
    sample_id, status, reference_modality, reference_pixel_size_um,
    matrices, diagnostics, warnings, t0, error, confidence: Optional[str] = None,
) -> Dict:
    return {
        "sample_id": sample_id,
        "status": status,
        "reference_modality": reference_modality,
        "reference_pixel_size_um": reference_pixel_size_um,
        "matrices": matrices,
        "diagnostics": diagnostics,
        "confidence": confidence if confidence is not None else ("low" if status != "ok" else "low"),
        "warnings": warnings,
        "runtime_s": time.time() - t0,
        "error": error,
    }


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

def run_batch(
    samples: List[Dict],
    reference: str = "auto",
    progress: Optional[Callable[[float, str], None]] = None,
    cancel=None,
) -> "tuple[List[Dict], pd.DataFrame]":
    """Run run_sample sequentially over samples, continuing after a
    per-sample failure (run_sample itself catches exceptions, but this adds
    a second layer of defense so one malformed sample dict cannot abort the
    batch). Returns (results, table): results is the list of run_sample
    dicts; table is a one-row-per-sample pandas DataFrame with sample_id,
    status, reference_modality, confidence, n_warnings, runtime_s, error.
    """
    results: List[Dict] = []
    n = max(len(samples), 1)
    for i, sample in enumerate(samples):
        if cancel is not None and cancel.is_set():
            results.append(
                _finish(sample.get("sample_id", "<unknown>"), "cancelled", None, None, {}, [], [], time.time(), None)
            )
            continue
        try:
            result = run_sample(sample, reference=reference, progress=progress, cancel=cancel)
        except Exception as exc:  # noqa: BLE001 - must continue to the next sample regardless
            result = _finish(
                sample.get("sample_id", "<unknown>"), "failed", None, None, {}, [],
                [f"unhandled exception in run_batch: {exc}"], time.time(),
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        results.append(result)
        if progress is not None:
            progress((i + 1) / n, f"completed {sample.get('sample_id', '<unknown>')}")

    rows = []
    for res in results:
        rows.append(
            {
                "sample_id": res["sample_id"],
                "status": res["status"],
                "reference_modality": res["reference_modality"],
                "confidence": res["confidence"],
                "n_warnings": len(res["warnings"]),
                "runtime_s": res["runtime_s"],
                "error": res["error"] or "",
            }
        )
    return results, pd.DataFrame(rows)
