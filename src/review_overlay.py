"""In-memory review overlays for the GUI (task 14). No UI imports, nothing is
written to disk.

build_review_images renders, at a display resolution (default 2 um/px), the
reference layer and one registered layer warped into the reference layer's
display grid with the matrix from pipeline.run_sample, plus the 40-point
keypoint positions in that grid. The compose_* functions turn the two
images into one RGB uint8 array for the viewer modes.

The display grid is for looking at the result only: it is a resampled
2 um/px view, not the registration grid, and no registered image is saved
anywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import data_access as da
import io_layers
import landmarks
import pipeline
import render

DISPLAY_UM_PER_PX = 2.0
N_MARKERS = 40
CHECKER_TILE_PX = 48
SIDE_BY_SIDE_GAP_PX = 8


def _scale_matrix(factor: float, offset: float = 0.0) -> np.ndarray:
    m = np.eye(3)
    m[0, 0] = m[1, 1] = factor
    m[0, 2] = m[1, 2] = offset
    return m


def _native_to_display(native_um: float, display_um: float) -> np.ndarray:
    """Native pixel (x, y) -> display pixel, with pixel centers aligned."""
    f = native_um / display_um
    return _scale_matrix(f, 0.5 * f - 0.5)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch an image to uint8 using the 1st to 99.5th
    percentile of its nonzero pixels (the zero background outside the
    tissue or the warped area is left out of the statistics and stays
    black). Falls back to all pixels if everything is zero."""
    a = np.asarray(img, dtype=np.float64)
    vals = a[a > 0]
    if vals.size == 0:
        vals = a.ravel()
    lo, hi = np.percentile(vals, [1.0, 99.5])
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)

def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


def registered_modalities(result: Dict) -> List[str]:
    """Layers that have a matrix onto the reference (not the reference itself)."""
    ref = result.get("reference_modality")
    return [m for m, mat in result.get("matrices", {}).items() if mat is not None and m != ref]


def _open_rgb(desc: Dict, display_um: float):
    """True-color uint8 HxWx3 of an RGB brightfield layer on the same display
    grid as io_layers.open_layer (same level choice and resampling), or None
    when the format is not supported here (the view then stays grayscale)."""
    if desc.get("kind") != "rgb_brightfield":
        return None
    fmt, path = desc.get("format"), Path(desc["path"])
    if fmt == "ome_zarr":
        arr, eff_um, _level = da.open_level(path, display_um)
        rgb = np.moveaxis(np.asarray(arr[0, :, 0, :, :]), 0, -1)
    elif fmt in ("ome_tiff", "tiff"):
        import tifffile
        levels = [lvl for lvl in desc["shape_per_level"] if lvl["um_per_px"] is not None]
        if not levels:
            return None
        best = levels[int(np.argmin([abs(lvl["um_per_px"] - display_um) for lvl in levels]))]
        with tifffile.TiffFile(str(path)) as tif:
            rgb = tif.series[0].levels[best["level"]].asarray()
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
            rgb = np.moveaxis(rgb[:3], 0, -1)
        else:
            rgb = rgb[..., :3]
        eff_um = best["um_per_px"]
    else:
        return None
    rgb, _ach = da.resample_to_um_exact(rgb, eff_um, display_um)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def build_review_images(
    layers: Dict, result: Dict, modality: str, display_um: float = DISPLAY_UM_PER_PX
) -> Dict:
    """Returns {"reference": HxW uint8, "moved": HxW uint8 (the layer warped
    into the reference grid), "markers": (n, 2) float array of display-grid
    (x, y) positions (the matrix applied to the thinned inlier points, as in
    the exported keypoints CSV), "display_um": float, "n_markers": int}."""
    matrix = result["matrices"].get(modality)
    if matrix is None:
        raise ValueError(f"{modality}: no matrix, nothing to show")
    matrix = np.asarray(matrix, dtype=float)
    ref_modality = result["reference_modality"]
    ref_layer = layers[ref_modality]
    ref_desc = pipeline.describe_layer(ref_layer)
    lay_desc = pipeline.describe_layer(layers[modality])

    if ref_desc["kind"] == "xenium" and ref_desc.get("xenium_source") == "rendered":
        ref_img, s_mat = render.render_nuclei_in_folder(Path(ref_layer["path"]), px_um=display_um)
        # reference px = Xenium um / 0.2125; s_mat maps render px -> Xenium um
        d_mat = np.diag([pipeline.REFERENCE_UM_PER_PX, pipeline.REFERENCE_UM_PER_PX, 1.0])
        ref_to_disp = np.linalg.inv(s_mat) @ d_mat
    else:
        ref_img, ref_ach = io_layers.open_layer(ref_desc, display_um, channel=ref_layer.get("channel"))
        ref_to_disp = _native_to_display(ref_desc["pixel_size_um"], ref_ach)

    if lay_desc["kind"] == "xenium" and lay_desc.get("xenium_source") == "rendered":
        # Xenium as a moved layer (another modality is the reference): its frame is Xenium um / 0.2125
        mov_img, mov_s = render.render_nuclei_in_folder(Path(layers[modality]["path"]), px_um=display_um)
        d_mat = np.diag([pipeline.REFERENCE_UM_PER_PX, pipeline.REFERENCE_UM_PER_PX, 1.0])
        mov_disp_to_native = np.linalg.inv(d_mat) @ mov_s
    else:
        mov_img, mov_ach = io_layers.open_layer(lay_desc, display_um, channel=layers[modality].get("channel"))
        mov_disp_to_native = np.linalg.inv(_native_to_display(lay_desc["pixel_size_um"], mov_ach))
    mov_to_ref_disp = ref_to_disp @ matrix @ mov_disp_to_native

    h, w = ref_img.shape[:2]
    warp_flags = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped = cv2.warpAffine(
        np.asarray(mov_img, dtype=np.float32), mov_to_ref_disp[:2], (w, h), borderValue=0.0, **warp_flags)

    # True-color copies, only for side by side (None if not an RGB layer).
    ref_rgb = _open_rgb(ref_desc, display_um) if ref_desc["kind"] != "xenium" else None
    mov_rgb = _open_rgb(lay_desc, display_um)
    if ref_rgb is not None and ref_rgb.shape[:2] != ref_img.shape[:2]:
        ref_rgb = None
    if mov_rgb is not None:
        mov_rgb = cv2.warpAffine(mov_rgb, mov_to_ref_disp[:2], (w, h), borderValue=(0, 0, 0), **warp_flags)

    markers = np.zeros((0, 2))
    n_matched = 0
    step = next((d for d in result.get("diagnostics", []) if d["step"].startswith(f"{modality}_to_")), None)
    if step is not None and step.get("src_pts") is not None and len(step["src_pts"]) > 0:
        n_matched = len(step["src_pts"])
        src = np.asarray(step["src_pts"], dtype=float)
        dst = np.asarray(step["dst_pts"], dtype=float)
        src_sel, _dst_sel, _idx = landmarks.suggest_landmarks(src, dst, n=N_MARKERS)
        markers = _apply(ref_to_disp, _apply(matrix, np.asarray(src_sel, dtype=float)))

    return {
        "reference": to_uint8(ref_img), "moved": to_uint8(warped), "markers": markers,
        "reference_rgb": ref_rgb, "moved_rgb": mov_rgb,
        "display_um": float(display_um), "n_markers": int(len(markers)), "n_matched": int(n_matched),
    }


def _gray_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def _magenta(reference: np.ndarray, weight: float = 1.0) -> np.ndarray:
    v = np.clip(reference.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return np.stack([v, np.zeros_like(v), v], axis=2)


def _green(moved: np.ndarray, weight: float = 1.0) -> np.ndarray:
    v = np.clip(moved.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return np.stack([np.zeros_like(v), v, np.zeros_like(v)], axis=2)


def compose_blend(reference: np.ndarray, moved: np.ndarray, alpha: float, color: bool = False) -> np.ndarray:
    """alpha 0 shows only the reference, alpha 1 only the registered layer.
    color=True: reference in magenta and registered layer in green, added,
    so where they agree the result is white-gray; at alpha 0.5 both layers
    have full weight."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if color:
        out = _magenta(reference, min(1.0, 2.0 * (1.0 - a))).astype(np.int16) + _green(moved, min(1.0, 2.0 * a))
        return np.clip(out, 0, 255).astype(np.uint8)
    out = (1.0 - a) * reference.astype(np.float32) + a * moved.astype(np.float32)
    return _gray_rgb(np.clip(out, 0, 255).astype(np.uint8))


def compose_checkerboard(reference: np.ndarray, moved: np.ndarray, tile_px: int = CHECKER_TILE_PX,
                         color: bool = False) -> np.ndarray:
    h, w = reference.shape[:2]
    yy, xx = np.indices((h, w))
    use_moved = ((yy // tile_px) + (xx // tile_px)) % 2 == 1
    if color:
        return np.where(use_moved[:, :, None], _green(moved), _magenta(reference)).astype(np.uint8)
    return _gray_rgb(np.where(use_moved, moved, reference).astype(np.uint8))


def compose_side_by_side(reference: np.ndarray, moved: np.ndarray, gap_px: int = SIDE_BY_SIDE_GAP_PX,
                         reference_rgb=None, moved_rgb=None) -> np.ndarray:
    """Reference on the left, registered layer on the right, in one canvas
    (so pan and zoom are shared). Grayscale, except that a layer with a
    true-color version (H&E) is shown in true color."""
    h, w = reference.shape[:2]
    canvas = np.full((h, 2 * w + gap_px, 3), 40, dtype=np.uint8)
    canvas[:, :w] = _gray_rgb(reference) if reference_rgb is None else reference_rgb
    canvas[:, w + gap_px:] = _gray_rgb(moved) if moved_rgb is None else moved_rgb
    return canvas


def compose_side_panes(reference: np.ndarray, moved: np.ndarray, reference_rgb=None, moved_rgb=None):
    """The two side-by-side panes as separate arrays (left: reference, right:
    registered layer), same grid and size, so two linked views can show them.
    Grayscale, except a layer with a true-color version (H&E)."""
    left = _gray_rgb(reference) if reference_rgb is None else reference_rgb
    right = _gray_rgb(moved) if moved_rgb is None else moved_rgb
    return left, right


def compose_flicker_frame(reference: np.ndarray, moved: np.ndarray, show_moved: bool, color: bool = False) -> np.ndarray:
    if color:
        return _green(moved) if show_moved else _magenta(reference)
    return _gray_rgb(moved if show_moved else reference)
