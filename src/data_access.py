"""Working-resolution data access for the UnumLocalia OME-NGFF (Zarr) images.

All functions read pixel size and channel information from the OME-NGFF
metadata embedded in each core's .ome.zarr store. Nothing is read from a
hardcoded constant table.

Store layout (confirmed with zarr.open on core01_hcc):

    <core>_he.ome.zarr/         root group, attrs = {"bioformats2raw.layout": 3}
      0/                        OME series group (only one series present)
        .zattrs                 {"multiscales": [...], "omero": {...}}
        0/ .. 5/                pyramid levels, arrays shaped (T, C, Z, Y, X)

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from skimage.transform import rescale, resize


# ---------------------------------------------------------------------------
# Metadata access
# ---------------------------------------------------------------------------

def _open_series_group(zarr_path: Path):
    """Open the single OME series group inside a bioformats2raw Zarr store.

    Raises ValueError (does not guess) if there is not exactly one numeric
    series group, or if it carries no multiscales metadata.
    """
    root = zarr.open(str(zarr_path), mode="r")
    series_keys = sorted(k for k in root.group_keys() if k.isdigit())
    if not series_keys:
        raise ValueError(f"No numeric OME series group found under {zarr_path}")
    if len(series_keys) > 1:
        raise ValueError(
            f"Expected exactly one OME series group under {zarr_path}, "
            f"found {series_keys}. Not guessing which one to use."
        )
    series = root[series_keys[0]]
    if "multiscales" not in series.attrs:
        raise ValueError(f"Series group {zarr_path}/{series_keys[0]} has no multiscales metadata.")
    return series


def get_multiscale_metadata(zarr_path: Path) -> Dict:
    """Return the first multiscales entry (axes and per-level scale transforms)."""
    series = _open_series_group(zarr_path)
    multiscales = series.attrs["multiscales"]
    return multiscales[0]


def get_omero_metadata(zarr_path: Path) -> Optional[Dict]:
    """Return the omero metadata block (channel labels, colors, windows), if present."""
    series = _open_series_group(zarr_path)
    return series.attrs.get("omero")


def list_pyramid_levels(zarr_path: Path) -> List[Dict]:
    """List every pyramid level with its path, axes-ordered scale, and um/px.

    Used to verify the pyramid scale factors per level (task 4, item 1).
    Returns one dict per level: {"path", "scale", "x_um", "y_um", "unit"}.
    """
    multiscale = get_multiscale_metadata(zarr_path)
    axes = multiscale["axes"]
    x_idx = next(i for i, a in enumerate(axes) if a["name"] == "x")
    y_idx = next(i for i, a in enumerate(axes) if a["name"] == "y")
    x_unit = axes[x_idx].get("unit")
    y_unit = axes[y_idx].get("unit")

    levels = []
    for dataset in multiscale["datasets"]:
        scale = dataset["coordinateTransformations"][0]["scale"]
        levels.append(
            {
                "path": dataset["path"],
                "scale": scale,
                "x_um": scale[x_idx] if x_unit == "micrometer" else None,
                "y_um": scale[y_idx] if y_unit == "micrometer" else None,
                "x_unit": x_unit,
                "y_unit": y_unit,
            }
        )
    return levels


def get_pixel_size_um(zarr_path: Path, level: int = 0) -> float:
    """Read the x/y pixel size in micrometers at a given pyramid level.

    Reads the OME-NGFF "coordinateTransformations" scale for the requested
    level's x and y axes, not a hardcoded constant. Raises ValueError if the
    axis unit is not micrometers, or if x and y pixel size disagree (so
    nothing is silently averaged or guessed).
    """
    levels = list_pyramid_levels(zarr_path)
    matching = [lvl for lvl in levels if lvl["path"] == str(level)]
    if not matching:
        available = [lvl["path"] for lvl in levels]
        raise ValueError(f"Level {level} not found under {zarr_path}. Available levels: {available}")
    lvl = matching[0]
    if lvl["x_unit"] != "micrometer" or lvl["y_unit"] != "micrometer":
        raise ValueError(
            f"Expected micrometer axis units at {zarr_path} level {level}, "
            f"got x_unit={lvl['x_unit']} y_unit={lvl['y_unit']}."
        )
    if lvl["x_um"] is None or lvl["y_um"] is None:
        raise ValueError(f"Missing x/y scale at {zarr_path} level {level}.")
    if not np.isclose(lvl["x_um"], lvl["y_um"], rtol=1e-6):
        raise ValueError(
            f"x and y pixel size disagree at {zarr_path} level {level}: "
            f"x={lvl['x_um']} y={lvl['y_um']}. Not averaging or guessing."
        )
    return float(lvl["x_um"])


def get_pyramid_scale_factors(zarr_path: Path) -> List[Dict]:
    """Return, for every level, the pixel size and its ratio to level 0.

    This is the "verify the pyramid scale factors per level" check from
    task 4, item 1: each level's um/px should be level0_um_px * 2**level
    for a standard bioformats2raw power-of-two pyramid.
    """
    levels = list_pyramid_levels(zarr_path)
    base_x = levels[0]["x_um"]
    out = []
    for lvl in levels:
        out.append(
            {
                "level": int(lvl["path"]),
                "x_um": lvl["x_um"],
                "y_um": lvl["y_um"],
                "scale_vs_level0": lvl["x_um"] / base_x if base_x else None,
            }
        )
    return out


def get_channel_index(zarr_path: Path, label: str) -> int:
    """Find a channel index by exact label match in the omero metadata.

    This is how the DAPI channel index is determined: by reading the
    "omero.channels[*].label" field from the zarr metadata, not by
    assuming a fixed position. Raises ValueError (no guessing) if the
    label is not found, and lists the labels that were found instead.
    """
    omero = get_omero_metadata(zarr_path)
    if omero is None or "channels" not in omero:
        raise ValueError(f"No omero channel metadata found at {zarr_path}.")
    labels = [ch.get("label") for ch in omero["channels"]]
    for i, lbl in enumerate(labels):
        if lbl == label:
            return i
    raise ValueError(f"Channel label {label!r} not found at {zarr_path}. Available labels: {labels}")


# ---------------------------------------------------------------------------
# Lazy array access
# ---------------------------------------------------------------------------

def open_level(zarr_path: Path, target_um_per_px: float) -> Tuple[zarr.Array, float, int]:
    """Open the pyramid level closest to target_um_per_px.

    Returns (array, effective_um_per_px, level_index). The array is a lazy
    zarr.core.Array (dask-free, chunked on disk); no data is read into
    memory until it is sliced or converted with np.asarray.
    """
    levels = list_pyramid_levels(zarr_path)
    diffs = [abs(lvl["x_um"] - target_um_per_px) for lvl in levels]
    best = int(np.argmin(diffs))
    best_level = levels[best]

    series = _open_series_group(zarr_path)
    array = series[best_level["path"]]
    return array, float(best_level["x_um"]), int(best_level["path"])


# ---------------------------------------------------------------------------
# Modality-specific loaders
# ---------------------------------------------------------------------------

def load_he_rgb(zarr_path: Path, working_um_per_px: float = 2.0) -> Tuple[np.ndarray, float, int]:
    """Load the H&E image as an (H, W, 3) uint8 array at a working resolution.

    Returns (rgb_array, effective_um_per_px, level_index). The array has
    axes (T=1, C=3, Z=1, Y, X) on disk; channels are moved to the last axis
    to form a standard RGB image.
    """
    array, eff_um_per_px, level = open_level(zarr_path, working_um_per_px)
    # array shape: (T, C, Z, Y, X); H&E has 3 channels (RGB), T=1, Z=1.
    chw = np.asarray(array[0, :, 0, :, :])
    rgb = np.moveaxis(chw, 0, -1)
    return rgb, eff_um_per_px, level


def load_comet_dapi(
    zarr_path: Path, working_um_per_px: float = 2.0, dapi_label: str = "DAPI"
) -> Tuple[np.ndarray, float, int, int]:
    """Load the COMET DAPI channel as a (H, W) array at a working resolution.

    The DAPI channel index is found from the zarr store's own
    "omero.channels[*].label" metadata (see get_channel_index), not assumed
    to be channel 0. Returns (dapi_array, effective_um_per_px, level_index,
    dapi_channel_index).
    """
    dapi_index = get_channel_index(zarr_path, dapi_label)
    array, eff_um_per_px, level = open_level(zarr_path, working_um_per_px)
    dapi = np.asarray(array[0, dapi_index, 0, :, :])
    return dapi, eff_um_per_px, level, dapi_index


# ---------------------------------------------------------------------------
# Explicit resampling to a common pixel size
# ---------------------------------------------------------------------------

def resample_to_um(
    img: np.ndarray, current_um_per_px: float, target_um_per_px: float, order: int = 1
) -> Tuple[np.ndarray, float]:
    """Resample img from current_um_per_px to (as close as possible to) target_um_per_px.

    Pyramid levels only give a handful of discrete resolutions (doubling
    each level), so two modalities asked for the same nominal working
    resolution can still land on different actual um/px (see
    outputs/data_notes.md). This function explicitly resamples an already
    loaded array to a target pixel size with anti-aliasing
    (skimage.transform.rescale, which applies a Gaussian pre-filter before
    downsampling to avoid aliasing), so H&E and COMET can be brought to
    exactly the same working pixel size when that is required (e.g. for
    density-map comparison in automatic registration methods).

    img may be 2D (single channel, e.g. DAPI) or 3D with channels last
    (e.g. RGB H&E); rescale is applied per-channel in the 3D case via
    channel_axis=-1. order=1 (bilinear) by default; use order=0 for a
    label/mask image where interpolation must not invent new values.

    Because the output shape is rounded to the nearest integer pixel, the
    achieved um/px can differ very slightly from target_um_per_px; the
    actual achieved value is returned alongside the resampled array as
    (resampled_img, achieved_um_per_px), rather than silently assumed to be
    exactly the target.
    """
    scale_factor = current_um_per_px / target_um_per_px
    input_dtype = img.dtype

    if img.ndim == 2:
        out = rescale(
            img, scale_factor, order=order, anti_aliasing=(scale_factor < 1.0), preserve_range=True
        )
    elif img.ndim == 3:
        out = rescale(
            img,
            scale_factor,
            order=order,
            anti_aliasing=(scale_factor < 1.0),
            preserve_range=True,
            channel_axis=-1,
        )
    else:
        raise ValueError(f"resample_to_um expects a 2D or 3D array, got shape {img.shape}")

    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        out = np.clip(np.round(out), info.min, info.max).astype(input_dtype)
    else:
        out = out.astype(input_dtype)

    achieved_scale = out.shape[0] / img.shape[0]
    achieved_um_per_px = float(current_um_per_px / achieved_scale)
    return out, achieved_um_per_px


def _best_crop_and_size(n_in: int, scale: float, max_crop_px: int) -> Tuple[int, int]:
    """Pick how many trailing input pixels to drop (0..max_crop_px) so that
    round(n_kept * scale) is as close as possible to n_kept * scale, i.e.
    the output pixel size is as close as possible to the requested one.
    Returns (n_kept, n_out); ties prefer dropping fewer pixels."""
    best = None
    for crop in range(0, max_crop_px + 1):
        n_kept = n_in - crop
        if n_kept < 2:
            break
        exact = n_kept * scale
        n_out = max(1, int(round(exact)))
        err = abs(n_out / exact - 1.0)
        if best is None or err < best[0] - 1e-15:
            best = (err, n_kept, n_out)
    return best[1], best[2]


def resample_to_um_exact(
    img: np.ndarray, current_um_per_px: float, target_um_per_px: float, order: int = 1, max_crop_px: int = 8
) -> Tuple[np.ndarray, float]:
    """Task 12, item 2: resample img ONCE, and report the effective pixel
    size exactly.

    Same filtering as resample_to_um (skimage resize, anti-aliased when
    downsampling, order=1 by default). The difference is that up to
    max_crop_px trailing rows and columns of the INPUT are dropped, chosen
    per axis, so that the integer output shape reproduces target_um_per_px
    almost exactly (typically within 0.01 percent) instead of up to about
    0.1 percent off from shape rounding. The top-left pixel (0, 0) is
    never moved, so pixel coordinates keep their origin.

    Returns (resampled_img, effective_um_per_px). The effective size is
    computed from the shapes actually used per axis (current_um * kept /
    out), and the geometric mean of the x and y values is returned. If
    current_um_per_px equals target_um_per_px exactly, img is returned
    unchanged (no resampling at all) with that pixel size.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"resample_to_um_exact expects a 2D or 3D array, got shape {img.shape}")
    if current_um_per_px == target_um_per_px:
        return img, float(current_um_per_px)

    scale = current_um_per_px / target_um_per_px
    h_kept, h_out = _best_crop_and_size(img.shape[0], scale, max_crop_px)
    w_kept, w_out = _best_crop_and_size(img.shape[1], scale, max_crop_px)
    cropped = img[:h_kept, :w_kept]

    out_shape = (h_out, w_out) if img.ndim == 2 else (h_out, w_out, img.shape[2])
    out = resize(
        cropped, out_shape, order=order, anti_aliasing=(scale < 1.0), preserve_range=True
    )
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.round(out), info.min, info.max).astype(img.dtype)
    else:
        out = out.astype(img.dtype)

    um_y = current_um_per_px * h_kept / h_out
    um_x = current_um_per_px * w_kept / w_out
    return out, float(np.sqrt(um_x * um_y))
