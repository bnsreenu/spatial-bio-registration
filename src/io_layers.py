"""Task 11, item 1: a format-detecting input layer.

describe_input(path) inspects a path and returns a plain descriptor dict
(no classes), identifying:

  - "rgb_brightfield": a 3-channel, 8-bit image such as H&E, from an
    OME-Zarr store (bioformats2raw layout, the UnumLocalia convention),
    an OME-TIFF, a plain (non-OME) TIFF, or an Aperio SVS / Hamamatsu NDPI
    whole-slide file (via OpenSlide).
  - "multichannel_fluorescence": a multi-channel image such as COMET, from
    an OME-Zarr store or an OME-TIFF, with channel names read from the
    file's own metadata (omero channel labels, or OME-XML Channel Name).
  - "xenium": a Xenium output folder (cells.csv, nucleus_boundaries.parquet,
    cell_boundaries.parquet), used either via a real morphology image if
    one is present, or by rendering nucleus_boundaries.parquet with
    render.render_nuclei_in_folder (never transcripts.parquet, which is
    never read anywhere in this module).

open_layer(descriptor, target_um_per_px, channel=None) returns the image
at (as close as possible to) target_um_per_px, reusing data_access.
open_level / resample_to_um_exact (one resample, effective pixel size
returned exactly; task 12) for the resampling, and prep.he_to_hematoxylin
/ prep.normalize_dapi for the same stain/contrast normalization already
used throughout this project, so its output is directly usable by
engine.register_pair.

Pixel size is never silently assumed: describe_input only reports a
pixel_size_um it actually read from metadata, flags it with a warning if
it looks implausible (outside 0.05 to 5 um/px, a generous range spanning
whole-slide brightfield down to very high resolution fluorescence), and
open_layer refuses to proceed with a missing or flagged pixel size unless
the caller supplies native_um_per_px_override explicitly.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
No UI imports are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import data_access as da
import prep
import render

PLAUSIBLE_UM_PER_PX = (0.05, 5.0)

XENIUM_MARKER_FILES = (
    "cells.csv",
    "nucleus_boundaries.parquet",
    "cell_boundaries.parquet",
    "cell_boundaries_comet_space.geojson",
)

SUPPORTED_FORMATS = [
    "OME-Zarr (a directory ending in .ome.zarr, bioformats2raw layout)",
    "OME-TIFF (.ome.tif / .ome.tiff, read with tifffile)",
    "plain TIFF (.tif / .tiff, read with tifffile; pixel size only if "
    "resolution tags are present)",
    "Aperio SVS (.svs) or Hamamatsu NDPI (.ndpi), read with OpenSlide if installed",
    "a Xenium output folder (containing cells.csv, nucleus_boundaries.parquet, "
    "or cell_boundaries.parquet; optionally a morphology image)",
]


# ---------------------------------------------------------------------------
# describe_input: format and kind detection
# ---------------------------------------------------------------------------

def _plausible(pixel_size_um: Optional[float]) -> bool:
    if pixel_size_um is None:
        return False
    lo, hi = PLAUSIBLE_UM_PER_PX
    return lo <= pixel_size_um <= hi


def _is_ome_zarr_dir(path: Path) -> bool:
    return path.is_dir() and (path.name.lower().endswith(".ome.zarr") or (path / ".zgroup").exists())


def _is_xenium_folder(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in XENIUM_MARKER_FILES)


def _find_morphology_file(xenium_folder: Path) -> Optional[Path]:
    """Look for a file or folder whose name starts with "morphology"
    (case-insensitive) directly inside the Xenium folder, including a
    morphology_focus folder of per-z-plane images. Returns a single file
    path (the first match, sorted, if several are found) or None. If more
    than one candidate is found, which one was picked is still reported
    (no attempt to silently merge z-planes), since no such file exists in
    data/UnumLocalia to validate the right behavior against (see
    outputs/task11_notes.md).
    """
    candidates: List[Path] = []
    for entry in sorted(xenium_folder.iterdir()):
        if entry.name.lower().startswith("morphology"):
            if entry.is_file():
                candidates.append(entry)
            elif entry.is_dir():
                candidates.extend(sorted(p for p in entry.iterdir() if p.is_file()))
    return candidates[0] if candidates else None


def _describe_xenium(path: Path) -> Dict:
    morphology_path = _find_morphology_file(path)
    warnings: List[str] = []
    if morphology_path is not None:
        inner = _describe_tiff_like(morphology_path)
        inner["path"] = str(path)
        inner["xenium_source"] = "morphology"
        inner["morphology_path"] = str(morphology_path)
        inner["kind"] = "xenium"
        inner["format"] = "xenium_folder_morphology"
        inner["warnings"] = inner.get("warnings", []) + [
            f"using morphology image {morphology_path.name} found in this Xenium folder "
            "(this has not been tested on a real Xenium morphology image)."
        ]
        return inner

    warnings.append(
        "no morphology image found in this Xenium folder; nuclei will be drawn "
        "from nucleus_boundaries.parquet instead of "
        "a real morphology/DAPI image. transcripts.parquet is never read."
    )
    return {
        "path": str(path),
        "kind": "xenium",
        "format": "xenium_folder_rendered",
        "xenium_source": "rendered",
        "pixel_size_um": None,
        "channel_names": None,
        "shape_per_level": None,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# OME-Zarr
# ---------------------------------------------------------------------------

def _describe_zarr(path: Path) -> Dict:
    warnings: List[str] = []
    levels = da.list_pyramid_levels(path)
    series = da._open_series_group(path)
    array0 = series[levels[0]["path"]]
    shape0 = array0.shape  # (T, C, Z, Y, X)
    n_channels = shape0[1]
    dtype = array0.dtype

    try:
        pixel_size_um = da.get_pixel_size_um(path, level=0)
    except ValueError as exc:
        pixel_size_um = None
        warnings.append(f"could not read a pixel size: {exc}")

    if pixel_size_um is not None and not _plausible(pixel_size_um):
        warnings.append(
            f"pixel size {pixel_size_um:g} um/px is outside the plausible range "
            f"{PLAUSIBLE_UM_PER_PX}; an override is required before this input can be used."
        )

    omero = da.get_omero_metadata(path)
    channel_names = None
    if omero is not None and "channels" in omero:
        channel_names = [ch.get("label") for ch in omero["channels"]]
        colors = [ch.get("color") for ch in omero["channels"]]
    else:
        colors = None
        warnings.append("no omero channel metadata found; channel names are unknown.")

    # Kind heuristic, documented plainly (not a silent guess): 3 channels
    # with 8-bit depth and colors that are exactly red/green/blue (in that
    # order) is treated as an RGB brightfield image (H&E in this dataset);
    # anything else with more than one channel is treated as multichannel
    # fluorescence. A single-channel store would not match either rule and
    # is reported as "unknown" rather than guessed.
    if n_channels == 3 and dtype == np.uint8 and colors == ["FF0000", "00FF00", "0000FF"]:
        kind = "rgb_brightfield"
        channel_names = ["R", "G", "B"]
    elif n_channels == 3 and dtype == np.uint8:
        kind = "rgb_brightfield"
        warnings.append(
            f"3 channels, 8-bit, but channel colors {colors!r} are not the expected "
            "[FF0000, 00FF00, 0000FF] red/green/blue order; treating as RGB brightfield "
            "anyway based on channel count and bit depth, but this is a heuristic, not a "
            "confirmed RGB image."
        )
    elif n_channels >= 1:
        kind = "multichannel_fluorescence"
    else:
        kind = "unknown"
        warnings.append(f"could not classify input kind from {n_channels} channels, dtype {dtype}.")

    shape_per_level = [
        {
            "level": int(lvl["path"]),
            "shape": tuple(int(s) for s in series[lvl["path"]].shape),
            "um_per_px": lvl["x_um"],
        }
        for lvl in levels
    ]

    return {
        "path": str(path),
        "kind": kind,
        "format": "ome_zarr",
        "pixel_size_um": pixel_size_um,
        "channel_names": channel_names,
        "shape_per_level": shape_per_level,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# OME-TIFF / plain TIFF
# ---------------------------------------------------------------------------

def _unit_to_um_factor(unit: Optional[str]) -> Optional[float]:
    if unit is None:
        return None
    unit = unit.strip().lower()
    table = {
        "\u00b5m": 1.0, "um": 1.0, "micrometer": 1.0, "micron": 1.0,
        "nm": 1e-3, "nanometer": 1e-3,
        "mm": 1e3, "millimeter": 1e3,
        "cm": 1e4, "centimeter": 1e4,
    }
    return table.get(unit)


def _describe_tiff_like(path: Path) -> Dict:
    """describe_input logic for a .tif/.tiff/.ome.tif/.ome.tiff file,
    including a Xenium morphology file (also a TIFF in practice)."""
    import tifffile

    warnings: List[str] = []
    pixel_size_um = None
    channel_names = None
    kind = "unknown"

    with tifffile.TiffFile(str(path)) as tif:
        base_shape = tif.series[0].shape
        base_dtype = tif.series[0].dtype
        axes = tif.series[0].axes

        shape_per_level = []
        for i, level in enumerate(tif.series[0].levels):
            shape_per_level.append({"level": i, "shape": tuple(int(s) for s in level.shape), "um_per_px": None})

        if tif.is_ome and tif.ome_metadata:
            try:
                ome = tifffile.xml2dict(tif.ome_metadata)
                image = ome["OME"]["Image"]
                if isinstance(image, list):
                    image = image[0]
                pixels = image["Pixels"]
                size_x = pixels.get("PhysicalSizeX")
                size_y = pixels.get("PhysicalSizeY")
                unit_x = pixels.get("PhysicalSizeXUnit", "um")
                unit_y = pixels.get("PhysicalSizeYUnit", "um")
                factor_x = _unit_to_um_factor(unit_x)
                factor_y = _unit_to_um_factor(unit_y)
                if size_x is not None and size_y is not None and factor_x and factor_y:
                    px_x_um = float(size_x) * factor_x
                    px_y_um = float(size_y) * factor_y
                    if np.isclose(px_x_um, px_y_um, rtol=1e-3):
                        pixel_size_um = px_x_um
                    else:
                        warnings.append(
                            f"x pixel size ({px_x_um:g} um) and y pixel size ({px_y_um:g} um) "
                            "disagree; not averaging or guessing."
                        )
                else:
                    warnings.append("OME-XML Pixels element has no usable PhysicalSizeX/Y.")

                samples_per_pixel = pixels.get("SizeC")
                channel_entries = pixels.get("Channel")
                if channel_entries is not None:
                    if isinstance(channel_entries, dict):
                        channel_entries = [channel_entries]
                    channel_names = [c.get("Name") for c in channel_entries]

                if samples_per_pixel is not None and int(samples_per_pixel) == 3 and base_dtype == np.uint8:
                    kind = "rgb_brightfield"
                    channel_names = channel_names or ["R", "G", "B"]
                elif samples_per_pixel is not None and int(samples_per_pixel) >= 1:
                    kind = "multichannel_fluorescence"
            except Exception as exc:  # noqa: BLE001 - metadata parsing is best-effort
                warnings.append(f"failed to parse OME-XML metadata: {exc}")
        else:
            page = tif.pages[0]
            photometric = page.photometric
            if str(photometric).upper().endswith("RGB") or (page.samplesperpixel == 3 and base_dtype == np.uint8):
                kind = "rgb_brightfield"
                channel_names = ["R", "G", "B"]
            else:
                kind = "multichannel_fluorescence"
                channel_names = [f"Channel {i}" for i in range(page.samplesperpixel if "C" in axes else 1)]

            x_res = page.tags.get("XResolution")
            res_unit = page.tags.get("ResolutionUnit")
            if x_res is not None and res_unit is not None:
                num, den = x_res.value
                if num > 0:
                    px_per_unit = num / den
                    unit_to_um = {2: 25400.0, 3: 10000.0}.get(int(res_unit.value))  # inch, centimeter
                    if unit_to_um:
                        pixel_size_um = unit_to_um / px_per_unit
            if pixel_size_um is None:
                warnings.append(
                    "no OME-XML metadata and no usable TIFF resolution tags; pixel size is unknown."
                )

        for lvl in shape_per_level:
            if pixel_size_um is not None:
                base_x = base_shape[axes.index("X")]
                this_x = lvl["shape"][axes.index("X")] if len(lvl["shape"]) == len(axes) else lvl["shape"][-1]
                lvl["um_per_px"] = pixel_size_um * (base_x / this_x)

    if pixel_size_um is not None and not _plausible(pixel_size_um):
        warnings.append(
            f"pixel size {pixel_size_um:g} um/px is outside the plausible range "
            f"{PLAUSIBLE_UM_PER_PX}; an override is required before this input can be used."
        )

    is_ome = path.name.lower().endswith((".ome.tif", ".ome.tiff"))
    return {
        "path": str(path),
        "kind": kind,
        "format": "ome_tiff" if is_ome else "tiff",
        "pixel_size_um": pixel_size_um,
        "channel_names": channel_names,
        "shape_per_level": shape_per_level,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# SVS / NDPI (OpenSlide)
# ---------------------------------------------------------------------------

def _describe_openslide(path: Path) -> Dict:
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            f"OpenSlide (openslide-python, plus its native library) is required to read "
            f"{path.name!r} and is not available: {exc}. Install openslide-python and the "
            "OpenSlide native library (e.g. the openslide-bin package on Windows) to use "
            "SVS/NDPI inputs."
        ) from exc

    warnings: List[str] = [
        "SVS/NDPI support has not been tested on a real whole-slide file."
    ]

    slide = openslide.OpenSlide(str(path))
    try:
        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        pixel_size_um = None
        if mpp_x is not None and mpp_y is not None:
            mpp_x_f, mpp_y_f = float(mpp_x), float(mpp_y)
            if np.isclose(mpp_x_f, mpp_y_f, rtol=1e-3):
                pixel_size_um = mpp_x_f
            else:
                warnings.append(
                    f"openslide.mpp-x ({mpp_x_f:g}) and openslide.mpp-y ({mpp_y_f:g}) disagree; "
                    "not averaging or guessing."
                )
        else:
            warnings.append("no openslide.mpp-x/mpp-y property found; pixel size is unknown.")

        if pixel_size_um is not None and not _plausible(pixel_size_um):
            warnings.append(
                f"pixel size {pixel_size_um:g} um/px is outside the plausible range "
                f"{PLAUSIBLE_UM_PER_PX}; an override is required before this input can be used."
            )

        shape_per_level = []
        for i, (w, h) in enumerate(slide.level_dimensions):
            downsample = slide.level_downsamples[i]
            level_um = pixel_size_um * downsample if pixel_size_um is not None else None
            shape_per_level.append({"level": i, "shape": (h, w, 3), "um_per_px": level_um})
    finally:
        slide.close()

    return {
        "path": str(path),
        "kind": "rgb_brightfield",
        "format": path.suffix.lower().lstrip("."),
        "pixel_size_um": pixel_size_um,
        "channel_names": ["R", "G", "B"],
        "shape_per_level": shape_per_level,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# describe_input dispatcher
# ---------------------------------------------------------------------------

def describe_input(path: Union[str, Path]) -> Dict:
    """Detect the input type and return a descriptor dict:

      path, kind ("rgb_brightfield" / "multichannel_fluorescence" / "xenium"),
      format, pixel_size_um (float or None), channel_names (list or None),
      shape_per_level (list of {level, shape, um_per_px}, or None for a
      rendered Xenium input), warnings (list of str).

    Raises ValueError listing SUPPORTED_FORMATS if path does not match any
    known format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if _is_xenium_folder(path):
        return _describe_xenium(path)
    if _is_ome_zarr_dir(path):
        return _describe_zarr(path)
    if path.is_file() and path.suffix.lower() in (".svs", ".ndpi"):
        return _describe_openslide(path)
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
        return _describe_tiff_like(path)

    raise ValueError(
        f"Unsupported input {path}. Supported formats are:\n- " + "\n- ".join(SUPPORTED_FORMATS)
    )


# ---------------------------------------------------------------------------
# open_layer
# ---------------------------------------------------------------------------

def _resolve_native_um_per_px(descriptor: Dict, native_um_per_px_override: Optional[float]) -> Optional[float]:
    if descriptor["kind"] == "xenium" and descriptor.get("xenium_source") == "rendered":
        # No native pixel size exists for point/polygon data; it is
        # rendered directly at whatever resolution open_layer is asked
        # for, so no override is required for this case specifically.
        return None

    pixel_size_um = descriptor.get("pixel_size_um")
    plausible = _plausible(pixel_size_um)
    if native_um_per_px_override is not None:
        return float(native_um_per_px_override)
    if pixel_size_um is None or not plausible:
        raise ValueError(
            f"{descriptor['path']}: pixel size is "
            + ("missing" if pixel_size_um is None else f"implausible ({pixel_size_um:g} um/px)")
            + ". Not guessing; pass native_um_per_px_override= to open_layer explicitly."
        )
    return float(pixel_size_um)


def open_layer(
    descriptor: Dict,
    target_um_per_px: float,
    channel: Optional[Union[int, str]] = None,
    native_um_per_px_override: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return (image, achieved_um_per_px) at (as close as possible to)
    target_um_per_px, for the given descriptor (from describe_input).

      - "rgb_brightfield": returns prep.he_to_hematoxylin's normalized
        hematoxylin channel (percentile-normalized to 0..1), the same
        preprocessing used for H&E everywhere else in this project.
      - "multichannel_fluorescence": returns prep.normalize_dapi's
        percentile-normalized 0..1 image of the requested channel. channel
        may be an int index, or a str matched case-insensitively against
        channel_names; if channel is None, a channel named "dapi"
        (case-insensitive, substring match) is used if exactly one such
        channel exists. Raises ValueError listing the available channel
        names if no channel can be resolved this way.
      - "xenium": with a real morphology image, behaves like
        rgb_brightfield or multichannel_fluorescence depending on that
        image's own kind (see describe_input's xenium branch); otherwise
        renders nuclei directly at target_um_per_px with
        render.render_nuclei_in_folder (exact match, not nearest-pyramid-
        level-then-resize, since a rendered image has no pyramid).

    Pixel size is required to interpret native pyramid levels (zarr, tiff)
    and whole-slide files; see _resolve_native_um_per_px for when
    native_um_per_px_override is required.

    Task 12: the returned achieved_um_per_px is the exact effective pixel
    size of the returned array (computed from the shapes actually used,
    see data_access.resample_to_um_exact). The image is resampled once,
    from the nearest pyramid level; up to 8 trailing native rows/columns
    may be dropped so the effective size lands within about 0.01 percent
    of the target. Pass the returned value, not the requested one, to
    engine.register_pair.
    """
    kind = descriptor["kind"]
    fmt = descriptor["format"]

    if kind == "xenium" and descriptor.get("xenium_source") == "rendered":
        img, _s_mat = render.render_nuclei_in_folder(Path(descriptor["path"]), px_um=target_um_per_px)
        return img.astype(np.float64), float(target_um_per_px)

    if fmt == "xenium_folder_morphology":
        # Delegate to the inner morphology-image descriptor's own format,
        # already folded into this descriptor by _describe_xenium.
        inner = dict(descriptor)
        inner["path"] = descriptor["morphology_path"]
        return open_layer(
            inner, target_um_per_px, channel=channel, native_um_per_px_override=native_um_per_px_override
        )

    native_um_per_px = _resolve_native_um_per_px(descriptor, native_um_per_px_override)

    if fmt == "ome_zarr":
        path = Path(descriptor["path"])
        if kind == "rgb_brightfield":
            rgb, eff_um, _level = da.open_level(path, target_um_per_px)
            chw = np.asarray(rgb[0, :, 0, :, :])
            rgb_arr = np.moveaxis(chw, 0, -1)
            rgb_arr, achieved_um = da.resample_to_um_exact(rgb_arr, eff_um, target_um_per_px)
            return prep.he_to_hematoxylin(rgb_arr), achieved_um
        if kind == "multichannel_fluorescence":
            idx = _resolve_channel_index(descriptor, channel)
            arr, eff_um, _level = da.open_level(path, target_um_per_px)
            plane = np.asarray(arr[0, idx, 0, :, :])
            plane, achieved_um = da.resample_to_um_exact(plane, eff_um, target_um_per_px)
            return prep.normalize_dapi(plane), achieved_um
        raise ValueError(f"Cannot open_layer for kind={kind!r} at {descriptor['path']}")

    if fmt in ("ome_tiff", "tiff"):
        return _open_layer_tiff(descriptor, target_um_per_px, channel, native_um_per_px)

    if fmt in ("svs", "ndpi"):
        return _open_layer_openslide(descriptor, target_um_per_px, native_um_per_px)

    raise ValueError(f"Cannot open_layer for format={fmt!r} at {descriptor['path']}")


def _resolve_channel_index(descriptor: Dict, channel: Optional[Union[int, str]]) -> int:
    channel_names = descriptor.get("channel_names")
    if isinstance(channel, int):
        return channel
    if isinstance(channel, str):
        if channel_names is None:
            raise ValueError(f"{descriptor['path']}: channel names are unknown, cannot resolve {channel!r} by name.")
        for i, name in enumerate(channel_names):
            if name is not None and name.lower() == channel.lower():
                return i
        raise ValueError(
            f"{descriptor['path']}: channel {channel!r} not found. Available channels: {channel_names}"
        )
    # channel is None: look for a channel named like DAPI.
    if channel_names is None:
        raise ValueError(
            f"{descriptor['path']}: channel names are unknown and no channel was requested; "
            "pass channel= explicitly (an int index or a label)."
        )
    dapi_matches = [i for i, name in enumerate(channel_names) if name is not None and "dapi" in name.lower()]
    if len(dapi_matches) == 1:
        return dapi_matches[0]
    if len(dapi_matches) > 1:
        raise ValueError(
            f"{descriptor['path']}: more than one channel name looks like DAPI "
            f"({[channel_names[i] for i in dapi_matches]}); pass channel= explicitly."
        )
    raise ValueError(
        f"{descriptor['path']}: no channel named DAPI found; pass channel= explicitly. "
        f"Available channels: {channel_names}"
    )


def _open_layer_tiff(
    descriptor: Dict, target_um_per_px: float, channel: Optional[Union[int, str]], native_um_per_px: Optional[float]
) -> Tuple[np.ndarray, float]:
    import tifffile

    path = Path(descriptor["path"])
    levels = descriptor["shape_per_level"]
    known_um_levels = [lvl for lvl in levels if lvl["um_per_px"] is not None]
    if known_um_levels:
        diffs = [abs(lvl["um_per_px"] - target_um_per_px) for lvl in known_um_levels]
        best = known_um_levels[int(np.argmin(diffs))]
    else:
        best = dict(levels[0])
        best["um_per_px"] = native_um_per_px

    with tifffile.TiffFile(str(path)) as tif:
        level_series = tif.series[0].levels[best["level"]]
        arr = level_series.asarray()
        axes = level_series.axes

    if descriptor["kind"] == "rgb_brightfield":
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        resampled, achieved_um = da.resample_to_um_exact(arr, best["um_per_px"], target_um_per_px)
        return prep.he_to_hematoxylin(resampled), achieved_um

    idx = _resolve_channel_index(descriptor, channel)
    if arr.ndim == 3:
        # Use the axes tifffile reports; shape heuristics fail when C < Y.
        c_axis = axes.index("C") if "C" in axes and len(axes) == 3 else (
            0 if arr.shape[0] < arr.shape[-1] else 2)
        plane = np.take(arr, idx, axis=c_axis)
    else:
        plane = arr
    resampled, achieved_um = da.resample_to_um_exact(plane, best["um_per_px"], target_um_per_px)
    return prep.normalize_dapi(resampled), achieved_um


def _open_layer_openslide(
    descriptor: Dict, target_um_per_px: float, native_um_per_px: Optional[float]
) -> Tuple[np.ndarray, float]:
    import openslide

    path = Path(descriptor["path"])
    slide = openslide.OpenSlide(str(path))
    try:
        levels = descriptor["shape_per_level"]
        um_levels = [
            lvl["um_per_px"] if lvl["um_per_px"] is not None else native_um_per_px * (2 ** lvl["level"])
            for lvl in levels
        ]
        diffs = [abs(u - target_um_per_px) for u in um_levels]
        best_idx = int(np.argmin(diffs))
        level_um = um_levels[best_idx]
        w, h = slide.level_dimensions[best_idx]
        region = slide.read_region((0, 0), best_idx, (w, h)).convert("RGB")
        rgb = np.asarray(region)
    finally:
        slide.close()

    resampled, achieved_um = da.resample_to_um_exact(rgb, level_um, target_um_per_px)
    return prep.he_to_hematoxylin(resampled), achieved_um
