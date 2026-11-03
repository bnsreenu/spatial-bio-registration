"""Task 8, item 1: render Xenium point/polygon data into raster images.

Xenium nucleus_boundaries.parquet gives polygon vertices (vertex_x,
vertex_y) in micrometers, with y increasing downward (task wording,
matching the usual image row convention, so no flip is applied here).
cells.csv gives one centroid (x_centroid, y_centroid) per cell, same
units and orientation.

Pixel-center convention: rendered pixel (col, row) as an INDEX has its
center at Xenium micrometer coordinate (minx + col * px_um, miny + row *
px_um). This is the same convention used elsewhere in this project for
pixel coordinates (data_access.py, fit_transforms.py): a pixel's integer
index is the point that gets transformed, not a pixel corner.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

PAD_UM = 20.0


def _xenium_folder(data_dir: Path, core_id: str) -> Path:
    return data_dir / core_id / "xenium"


def _bounds(xs: np.ndarray, ys: np.ndarray, pad_um: float = PAD_UM) -> Tuple[float, float, float, float]:
    return float(xs.min() - pad_um), float(xs.max() + pad_um), float(ys.min() - pad_um), float(ys.max() + pad_um)


def _s_matrix(minx: float, miny: float, px_um: float) -> np.ndarray:
    """3x3 matrix mapping rendered pixel (col, row) -> Xenium micrometer (x, y)."""
    return np.array([[px_um, 0.0, minx], [0.0, px_um, miny], [0.0, 0.0, 1.0]], dtype=np.float64)


def render_nuclei(
    data_dir: Path, core_id: str, px_um: float = 1.0, blur_sigma_um: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize nucleus_boundaries.parquet polygons into a density image.

    Fills each cell's nucleus polygon (grouped by cell_id, vertex_x/
    vertex_y in micrometers) with cv2.fillPoly into a binary mask at
    px_um resolution, then applies a light Gaussian blur (blur_sigma_um,
    converted to pixels via px_um) so the image has smooth gradients
    usable by feature detectors, and normalizes to 0..1.

    Returns (img01, S): S is the explicit 3x3 matrix mapping rendered
    pixel coordinates to Xenium micrometer coordinates (see module
    docstring for the pixel-center convention):

        [x, y, 1]^T = S @ [col, row, 1]^T
    """
    return render_nuclei_in_folder(_xenium_folder(data_dir, core_id), px_um=px_um, blur_sigma_um=blur_sigma_um)


def render_nuclei_in_folder(
    xenium_folder: Path, px_um: float = 1.0, blur_sigma_um: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as render_nuclei, but takes the Xenium folder path directly,
    rather than (data_dir, core_id). Added for task 11's src/io_layers.py,
    which discovers Xenium folders by any folder convention, not only
    data/UnumLocalia's <data_dir>/<core_id>/xenium layout. render_nuclei
    delegates to this function; its own behavior and return values are
    unchanged.
    """
    path = Path(xenium_folder) / "nucleus_boundaries.parquet"
    df = pd.read_parquet(path)

    minx, maxx, miny, maxy = _bounds(
        df["vertex_x"].to_numpy(dtype=float), df["vertex_y"].to_numpy(dtype=float)
    )
    width = int(np.ceil((maxx - minx) / px_um)) + 1
    height = int(np.ceil((maxy - miny) / px_um)) + 1

    canvas = np.zeros((height, width), dtype=np.uint8)
    for _cid, group in df.groupby("cell_id"):
        xs = (group["vertex_x"].to_numpy(dtype=float) - minx) / px_um
        ys = (group["vertex_y"].to_numpy(dtype=float) - miny) / px_um
        if xs.shape[0] < 3:
            continue
        pts = np.round(np.stack([xs, ys], axis=1)).astype(np.int32)
        cv2.fillPoly(canvas, [pts], color=1)

    img = canvas.astype(np.float64)
    sigma_px = blur_sigma_um / px_um
    if sigma_px > 0:
        img = gaussian_filter(img, sigma=sigma_px)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val

    s_mat = _s_matrix(minx, miny, px_um)
    return img, s_mat


def render_centroids(
    data_dir: Path, core_id: str, px_um: float = 1.0, sigma_um: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Splat cells.csv centroids as Gaussians into a density image.

    Places a 1 at each cell centroid's nearest rendered pixel, then
    Gaussian-smooths with sigma_um (converted to pixels via px_um) and
    normalizes to 0..1. Same return signature and pixel-center convention
    as render_nuclei: (img01, S) with S mapping rendered pixel (col, row)
    to Xenium micrometer (x, y).
    """
    path = _xenium_folder(data_dir, core_id) / "cells.csv"
    df = pd.read_csv(path)

    minx, maxx, miny, maxy = _bounds(
        df["x_centroid"].to_numpy(dtype=float), df["y_centroid"].to_numpy(dtype=float)
    )
    width = int(np.ceil((maxx - minx) / px_um)) + 1
    height = int(np.ceil((maxy - miny) / px_um)) + 1

    xs = (df["x_centroid"].to_numpy(dtype=float) - minx) / px_um
    ys = (df["y_centroid"].to_numpy(dtype=float) - miny) / px_um
    xs_i = np.clip(np.round(xs).astype(int), 0, width - 1)
    ys_i = np.clip(np.round(ys).astype(int), 0, height - 1)

    canvas = np.zeros((height, width), dtype=np.float64)
    np.add.at(canvas, (ys_i, xs_i), 1.0)

    sigma_px = sigma_um / px_um
    img = gaussian_filter(canvas, sigma=sigma_px)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val

    s_mat = _s_matrix(minx, miny, px_um)
    return img, s_mat
