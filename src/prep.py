"""Image preprocessing helpers for QC: stain separation, normalization, masking.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
These are preprocessing utilities for visualization and evaluation; they are
not a registration method.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from skimage.util import img_as_float


def percentile_normalize(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip to the [low, high] percentile range and rescale to 0..1.

    Used by both he_to_hematoxylin and normalize_dapi so both modalities get
    the same contrast-normalization treatment before QC comparison.
    """
    img = np.asarray(img, dtype=np.float64)
    p_lo, p_hi = np.percentile(img, [low, high])
    if p_hi <= p_lo:
        # Degenerate (near-constant) image: avoid dividing by zero.
        return np.zeros_like(img)
    clipped = np.clip(img, p_lo, p_hi)
    return (clipped - p_lo) / (p_hi - p_lo)


def he_to_hematoxylin(rgb: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Convert an RGB H&E image to a percentile-normalized hematoxylin channel.

    Uses skimage.color.rgb2hed to separate the H&E stain into Hematoxylin,
    Eosin, and DAB-like components, keeps the Hematoxylin channel (index 0,
    which stains cell nuclei, the natural counterpart to DAPI), and
    percentile-normalizes it to 0..1 for visual and metric comparability
    with normalize_dapi.
    """
    rgb_float = img_as_float(np.asarray(rgb))
    hed = rgb2hed(rgb_float)
    hematoxylin = hed[..., 0]
    return percentile_normalize(hematoxylin, low, high)


def normalize_dapi(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-clip and scale a DAPI image to 0..1."""
    return percentile_normalize(img, low, high)


def tissue_mask(
    img: np.ndarray,
    um_per_px: float,
    opening_radius_um: float = 2.0,
    closing_radius_um: float = 40.0,
    min_object_size: int = 256,
) -> np.ndarray:
    """A simple, documented tissue mask from a single-channel 0..1 image.

    Steps (all standard skimage/scipy building blocks, no learned model):
      1. Otsu threshold on the image to separate foreground (tissue /
         nuclei signal) from background.
      2. Binary opening (disk, opening_radius_um converted to pixels) to
         remove isolated speckle noise in the background.
      3. Binary closing (disk, closing_radius_um converted to pixels) to
         merge the sparse, speckled nuclei signal into a single solid
         tissue region. Both the hematoxylin channel and DAPI only light
         up nuclei, so the raw Otsu mask is a field of small dots, not a
         solid tissue blob; closing_radius_um must be large enough to
         bridge the typical spacing between nuclei.
      4. Remove small connected components below min_object_size pixels.
      5. Fill fully enclosed holes.

    Radii are specified in micrometers and converted to pixels with
    um_per_px (round(radius_um / um_per_px), minimum 1 px) so the same
    physical closing radius is used regardless of working resolution,
    unlike pixel-radius parameters which would need manual rescaling.

    The default closing_radius_um=40 was chosen by inspecting the
    resulting mask on core01_hcc H&E hematoxylin: smaller physical radii
    left holes between nuclei clusters, 40 um filled the tissue core into
    one solid disc. See outputs/qc_masks.png and
    outputs/matrix_reproduction_notes.md (task 5, item 0a) for a check of
    whether this generalizes across cores: core04_hca still has much lower
    mask coverage than the other cores even at this radius, which was
    verified to reflect genuinely sparser nuclei density (confirmed
    independently in both H&E hematoxylin and COMET DAPI raw Otsu
    foreground fractions, before any morphology), not a thresholding
    failure specific to one modality or core.

    Intended for hematoxylin or DAPI images where tissue/nuclei are bright
    relative to background. Returns a boolean mask the same shape as img.
    """
    img = np.asarray(img, dtype=np.float64)
    threshold = threshold_otsu(img)
    mask = img > threshold

    opening_radius_px = max(1, round(opening_radius_um / um_per_px))
    closing_radius_px = max(1, round(closing_radius_um / um_per_px))

    mask = binary_opening(mask, footprint=disk(opening_radius_px))
    mask = binary_closing(mask, footprint=disk(closing_radius_px))
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = ndi.binary_fill_holes(mask)
    return mask
