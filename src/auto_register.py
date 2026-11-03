"""Automatic H&E to COMET registration baselines (task 5).

Two independent baseline methods, both producing a similarity transform
that maps working-resolution H&E pixel coordinates to working-resolution
COMET pixel coordinates:

  Method A (feature-based): cv2.SIFT_create / cv2.AKAZE_create keypoints
  and descriptors on (i) normalized hematoxylin/DAPI images and (ii)
  Gaussian-smoothed nucleus-density maps, matched with a ratio test, fit
  with cv2.estimateAffinePartial2D (RANSAC), which already returns a
  similarity (4 degrees of freedom: scale, rotation, translation).

  Method B (coarse-to-fine intensity): brute-force search over a small
  scale range (pixel-size ratio +/- 3 percent) and a limited rotation
  range (-10 to +10 degrees at 0.5 degree steps; see
  outputs/auto_baselines_notes.md for why a general-purpose tool would
  need the full 360 degree range), translation from phase cross-
  correlation on the nucleus-density maps, then cv2.findTransformECC
  affine refinement, projected back onto the nearest similarity.

Both methods work on images already resampled to the SAME working pixel
size with data_access.resample_to_um, so the only scale unknown is the
residual calibration mismatch between the two imaging systems, not the
nominal pixel-size difference.

Only functions are used here, no classes, per spatial_registration/CONTEXT.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from skimage.registration import phase_cross_correlation


# ---------------------------------------------------------------------------
# Shared image preparation
# ---------------------------------------------------------------------------

def to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert a 0..1 float image to uint8 for OpenCV feature detectors."""
    return np.clip(np.asarray(img01, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)


def detect_nuclei_blobs(
    img01: np.ndarray,
    um_per_px: float,
    min_sigma_um: float = 2.0,
    max_sigma_um: float = 6.0,
    num_sigma: int = 5,
    threshold: float = 0.05,
) -> np.ndarray:
    """Detect nucleus-like blobs with skimage.feature.blob_log.

    Sigma range is given in micrometers and converted to pixels with
    um_per_px, so nucleus size assumptions do not need to be rescaled by
    hand for different working resolutions. Returns an (N, 3) array of
    (y, x, sigma_px); empty (0, 3) array if nothing passes threshold.
    """
    min_sigma_px = max(0.5, min_sigma_um / um_per_px)
    max_sigma_px = max(min_sigma_px + 0.5, max_sigma_um / um_per_px)
    blobs = blob_log(
        img01,
        min_sigma=min_sigma_px,
        max_sigma=max_sigma_px,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    return blobs


def nucleus_density_map(
    img01: np.ndarray,
    um_per_px: float,
    sigma_um: float = 15.0,
    blob_kwargs: Optional[Dict] = None,
) -> Tuple[np.ndarray, int]:
    """Build a smoothed nucleus-density map from blob-detected nucleus centers.

    Places a 1 at each detected blob center, Gaussian-smooths with
    sigma_um (converted to pixels via um_per_px) to produce a continuous
    density field, then percentile-normalizes to 0..1. This is intended to
    be a modality-invariant structural signal (nucleus spatial pattern)
    for comparing H&E hematoxylin and COMET DAPI, which otherwise differ
    in absolute intensity scale and noise characteristics.

    Returns (density_map, n_blobs_detected).
    """
    blob_kwargs = blob_kwargs or {}
    blobs = detect_nuclei_blobs(img01, um_per_px, **blob_kwargs)
    points = np.zeros(img01.shape, dtype=np.float64)
    if blobs.shape[0] > 0:
        ys = np.clip(np.round(blobs[:, 0]).astype(int), 0, img01.shape[0] - 1)
        xs = np.clip(np.round(blobs[:, 1]).astype(int), 0, img01.shape[1] - 1)
        np.add.at(points, (ys, xs), 1.0)

    sigma_px = sigma_um / um_per_px
    density = gaussian_filter(points, sigma=sigma_px)
    max_val = density.max()
    if max_val > 0:
        density = density / max_val
    return density, int(blobs.shape[0])


# ---------------------------------------------------------------------------
# Similarity transform helpers
# ---------------------------------------------------------------------------

def similarity_matrix(scale: float, rotation_deg: float, tx: float, ty: float) -> np.ndarray:
    """Build a 3x3 similarity matrix from scale, rotation (deg), and translation."""
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    m = np.array(
        [
            [scale * c, -scale * s, tx],
            [scale * s, scale * c, ty],
            [0.0, 0.0, 1.0],
        ]
    )
    return m


def compose_about_centers(
    scale: float, rotation_deg: float, center_src: Tuple[float, float], center_dst: Tuple[float, float]
) -> np.ndarray:
    """Build M = T(center_dst) @ R(rotation_deg)*scale @ T(-center_src).

    Used as the coarse-to-fine candidate: rotate and scale the source
    image about its own center, then place that center at the destination
    image's center. Any residual offset (if the content is not perfectly
    centered in each crop) is found afterwards by phase cross-correlation.
    """
    t_src = np.eye(3)
    t_src[0, 2] = -center_src[0]
    t_src[1, 2] = -center_src[1]

    rs = similarity_matrix(scale, rotation_deg, 0.0, 0.0)

    t_dst = np.eye(3)
    t_dst[0, 2] = center_dst[0]
    t_dst[1, 2] = center_dst[1]

    return t_dst @ rs @ t_src


def project_affine_to_similarity(m_affine: np.ndarray) -> np.ndarray:
    """Project an affine matrix's linear part onto the nearest similarity.

    Uses the standard result that the closest (Frobenius-norm) similarity
    to a 2x2 linear map A is scale * R, where A = U S V^T (SVD),
    R = U V^T (corrected for reflection if det(R) < 0), and scale is the
    mean of the singular values. Translation is kept unchanged.
    """
    a = np.asarray(m_affine[:2, :2], dtype=np.float64)
    u, s, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    scale = float(np.mean(s))

    out = np.eye(3)
    out[:2, :2] = scale * r
    out[0, 2] = m_affine[0, 2]
    out[1, 2] = m_affine[1, 2]
    return out


def decompose_rotation_deg(m: np.ndarray) -> float:
    """Rotation in degrees implied by the first column of a 2x2 linear block."""
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


# ---------------------------------------------------------------------------
# Method A: feature-based
# ---------------------------------------------------------------------------

def feature_based_similarity(
    src01: np.ndarray,
    dst01: np.ndarray,
    method: str = "sift",
    ratio: float = 0.75,
    ransac_thresh_px: float = 3.0,
    return_points: bool = False,
) -> Dict:
    """Detect, match, and fit a similarity transform mapping src into dst.

    method is "sift" (cv2.SIFT_create, float descriptors, L2 ratio test) or
    "akaze" (cv2.AKAZE_create, binary descriptors, Hamming ratio test).
    Matches pass Lowe's ratio test (good if m.distance < ratio * n.distance
    for the two nearest neighbors), then cv2.estimateAffinePartial2D with
    RANSAC fits a similarity (scale, rotation, translation only).

    Returns a dict with keys: status ("ok" or a failure reason), M (3x3,
    None on failure), n_keypoints_src, n_keypoints_dst, n_matches,
    n_inliers, inlier_ratio.

    return_points (task 9, item 3: keypoint export) is additive and
    defaults to False, so every existing caller is unaffected. When True,
    the result also carries "pts_src", "pts_dst" (all ratio-test matches,
    in each image's own pixel coordinates, aligned by row) and
    "inlier_mask" (bool array, same length, True for the points RANSAC
    kept), or None for all three if matching failed before RANSAC ran.
    """
    src_u8 = to_uint8(src01)
    dst_u8 = to_uint8(dst01)

    if method == "sift":
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif method == "akaze":
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'sift' or 'akaze'.")

    kp_src, des_src = detector.detectAndCompute(src_u8, None)
    kp_dst, des_dst = detector.detectAndCompute(dst_u8, None)

    result = {
        "status": "ok",
        "M": None,
        "n_keypoints_src": 0 if kp_src is None else len(kp_src),
        "n_keypoints_dst": 0 if kp_dst is None else len(kp_dst),
        "n_matches": 0,
        "n_inliers": 0,
        "inlier_ratio": float("nan"),
    }
    if return_points:
        result["pts_src"] = None
        result["pts_dst"] = None
        result["inlier_mask"] = None

    if des_src is None or des_dst is None or len(des_src) < 2 or len(des_dst) < 2:
        result["status"] = "too_few_keypoints"
        return result

    matcher = cv2.BFMatcher(norm)
    knn_matches = matcher.knnMatch(des_src, des_dst, k=2)
    good = [m for pair in knn_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    result["n_matches"] = len(good)

    if len(good) < 3:
        result["status"] = "too_few_matches"
        return result

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in good])
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in good])

    if return_points:
        result["pts_src"] = pts_src
        result["pts_dst"] = pts_dst
        result["inlier_mask"] = None

    m_2x3, inliers = cv2.estimateAffinePartial2D(
        pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px
    )

    if m_2x3 is None:
        result["status"] = "ransac_failed"
        return result

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    result["n_inliers"] = n_inliers
    result["inlier_ratio"] = float(n_inliers / len(good))
    result["M"] = np.vstack([m_2x3, [0.0, 0.0, 1.0]])
    if return_points:
        result["inlier_mask"] = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(good), dtype=bool)
    if n_inliers < 3:
        result["status"] = "too_few_inliers"
    return result


# ---------------------------------------------------------------------------
# Method B: coarse-to-fine intensity (density-map based)
# ---------------------------------------------------------------------------

def coarse_scale_rotation_translation_search(
    src_density: np.ndarray,
    dst_density: np.ndarray,
    scale_search_frac: float = 0.03,
    n_scale_steps: int = 7,
    rot_range_deg: float = 10.0,
    rot_step_deg: float = 0.5,
) -> Dict:
    """Brute-force search over scale and rotation; translation by phase correlation.

    Both density maps are assumed to already be at the same working pixel
    size (data_access.resample_to_um), so the scale search is only for a
    residual calibration mismatch between the two imaging systems, hence
    the narrow +/- scale_search_frac range (not a full scale search).

    Rotation is searched over [-rot_range_deg, +rot_range_deg] at
    rot_step_deg steps. This is a deliberately narrow range appropriate
    for roughly pre-aligned serial sections; a general-purpose tool would
    need the full 0-360 degree range, which this function does not search
    (documented, not hidden).

    For each (scale, rotation) candidate, src_density is rotated and
    scaled about its own center and placed at dst_density's center
    (auto_register.compose_about_centers), then skimage.registration.
    phase_cross_correlation finds the residual translation against
    dst_density. The candidate with the highest cross-correlation peak
    (1 - error, as returned by phase_cross_correlation) is kept.

    Returns a dict: status, M (3x3, src->dst, at the shared working
    resolution), best_scale, best_rotation_deg, best_score, n_candidates.
    """
    h_src, w_src = src_density.shape
    h_dst, w_dst = dst_density.shape
    center_src = (w_src / 2.0, h_src / 2.0)
    center_dst = (w_dst / 2.0, h_dst / 2.0)

    scales = np.linspace(1.0 - scale_search_frac, 1.0 + scale_search_frac, n_scale_steps)
    rotations = np.arange(-rot_range_deg, rot_range_deg + 1e-9, rot_step_deg)

    best = {"score": -np.inf, "scale": None, "rotation_deg": None, "M": None}
    n_candidates = 0

    for scale in scales:
        for rotation_deg in rotations:
            n_candidates += 1
            m_candidate = compose_about_centers(scale, rotation_deg, center_src, center_dst)
            warped_src = cv2.warpAffine(
                src_density.astype(np.float32),
                m_candidate[:2, :],
                (w_dst, h_dst),
                flags=cv2.INTER_LINEAR,
                borderValue=0.0,
            )
            if warped_src.max() <= 0 or dst_density.max() <= 0:
                continue
            try:
                shift_yx, error, _ = phase_cross_correlation(
                    dst_density, warped_src, upsample_factor=4, normalization=None
                )
            except Exception:
                continue
            score = 1.0 - float(error)
            if score > best["score"]:
                t_residual = np.eye(3)
                t_residual[0, 2] = float(shift_yx[1])
                t_residual[1, 2] = float(shift_yx[0])
                m_refined = t_residual @ m_candidate
                best.update(
                    score=score, scale=float(scale), rotation_deg=float(rotation_deg), M=m_refined
                )

    if best["M"] is None:
        return {
            "status": "search_failed",
            "M": None,
            "best_scale": None,
            "best_rotation_deg": None,
            "best_score": None,
            "n_candidates": n_candidates,
        }

    return {
        "status": "ok",
        "M": best["M"],
        "best_scale": best["scale"],
        "best_rotation_deg": best["rotation_deg"],
        "best_score": best["score"],
        "n_candidates": n_candidates,
    }


def ecc_refine(
    src_density: np.ndarray, dst_density: np.ndarray, m_init: np.ndarray, n_iterations: int = 200, eps: float = 1e-7
) -> Dict:
    """Refine a coarse src->dst similarity with cv2.findTransformECC (affine).

    cv2.findTransformECC requires templateImage and inputImage to be the
    same size, but src_density and dst_density generally are not (H&E and
    COMET fields of view differ even after resample_to_um brings both to
    the same um/px). To work around this, src_density is first warped with
    m_init into dst_density's own shape (cv2.warpAffine(src, m_init[:2,:],
    dst_shape), consistent with the project's convention that M maps
    src_img's frame into the output/dst frame), giving src_warped already
    the same shape as dst_density and already close to aligned.

    cv2.findTransformECC(templateImage, inputImage, warpMatrix, ...) finds
    warpMatrix such that templateImage(p) = inputImage(warpMatrix(p)), i.e.
    warpMatrix maps TEMPLATE coordinates into INPUT coordinates (verified
    empirically against a synthetic known shift, see task 5 development
    notes). With template=dst_density, input=src_warped, and warp_init the
    identity (since src_warped is already pre-aligned by m_init), the
    returned warp maps dst coordinates into src_warped coordinates; its
    inverse R maps src_warped coordinates into dst coordinates, i.e. the
    residual correction on top of m_init. The final src->dst transform is
    M = R @ m_init.

    Returns a dict: status, M (3x3 src->dst affine, or None on failure),
    correlation (ECC's final correlation coefficient).
    """
    h_dst, w_dst = dst_density.shape
    try:
        src_warped = cv2.warpAffine(
            src_density.astype(np.float32),
            m_init[:2, :],
            (w_dst, h_dst),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
    except cv2.error as exc:
        return {"status": f"warp_failed: {exc}", "M": None, "correlation": None}

    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations, eps)

    try:
        cc, warp_ecc = cv2.findTransformECC(
            dst_density.astype(np.float32),
            src_warped,
            warp_init,
            cv2.MOTION_AFFINE,
            criteria,
        )
    except cv2.error as exc:
        return {"status": f"ecc_failed: {exc}", "M": None, "correlation": None}

    m_dst_to_src_warped = np.vstack([warp_ecc, [0.0, 0.0, 1.0]])
    try:
        r = np.linalg.inv(m_dst_to_src_warped)
    except np.linalg.LinAlgError:
        return {"status": "singular_refined", "M": None, "correlation": None}

    m_final = r @ m_init
    return {"status": "ok", "M": m_final, "correlation": float(cc)}


def coarse_to_fine_register(
    src01: np.ndarray,
    dst01: np.ndarray,
    um_per_px: float,
    scale_search_frac: float = 0.03,
    n_scale_steps: int = 7,
    rot_range_deg: float = 10.0,
    rot_step_deg: float = 0.5,
    sigma_um: float = 15.0,
) -> Dict:
    """Full Method B pipeline: density maps, coarse search, ECC refine, project.

    src01 and dst01 must already be at the same um_per_px (see
    data_access.resample_to_um). Returns a dict with the final 3x3
    src->dst similarity M (projected from the ECC affine result, or from
    the coarse search if ECC failed), plus diagnostics from each stage.
    """
    src_density, n_blobs_src = nucleus_density_map(src01, um_per_px, sigma_um=sigma_um)
    dst_density, n_blobs_dst = nucleus_density_map(dst01, um_per_px, sigma_um=sigma_um)

    coarse = coarse_scale_rotation_translation_search(
        src_density,
        dst_density,
        scale_search_frac=scale_search_frac,
        n_scale_steps=n_scale_steps,
        rot_range_deg=rot_range_deg,
        rot_step_deg=rot_step_deg,
    )

    result = {
        "status": coarse["status"],
        "M": None,
        "n_blobs_src": n_blobs_src,
        "n_blobs_dst": n_blobs_dst,
        "coarse_scale": coarse["best_scale"],
        "coarse_rotation_deg": coarse["best_rotation_deg"],
        "coarse_score": coarse["best_score"],
        "ecc_status": None,
        "ecc_correlation": None,
    }

    if coarse["M"] is None:
        return result

    ecc = ecc_refine(src_density, dst_density, coarse["M"])
    result["ecc_status"] = ecc["status"]
    result["ecc_correlation"] = ecc["correlation"]

    if ecc["M"] is not None:
        result["status"] = "ok"
        result["M"] = project_affine_to_similarity(ecc["M"])
    else:
        # ECC refinement failed; fall back to the coarse estimate (already
        # a similarity by construction) rather than discarding the method.
        result["status"] = "ok_coarse_only"
        result["M"] = coarse["M"]

    return result
