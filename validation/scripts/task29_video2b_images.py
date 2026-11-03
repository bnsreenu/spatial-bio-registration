"""Task 29: images for the Video 2b slides and Word document (read-only analysis, not part of the app).

Usage (from the repository root, UNUMLOCALIA_DIR set):
       python validation/scripts/task29_video2b_images.py [--core core01_hcc] [--out validation_rerun_output/video2b_images]

Everything comes from the app's own functions: pipeline.run_sample (matrices, diagnostics),
pipeline._open_working / render.render_nuclei_in_folder (working images), engine._prepare_working_pair and
engine._match_with_mirror_test (the exact matching call register_pair makes, which keeps all ratio-test matches
and the RANSAC inlier mask), review_overlay.build_review_images (warped layers for the D images).
No matrix, threshold or setting is changed. Random line subsets use seed 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_register as ar  # noqa: E402
import config  # noqa: E402
import data_access as da  # noqa: E402
import engine  # noqa: E402
import pipeline as pl  # noqa: E402
import render  # noqa: E402
import review_overlay as ro  # noqa: E402
import samples as sm  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
DPI = 200
TEAL = "#0F6E56"
RED = "#DC2626"
OUTLINE_TEAL = "#2DD4BF"
DATA = config.dataset_dir(required=False)
SEED = 0
MAX_ALL, MAX_INL = 200, 150
D_UM = 0.5  # display grid for the D images
REGION_UM = 400.0

MANIFEST = []


def log(msg):
    print(msg, flush=True)


def save(fig, path, group, what, use):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    h, w = plt.imread(path).shape[:2]
    MANIFEST.append({"file": path.name, "group": group, "what it shows": what, "intended use": use,
                     "width_px": w, "height_px": h})
    log(f"saved {path.name} {w}x{h}")


def new_fig(w_px, h_px, bg="black"):
    fig = plt.figure(figsize=(max(w_px, 50) / DPI, max(h_px, 50) / DPI), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


def show_gray(path, img, group, what, use, scale_bar_um=None, px_um=1.0):
    h, w = img.shape[:2]
    fig, ax = new_fig(w, h)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if scale_bar_um:
        add_scale_bar(ax, img, scale_bar_um / px_um)
    save(fig, path, group, what, use)


def add_scale_bar(ax, img, length_px):
    h, w = img.shape[:2]
    patch = img[int(h * 0.85):, : int(w * 0.25)]
    color = "black" if float(np.mean(patch)) > 0.5 else "white"
    x0, y0 = 0.04 * w, 0.95 * h
    ax.plot([x0, x0 + length_px], [y0, y0], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x0 + length_px / 2, y0 - 0.012 * h, "100 um", color=color, ha="center", va="bottom", fontsize=7)


def stretch(img, lo=1.0, hi=99.8):
    a = np.asarray(img, dtype=np.float64)
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0), 0, 1) if p1 > p0 else np.zeros_like(a)


def row_figure(path, images, captions, group, what, use, cmaps):
    h = 1000
    widths = [im.shape[1] * h / im.shape[0] for im in images]
    total_w = sum(widths)
    fig = plt.figure(figsize=((total_w + 60) / DPI * 0.6, (h * 1.0 + 60) / DPI * 0.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=widths, wspace=0.03)
    for i, (im, cap, cm) in enumerate(zip(images, captions, cmaps)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None, interpolation="nearest")
        ax.set_axis_off()
        ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=10, color="black")
    save(fig, path, group, what, use)


def xenium_polygons(xen_folder, px_um, s_mat):
    df = pd.read_parquet(Path(xen_folder) / "nucleus_boundaries.parquet")
    segs = []
    for _cid, g in df.groupby("cell_id"):
        xs = (g["vertex_x"].to_numpy(float) - s_mat[0, 2]) / px_um
        ys = (g["vertex_y"].to_numpy(float) - s_mat[1, 2]) / px_um
        if len(xs) >= 3:
            segs.append(np.column_stack([np.append(xs, xs[0]), np.append(ys, ys[0])]))
    return segs, df


def draw_outlines(ax, segs, lw=0.35, color=OUTLINE_TEAL):
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def subset(n, cap, rng):
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def side_by_side(left, right, gap=16):
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + gap + right.shape[1]), dtype=np.float64)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap:] = right
    return canvas, left.shape[1] + gap


def match_figure(path, left, right, pts_l, pts_r, inl, mode, rng, group, what, use, kp=None):
    canvas, off = side_by_side(left, right)
    fig, ax = new_fig(canvas.shape[1], canvas.shape[0])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    drawn = {}
    if mode == "keypoints":
        for pts, dx in ((kp[0], 0), (kp[1], off)):
            ax.scatter(pts[:, 0] + dx, pts[:, 1], s=1.5, c=TEAL, linewidths=0)
    else:
        kinds = [("outlier", np.nonzero(~inl)[0], RED, MAX_ALL)] if mode == "all" else []
        kinds.append(("inlier", np.nonzero(inl)[0], TEAL, MAX_ALL if mode == "all" else MAX_INL))
        for name, ids, color, cap in kinds:
            sel = ids[subset(len(ids), cap, rng)]
            drawn[name] = (len(ids), len(sel))
            segs = [[(pts_l[i, 0], pts_l[i, 1]), (pts_r[i, 0] + off, pts_r[i, 1])] for i in sel]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=0.6, alpha=0.8))
            ax.scatter(pts_l[sel, 0], pts_l[sel, 1], s=3, c=color, linewidths=0)
            ax.scatter(pts_r[sel, 0] + off, pts_r[sel, 1], s=3, c=color, linewidths=0)
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(canvas.shape[0], 0)
    save(fig, path, group, what, use)
    return drawn


def keypoints_of(img01):
    kps = cv2.AKAZE_create().detect(ar.to_uint8(img01), None)
    return np.array([k.pt for k in kps], dtype=float).reshape(-1, 2)


def step_matches(fixed, fixed_um, moving, moving_um):
    """The matching call register_pair makes (primary method, mirror test on), on the same working pair."""
    reg = engine.register_pair(fixed, moving, fixed_um, moving_um, method=pl.METHOD, mirror_test=True)
    fixed_ws, moving_ws, *_ = engine._prepare_working_pair(fixed, moving, fixed_um, moving_um)
    best = engine._match_with_mirror_test(fixed_ws, moving_ws, pl.METHOD, True)
    src, dst = np.asarray(best["pts_src"], float), np.asarray(best["pts_dst"], float)
    if best["mirrored"]:
        src[:, 0] = (moving_ws.shape[1] - 1) - src[:, 0]
    return reg, fixed_ws, moving_ws, src, dst, np.asarray(best["inlier_mask"], bool), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="core01_hcc")
    ap.add_argument("--out", default="validation_rerun_output/video2b_images")
    args = ap.parse_args()
    out = (ROOT / args.out / args.core)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sample = [s for s in sm.group_samples(sm.discover_samples(DATA)) if s["sample_id"] == args.core][0]
    L = sample["layers"]
    desc = {m: pl.describe_layer(L[m]) for m in L}
    xen_folder = Path(L["xenium"]["path"])

    # the app's own result (Xenium as reference)
    result = pl.run_sample(sample)
    log(f"run_sample: {result['status']} {result.get('confidence')}")
    mats = result["matrices"]

    # ---------------- Group A: native-like views at 2 um/px for display
    he_rgb = ro._open_rgb(desc["he"], 2.0)
    save_fig_a = {}
    fig, ax = new_fig(he_rgb.shape[1], he_rgb.shape[0], "white")
    ax.imshow(he_rgb, interpolation="nearest")
    save(fig, out / "A1_he_rgb.png", "A", "H&E color image (2 um/px display copy)", "deck slide 2, Word Figure 1")
    dapi_raw, _eff, _lvl, _idx = da.load_comet_dapi(Path(L["comet"]["path"]), 2.0)
    dapi_disp = stretch(dapi_raw, 1.0, 99.8)
    show_gray(out / "A2_comet_dapi.png", dapi_disp, "A", "COMET DAPI, 1st to 99.8th percentile stretch",
              "deck slide 2, Word Figure 1")
    xen2, _s2 = render.render_nuclei_in_folder(xen_folder, px_um=2.0)
    show_gray(out / "A3_xenium_nuclei.png", xen2, "A", "Xenium nuclei as the app renders them (2 um/px display copy)",
              "deck slide 2, Word Figure 1")
    row_figure(out / "A4_three_modalities_row.png", [he_rgb, dapi_disp, xen2], ["H&E", "COMET DAPI", "Xenium nuclei"],
               "A", "The three modalities side by side with captions", "deck slide 2, Word Figure 1",
               [None, "gray", "gray"])

    # ---------------- Group B: 1 um/px working images, as the app builds them
    he_w, he_um = pl._open_working(desc["he"], channel=L["he"].get("channel"))
    comet_w, comet_um = pl._open_working(desc["comet"], channel=L["comet"].get("channel"))
    xen_w, _s1 = render.render_nuclei_in_folder(xen_folder, px_um=pl.WORKING_UM_PER_PX)
    xen_um = pl.WORKING_UM_PER_PX
    log(f"working um/px: he {he_um:.5f} comet {comet_um:.5f} xenium {xen_um:.5f}")
    show_gray(out / "B1_he_hematoxylin_1um.png", he_w, "B", "H&E hematoxylin working image, 1 um/px, 100 um bar",
              "deck slide 4", 100, he_um)
    show_gray(out / "B2_comet_dapi_1um.png", comet_w, "B", "COMET DAPI working image, 1 um/px, 100 um bar",
              "deck slide 4", 100, comet_um)
    show_gray(out / "B3_xenium_nuclei_1um.png", xen_w, "B", "Xenium nuclei working image, 1 um/px, 100 um bar",
              "deck slide 4", 100, xen_um)
    row_figure(out / "B4_working_images_row.png", [he_w, comet_w, xen_w],
               ["H&E hematoxylin", "COMET DAPI", "Xenium nuclei"], "B", "The three working images side by side",
               "Word Figure 2", ["gray", "gray", "gray"])

    # ---------------- Group C: matching, both steps
    counts = {}
    steps = [
        ("he_to_comet", comet_w, comet_um, he_w, he_um, "H&E to COMET"),
        ("comet_to_xenium", comet_w, comet_um, xen_w, xen_um, "COMET to Xenium"),
    ]
    checks = {}
    for step, fixed, fixed_um, moving, moving_um, label in steps:
        reg, fixed_ws, moving_ws, src, dst, inl, best = step_matches(fixed, fixed_um, moving, moving_um)
        diag = next(d for d in result["diagnostics"] if d["step"] == step)
        # The step matrix of run_sample, compared with the one register_pair gives for the same pair
        step_matrix = diag["step_matrix"]
        if step == "he_to_comet":
            same_m = np.array_equal(step_matrix, reg["M"])
        else:
            same_m = None  # the stored matrix is composed into the reference frame; compared by counts below
        checks[step] = {"n_matches_equal": int(len(src)) == int(diag["n_matches"]),
                        "n_inliers_equal": int(inl.sum()) == int(diag["n_inliers"]),
                        "step_matrix_equal_register_pair": same_m, "mirrored": bool(best["mirrored"])}
        counts[step] = {"matches": int(len(src)), "inliers": int(inl.sum()), "outliers": int((~inl).sum())}
        log(f"{step}: matches {len(src)} inliers {int(inl.sum())} mirrored {best['mirrored']} checks {checks[step]}")

        if step == "he_to_comet":  # left = moving (H&E), right = fixed (COMET)
            left, right, pl_, pr_ = moving_ws, fixed_ws, src, dst
        else:  # left = fixed (COMET), right = moving (Xenium)
            left, right, pl_, pr_ = fixed_ws, moving_ws, dst, src
        kp = (keypoints_of(left), keypoints_of(right))
        counts[step]["keypoints_left"], counts[step]["keypoints_right"] = len(kp[0]), len(kp[1])
        match_figure(out / f"C1_keypoints_{step}.png", left, right, None, None, None, "keypoints", rng, "C",
                     f"{label}: AKAZE keypoints on the two working images", "spare", kp=kp)
        d_all = match_figure(out / f"C2_matches_all_{step}.png", left, right, pl_, pr_, inl, "all",
                             np.random.default_rng(SEED), "C",
                             f"{label}: all ratio-test matches, inliers teal, outliers red",
                             "deck slide 5 and 6, Word Figure 3" if step == "he_to_comet" else "spare")
        d_inl = match_figure(out / f"C3_matches_inliers_{step}.png", left, right, pl_, pr_, inl, "inliers",
                             np.random.default_rng(SEED), "C", f"{label}: RANSAC inliers only",
                             "deck slide 6, Word Figure 3" if step == "he_to_comet" else "spare")
        counts[step]["drawn_all"] = d_all
        counts[step]["drawn_inliers"] = d_inl

    # ---------------- Group D: overlays in the Xenium frame
    rev_comet = ro.build_review_images(L, result, "comet", display_um=D_UM)
    rev_he = ro.build_review_images(L, result, "he", display_um=D_UM)
    ref_img, s_d = render.render_nuclei_in_folder(xen_folder, px_um=D_UM)
    segs, vdf = xenium_polygons(xen_folder, D_UM, s_d)
    comet_d = rev_comet["moved"].astype(float) / 255.0
    he_d = rev_he["moved_rgb"]
    h, w = comet_d.shape

    fig, ax = new_fig(w, h)
    ax.imshow(comet_d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    draw_outlines(ax, segs)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    save(fig, out / "D1_xenium_outlines_on_comet.png", "D", "Xenium nucleus outlines over COMET DAPI in the Xenium frame",
         "spare")

    fig, ax = new_fig(w, h)
    ax.imshow(he_d, interpolation="nearest")
    draw_outlines(ax, segs)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    save(fig, out / "D2_xenium_outlines_on_he.png", "D", "Xenium nucleus outlines over H&E mapped through the chained matrix",
         "spare")

    vx, vy = vdf["vertex_x"].to_numpy(float), vdf["vertex_y"].to_numpy(float)
    cx, cy = (vx.min() + vx.max()) / 2.0, (vy.min() + vy.max()) / 2.0
    x0_um, y0_um = cx - REGION_UM / 2, cy - REGION_UM / 2
    x0 = (x0_um - s_d[0, 2]) / D_UM
    y0 = (y0_um - s_d[1, 2]) / D_UM
    n = REGION_UM / D_UM
    xs0, xs1, ys0, ys1 = int(round(x0)), int(round(x0 + n)), int(round(y0)), int(round(y0 + n))
    crop_he = he_d[ys0:ys1, xs0:xs1]
    crop_comet = comet_d[ys0:ys1, xs0:xs1]
    fig = plt.figure(figsize=(3 * n / DPI * 0.75 + 0.1, n / DPI * 0.75 + 0.3), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.02)
    for i, cap in enumerate(["H&E", "COMET DAPI", "Xenium outlines"]):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            ax.imshow(crop_he, interpolation="nearest")
        elif i == 1:
            ax.imshow(crop_comet, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.set_facecolor("black")
            ax.add_collection(LineCollection([s - [xs0, ys0] for s in segs], colors=OUTLINE_TEAL, linewidths=0.5))
            ax.set_xlim(0, crop_he.shape[1])
            ax.set_ylim(crop_he.shape[0], 0)
            ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    save(fig, out / "D3_three_modalities_one_frame.png", "D",
         "Same 400 um by 400 um region from H&E, COMET DAPI and Xenium outlines", "deck slide 7 (optional)")
    region = {"x_um": [x0_um, x0_um + REGION_UM], "y_um": [y0_um, y0_um + REGION_UM],
              "center_um": [cx, cy], "crop_px_he_shape": list(crop_he.shape), "crop_px_comet_shape": list(crop_comet.shape)}

    # ---------------- manifest, README, summary json
    mapping = {
        "A4_three_modalities_row.png": "deck slide 2 and Word Figure 1",
        "B1_he_hematoxylin_1um.png": "deck slide 4 (box 1)",
        "B2_comet_dapi_1um.png": "deck slide 4 (box 2)",
        "B3_xenium_nuclei_1um.png": "deck slide 4 (box 3)",
        "B4_working_images_row.png": "Word Figure 2",
        "C2_matches_all_he_to_comet.png": "deck slide 5; deck slide 6 (before); Word Figure 3 (option)",
        "C3_matches_inliers_he_to_comet.png": "deck slide 6 (after); Word Figure 3 (option)",
        "D3_three_modalities_one_frame.png": "deck slide 7 (optional)",
        "D1_xenium_outlines_on_comet.png": "spare",
        "D2_xenium_outlines_on_he.png": "spare",
    }
    mdf = pd.DataFrame(MANIFEST)
    for f in mdf["file"]:
        mdf.loc[mdf["file"] == f, "intended use"] = mapping.get(f, mdf.loc[mdf["file"] == f, "intended use"].iloc[0])
    mdf.to_csv(out / "manifest.csv", index=False)
    lines = [f"{r['file']}: {r['intended use']}" for _, r in mdf.iterrows()]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.Series({"counts": counts, "checks": checks, "region": region, "working_um": [he_um, comet_um, xen_um]}).to_json(
        out / "summary.json", indent=2)
    log(f"counts: {counts}")
    log(f"region: {region}")
    log(f"checks: {checks}")


if __name__ == "__main__":
    config.dataset_dir()
    main()
