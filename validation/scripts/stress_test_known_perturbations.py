"""Synthetic deformation stress test (measurement only).

Usage (from the repository root, UNUMLOCALIA_DIR set):
       python validation/scripts/stress_test_known_perturbations.py run [step ...] [--cores core01_hcc core03_tonsil]
       python validation/scripts/stress_test_known_perturbations.py plot
       python validation/scripts/stress_test_known_perturbations.py summary

Moving images are the working images the app hands to engine.register_pair
(1 um/px, so px == um). A known transform f (about the tissue centre) is
applied to the moving image; the truth for a moving point x is M_base @ x,
so the expected transform for the warped image is M_base @ inv(f).
Error = Euclidean distance (um) between M' f(G) and M_base G on the standard
in-tissue grid G (step 8 px of the tissue mask), which equals
evaluation.transform_difference_um for affine f (cross-checked, see
xcheck column) and is also defined for elastic f.
No engine code is changed. The only extra calls are
auto_register.feature_based_similarity(return_points=True) (existing
function) to get the raw ratio-test matches for the affine experiment.
cv2 RANSAC has no seed argument; AKAZE and cv2 RANSAC were deterministic in
repeated runs (baseline_repeat rows). The elastic fields use fixed seeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_register as ar  # noqa: E402
import config  # noqa: E402
import engine  # noqa: E402
import evaluation as ev  # noqa: E402
import pipeline as pl  # noqa: E402
import prep  # noqa: E402
import render  # noqa: E402
import samples as sm  # noqa: E402

DATA = config.dataset_dir(required=False)
OUT = ROOT / "validation_rerun_output"
CSV = OUT / "stress_test_core01_core03.csv"
CORES = ["core01_hcc", "core03_tonsil"]
STEPS = ["he_to_comet", "comet_to_xenium"]
ELASTIC_SIGMA_PX = 80.0
ACCURATE_UM = 2.0


# ---------------------------------------------------------------- transforms
def about(c, m):
    t1 = np.eye(3)
    t1[:2, 2] = -c
    t2 = np.eye(3)
    t2[:2, 2] = c
    return t2 @ m @ t1


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def stretch(pct, axis_deg):
    d = np.diag([1 + pct / 100.0, 1.0, 1.0])
    return rot(axis_deg) @ d @ rot(-axis_deg)


def shear(deg):
    m = np.eye(3)
    m[0, 1] = np.tan(np.deg2rad(deg))
    return m


def elastic_field(shape, peak_um, seed):
    rng = np.random.default_rng(seed)
    d = np.stack([gaussian_filter(rng.standard_normal(shape), ELASTIC_SIGMA_PX) for _ in range(2)])
    d *= peak_um / np.sqrt((d ** 2).sum(0)).max()
    return d  # d[0]=dx, d[1]=dy on the output grid


def apply_pts(m, p):
    return p @ m[:2, :2].T + m[:2, 2]


def elastic_forward(d, g):
    """y = x + d(y), solved by fixed-point iteration (inverse of the remap)."""
    y = g.copy()
    for _ in range(30):
        dx = map_coordinates(d[0], [y[:, 1], y[:, 0]], order=1, mode="nearest")
        dy = map_coordinates(d[1], [y[:, 1], y[:, 0]], order=1, mode="nearest")
        y = g + np.stack([dx, dy], 1)
    return y


def inject(img, ops):
    """Warp img by ops (list of ('affine', T3) or ('elastic', d)); return warped, point map."""
    h, w = img.shape
    out = img.astype(np.float32)
    for kind, v in ops:
        if kind == "affine":
            out = cv2.warpAffine(out, v[:2], (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            out = cv2.remap(out, xx - v[0].astype(np.float32), yy - v[1].astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def pmap(g):
        p = g.copy()
        for kind, v in ops:
            p = apply_pts(v, p) if kind == "affine" else elastic_forward(v, p)
        return p

    return out.astype(np.float64), pmap


def fit_sim_free(src, dst):
    """Least-squares similarity src->dst, with or without reflection; best of the two."""
    best = None
    for refl in (False, True):
        s = src * np.array([-1.0, 1.0]) if refl else src
        ms, md = s.mean(0), dst.mean(0)
        a, b = s - ms, dst - md
        u, sv, vt = np.linalg.svd(b.T @ a)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            sv[-1] *= -1
            r = u @ vt
        scale = sv.sum() / (a ** 2).sum()
        pred = scale * (a @ r.T) + md
        err = np.linalg.norm(pred - dst, axis=1)
        if best is None or err.mean() < best.mean():
            best = err
    return best


def quad_r2(pos, r):
    p = (pos - pos.mean(0)) / max(np.ptp(pos, axis=0).max() / 2, 1e-9)
    x, y = p[:, 0], p[:, 1]
    f = np.column_stack([np.ones(len(p)), x, y, x * x, x * y, y * y])
    ssr = sst = 0.0
    for k in range(2):
        c = np.linalg.lstsq(f, r[:, k], rcond=None)[0]
        ssr += ((r[:, k] - f @ c) ** 2).sum()
        sst += ((r[:, k] - r[:, k].mean()) ** 2).sum()
    obs = 1 - ssr / max(sst, 1e-12)
    n = len(p)
    return obs, 1 - (1 - obs) * (n - 1) / max(n - 6, 1)


# --------------------------------------------------------------------- cases
def build_cases(c, shape):
    A = lambda m: [("affine", about(c, m))]  # noqa: E731
    cases = []
    for t in (5, 20, 100, 500):
        m = np.eye(3)
        m[0, 2] = t
        cases.append(("a_translation", t, "um", 0, [("affine", m)]))
    for a in (1, 5, 15, 45, 90, 135, 180, 270):
        cases.append(("b_rotation", a, "deg", 0, A(rot(a))))
    cases.append(("c_mirror", "horizontal", "", 0, A(np.diag([-1.0, 1, 1]))))
    cases.append(("c_mirror", "vertical", "", 0, A(np.diag([1.0, -1, 1]))))
    for p in (-10, -5, -2, 2, 5, 10):
        s = 1 + p / 100.0
        cases.append(("d_scale", p, "percent", 0, A(np.diag([s, s, 1.0]))))
    for p in (1, 2, 5):
        cases.append(("e_stretch_x", p, "percent", 0, A(stretch(p, 0))))
    for p in (1, 2, 5):
        cases.append(("e_stretch_30deg", p, "percent", 0, A(stretch(p, 30))))
    for s in (1, 3):
        cases.append(("f_shear", s, "deg", 0, A(shear(s))))
    for pk in (2, 5, 10, 20):
        for seed in (0, 1, 2):
            cases.append(("g_elastic", pk, "um", seed, [("elastic", elastic_field(shape, pk, seed))]))
    cases.append(("h_rot30_stretch3", 30, "deg+3pct", 0, A(rot(30) @ stretch(3, 0))))
    cases.append(("h_rot90_elastic5", 90, "deg+5um", 0,
                  A(rot(90)) + [("elastic", elastic_field(shape, 5, 0))]))
    for p in (2, 5):
        cases.append(("i_wrong_pixel_size", p, "percent", 0, []))
    return cases


# ----------------------------------------------------------------- one run
def one_run(fixed, fixed_um, moving, moving_um, mbase, grid, ops, declared_scale=1.0):
    img, pmap = inject(moving, ops) if ops else (moving, lambda g: g.copy())
    P = pmap(grid)
    truth = apply_pts(mbase, grid)
    reg = engine.register_pair(fixed, img, fixed_um, moving_um * declared_scale, method=pl.METHOD, mirror_test=True)
    rec = {"confidence": reg["confidence"], "n_inliers": reg["n_inliers"], "inlier_ratio": reg["inlier_ratio"],
           "mirrored": reg["mirrored"], "returned": reg["M"] is not None}
    so = reg["second_opinion"]["diff_um"]
    rec["sift_diff_um"] = so["mean_um"] if so else np.nan
    scale_base = np.sqrt(abs(np.linalg.det(mbase[:2, :2])))
    if reg["M"] is None:
        return rec, None, None, None
    m = reg["M"]
    d = np.linalg.norm(apply_pts(m, P) - truth, axis=1)
    rec["err_mean_um"], rec["err_max_um"] = float(d.mean()), float(d.max())
    rec["succeeded"] = bool(d.mean() <= ACCURATE_UM)
    # cross-check against evaluation.transform_difference_um for affine injections
    if ops and all(k == "affine" for k, _ in ops):
        t_all = np.eye(3)
        for _, v in ops:
            t_all = v @ t_all
        e_mean, _, e_max, _ = ev.transform_difference_um(m, mbase @ np.linalg.inv(t_all), P, comet_px_um=1.0)
        rec["xcheck_abs_diff"] = abs(e_mean - d.mean())
    # floor: best similarity (any reflection) fitted to the true displacement field
    fl = fit_sim_free(P, grid) * scale_base
    rec["floor_mean_um"], rec["floor_max_um"] = float(fl.mean()), float(fl.max())
    # inlier residual detector
    src, dst = reg["src_pts"], reg["dst_pts"]
    if len(src) >= 10:
        res = dst - apply_pts(m, src)
        rec["inlier_rms_um"] = float(np.sqrt((res ** 2).sum(1).mean()))
        rec["quad_r2"], rec["quad_r2_adj"] = quad_r2(src, res)
    return rec, reg, img, P


def affine_experiment(fixed, img, reg, mbase, grid, P):
    """Full affine with own RANSAC on the raw ratio-test matches (same matcher as the engine)."""
    mov = cv2.flip(img.astype(np.float32), 1) if reg["mirrored"] else img
    raw = ar.feature_based_similarity(mov, fixed, method=pl.METHOD, return_points=True)
    if raw["pts_src"] is None or len(raw["pts_src"]) < 4:
        return {}
    a2, inl = cv2.estimateAffine2D(raw["pts_src"], raw["pts_dst"], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if a2 is None:
        return {}
    a3 = np.vstack([a2, [0, 0, 1.0]])
    if reg["mirrored"]:
        a3 = a3 @ engine.mirror_matrix(img.shape[1])
    d = np.linalg.norm(apply_pts(a3, P) - apply_pts(mbase, grid), axis=1)
    return {"aff_err_mean_um": float(d.mean()), "aff_err_max_um": float(d.max()),
            "aff_n_raw": int(len(raw["pts_src"])), "aff_n_inliers": int(inl.sum())}


# ---------------------------------------------------------------------- run
def load(core, step):
    s = [x for x in sm.group_samples(sm.discover_samples(DATA)) if x["sample_id"] == core][0]
    L = s["layers"]
    im = {}
    for m in ("he", "comet"):
        d = pl.describe_layer(L[m])
        im[m] = pl._open_working(d, channel=L[m].get("channel"))
    fixed, fixed_um = im["comet"]
    if step == "he_to_comet":
        moving, moving_um = im["he"]
    else:
        r, _ = render.render_nuclei_in_folder(Path(L["xenium"]["path"]), px_um=1.0)
        moving, moving_um = r, 1.0
    return fixed, fixed_um, moving, moving_um


def run(steps, cores):
    rows = []
    for core in cores:
        for step in steps:
            fixed, fixed_um, moving, moving_um = load(core, step)
            mask = prep.tissue_mask(moving, moving_um)
            grid = ev.sample_grid_in_mask(mask, moving_um, 1.0, step_px_mask=8)
            ys, xs = np.nonzero(mask)
            c = np.array([xs.mean(), ys.mean()])
            base = engine.register_pair(fixed, moving, fixed_um, moving_um, method=pl.METHOD, mirror_test=True)
            mbase = base["M"]
            print(core, step, "baseline", base["confidence"], base["n_inliers"], "grid", len(grid), flush=True)
            cases = [("baseline_repeat", 0, "", 0, [])] + build_cases(c, moving.shape)
            # self-test of the direction convention: tiny translation must give ~0 error
            for case, mag, unit, seed, ops in cases:
                t0 = time.time()
                dec = 1 + mag / 100.0 if case == "i_wrong_pixel_size" else 1.0
                rec, reg, img, P = one_run(fixed, fixed_um, moving, moving_um, mbase, grid, ops, dec)
                if reg is not None and case[0] in "defg":
                    rec.update(affine_experiment(fixed, img, reg, mbase, grid, P))
                rec.update(core=core, step=step, case=case, magnitude=mag, unit=unit, seed=seed,
                           seconds=time.time() - t0)
                rows.append(rec)
                pd.DataFrame(rows).to_csv(CSV, index=False)
                print(f"  {case} {mag} s{seed}: {rec['confidence']} inl={rec['n_inliers']} "
                      f"err={rec.get('err_mean_um', float('nan')):.2f}/{rec.get('err_max_um', float('nan')):.2f} "
                      f"floor={rec.get('floor_mean_um', float('nan')):.2f} {rec['seconds']:.1f}s", flush=True)
    return rows


# --------------------------------------------------------------------- plot
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    df = pd.read_csv(CSV)
    df = df[df.case != "baseline_repeat"]
    panels = [("a_translation", "Translation (um)"), ("b_rotation", "Rotation (deg)"),
              ("c_mirror", "Mirror"), ("d_scale", "Uniform scale (%)"),
              (["e_stretch_x", "e_stretch_30deg"], "Anisotropic stretch (%)"), ("f_shear", "Shear (deg)"),
              ("g_elastic", "Smooth elastic,\npeak displacement (um)"),
              (["h_rot30_stretch3", "h_rot90_elastic5"], "Combinations"),
              ("i_wrong_pixel_size", "Wrong pixel size (%)")]
    names = {"he_to_comet": "H&E to COMET", "comet_to_xenium": "COMET to Xenium (rendered)"}
    for step in df.step.unique():
        fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
        d = df[df.step == step]
        for ax, (cases, title) in zip(axes.ravel(), panels):
            cases = [cases] if isinstance(cases, str) else cases
            for core, col in zip(sorted(d.core.unique()), ("tab:blue", "tab:orange")):
                for case, ls in zip(cases, ("-", "--")):
                    s = d[(d.core == core) & (d.case == case)]
                    if s.empty:
                        continue
                    if case.startswith("c_") or case.startswith("h_"):
                        x = np.arange(len(s)) + (cases.index(case) if case.startswith("h_") else 0)
                        ax.set_xticks(range(2 if case.startswith("h_") else len(s)))
                        ax.set_xticklabels(["rot30+\nstretch3%", "rot90+\nelastic5um"] if case.startswith("h_") else [str(v) for v in s.magnitude])
                    else:
                        x = s.magnitude.astype(float).values
                    em = s.groupby("magnitude", sort=False).err_mean_um.mean().values if case.startswith("g_") else s.err_mean_um.values
                    ex = s.groupby("magnitude", sort=False).err_max_um.mean().values if case.startswith("g_") else s.err_max_um.values
                    if case.startswith("g_"):
                        x = s.magnitude.astype(float).unique()
                    lab = f"{core[:6]} {case.split('_', 1)[1]}"
                    ax.plot(x, np.clip(em, 0.05, None), "o" + ls, color=col, label=lab + " mean")
                    ax.plot(x, np.clip(ex, 0.05, None), "x:", color=col, alpha=0.6, label=lab + " max")
            ax.axhline(ACCURATE_UM, color="gray", lw=0.8)
            ax.set_yscale("log")
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Error (um)")
            if title.startswith("Rotation") or title.startswith("Translation"):
                ax.set_xscale("symlog" if title.startswith("Trans") else "linear")
            ax.legend(fontsize=6)
        fig.suptitle(f"Stress test, error versus injected magnitude\n{names[step]} (gray line = 2 um)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"stress_test_{step}.png", dpi=120)
        plt.close(fig)


# ------------------------------------------------------------------ summary
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def summary():
    df = pd.read_csv(CSV)
    df["floor_mean_um"] = df.floor_mean_um.fillna(np.nan)
    print("runs", len(df), "returned", int(df.returned.sum()))
    d = df[df.case != "baseline_repeat"]
    print("baseline_repeat err (um):", df[df.case == "baseline_repeat"][["core", "step", "err_mean_um", "err_max_um"]].to_string())
    print("xcheck max abs diff", df.xcheck_abs_diff.max())
    d = d.assign(accurate=d.err_mean_um <= ACCURATE_UM)
    print("\nconfidence x accurate (err_mean<=2um; no matrix counts as inaccurate)")
    print(pd.crosstab(d.confidence, d.accurate))
    print("\nhigh and err_mean>2:")
    print(d[(d.confidence == "high") & (d.err_mean_um > ACCURATE_UM)][["core", "step", "case", "magnitude", "seed", "n_inliers", "inlier_ratio", "err_mean_um", "floor_mean_um"]].to_string())
    print("\nlow and accurate:")
    print(d[(d.confidence == "low") & d.accurate][["core", "step", "case", "magnitude", "seed", "n_inliers", "inlier_ratio", "err_mean_um"]].to_string())
    print("\nno matrix:")
    print(d[~d.returned][["core", "step", "case", "magnitude", "seed"]].to_string())
    # affine
    a = d[d.case.str[0].isin(list("defg")) & d.aff_err_mean_um.notna()]
    print("\naffine vs similarity (cases d-g), mean of err_mean, by case")
    print(a.groupby(["step", "case"])[["err_mean_um", "aff_err_mean_um", "floor_mean_um"]].mean().round(2).to_string())
    print(a.groupby(["step", "case", "magnitude"])[["err_mean_um", "aff_err_mean_um", "floor_mean_um"]].mean().round(2).to_string())
    # detector
    sim = d[d.case.str[0].isin(list("abcdi")) & d.returned & d.succeeded.fillna(False)]
    deform = d[d.case.str[0].isin(list("efgh")) & d.returned]
    base = df[df.case == "baseline_repeat"]
    sim = pd.concat([sim, base])
    print("\ndetector: representable+accurate n=%d, deformed n=%d" % (len(sim), len(deform)))
    for step in d.step.unique():
        for met in ("inlier_rms_um", "quad_r2", "quad_r2_adj"):
            s0, s1 = sim[sim.step == step], deform[deform.step == step]
            print(step, met, "sim median %.3f [%.3f-%.3f]" % (s0[met].median(), s0[met].min(), s0[met].max()),
                  "| deformed median %.3f [%.3f-%.3f]" % (s1[met].median(), s1[met].min(), s1[met].max()),
                  "| AUC %.2f" % auc(s1[met].dropna(), s0[met].dropna()))
    # deformed only where floor >= 2 um (model cannot represent)
    big = deform[deform.floor_mean_um >= 1.0]
    print("\ndeformed with floor>=1um n=%d" % len(big))
    for step in d.step.unique():
        for met in ("inlier_rms_um", "quad_r2_adj"):
            s0, s1 = sim[sim.step == step], big[big.step == step]
            print(step, met, "AUC %.2f" % auc(s1[met].dropna(), s0[met].dropna()), "n_def", len(s1))
    print("\ndeformed by case/magnitude: detector values")
    print(deform.groupby(["step", "case", "magnitude"])[["floor_mean_um", "err_mean_um", "inlier_rms_um", "quad_r2_adj"]].mean().round(2).to_string())


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        config.dataset_dir()
        OUT.mkdir(parents=True, exist_ok=True)
        args = sys.argv[2:]
        cores = CORES
        if "--cores" in args:
            i = args.index("--cores")
            cores, args = args[i + 1:], args[:i]
        run(args or STEPS, cores)
    elif mode == "plot":
        plot()
    else:
        summary()
