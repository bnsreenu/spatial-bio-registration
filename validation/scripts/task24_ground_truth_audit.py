"""Task 24: ground-truth audit of the dataset's provided alignments (analysis only).

Run from the repository root with UNUMLOCALIA_DIR set:
    python validation/scripts/task24_ground_truth_audit.py
Writes validation_rerun_output/ground_truth_audit.csv (long format) and prints the tables used in
notes/task24_notes.md. Pipeline and engine behavior are not changed.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import evaluation as ev
import nuclei_check as nc
import pipeline as pl
import pipeline_compare as pc
import samples as sm
from fit_transforms import apply_transform

DATA = config.dataset_dir(required=False)
OUT = ROOT / "validation_rerun_output"
CORES = ["core01_hcc", "core02_non_tumour", "core03_tonsil", "core04_hca"]
FIXED_UM = 0.2125
N_BOOT = 2000
N_PERM = 2000

ROWS = []


def add(core, section, item, source, metric, value, unit="", note=""):
    ROWS.append({"core_id": core, "section": section, "item": item, "source": source, "metric": metric,
                 "value": value, "unit": unit, "note": note})


# ----------------------------------------------------------------------------- part 1
def read_keypoints(core, modality):
    df = pd.read_csv(DATA / core / modality / f"keypoints_{modality}.csv")
    return df[["alignmentX", "alignmentY"]].to_numpy(float), df[["fixedX", "fixedY"]].to_numpy(float)


def fit_similarity(src, dst):
    n = len(src)
    a = np.zeros((2 * n, 4))
    a[0::2] = np.stack([src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)], axis=1)
    a[1::2] = np.stack([src[:, 1], src[:, 0], np.zeros(n), np.ones(n)], axis=1)
    p = np.linalg.lstsq(a, dst.reshape(-1), rcond=None)[0]
    return np.array([[p[0], -p[1], p[2]], [p[1], p[0], p[3]], [0, 0, 1.0]])


def fit_affine(src, dst):
    a = np.column_stack([src, np.ones(len(src))])
    px = np.linalg.lstsq(a, dst[:, 0], rcond=None)[0]
    py = np.linalg.lstsq(a, dst[:, 1], rcond=None)[0]
    return np.array([px, py, [0, 0, 1.0]])


def resid_um(m, src, dst):
    return (dst - apply_transform(m, src)) * FIXED_UM


def stats(r):
    d = np.hypot(r[:, 0], r[:, 1])
    return {"mean": d.mean(), "median": np.median(d), "rms": np.sqrt((d ** 2).mean()), "max": d.max()}


def quad_r2(pos, r, rng):
    """Quadratic surface (6 terms per component) fit of residual vectors against position."""
    p = (pos - pos.mean(0)) / (np.ptp(pos, axis=0).max() / 2)
    x, y = p[:, 0], p[:, 1]
    feats = np.column_stack([np.ones(len(p)), x, y, x * x, x * y, y * y])

    def r2(res):
        ss_res = ss_tot = 0.0
        for k in range(2):
            coef = np.linalg.lstsq(feats, res[:, k], rcond=None)[0]
            ss_res += ((res[:, k] - feats @ coef) ** 2).sum()
            ss_tot += ((res[:, k] - res[:, k].mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    obs = r2(r)
    n = len(p)
    adj = 1.0 - (1.0 - obs) * (n - 1) / (n - 6)
    null = np.array([r2(r[rng.permutation(n)]) for _ in range(N_PERM)])
    return obs, adj, float((null >= obs).mean()), float(np.percentile(null, 95))


def part1(core):
    rng = np.random.default_rng(0)
    for mod in ("he", "comet"):
        src, dst = read_keypoints(core, mod)
        m = pc.provided_matrix(DATA, core, mod)
        n = len(src)
        add(core, "1_keypoints", mod, "provided", "n_keypoints", n)
        r = resid_um(m, src, dst)
        for k, v in stats(r).items():
            add(core, "1_keypoints", mod, "provided_matrix", f"residual_{k}", v, "um")
        obs, adj, p, null95 = quad_r2(src, r, rng)
        add(core, "1_keypoints", mod, "provided_matrix", "quadratic_surface_R2", obs, "", "6 terms per component")
        add(core, "1_keypoints", mod, "provided_matrix", "quadratic_surface_R2_adjusted", adj)
        add(core, "1_keypoints", mod, "provided_matrix", "quadratic_surface_R2_perm_p", p, "", f"{N_PERM} shuffles of residual vectors")
        add(core, "1_keypoints", mod, "provided_matrix", "quadratic_surface_R2_null_95pct", null95)
        ms, ma = fit_similarity(src, dst), fit_affine(src, dst)
        rs, ra = stats(resid_um(ms, src, dst)), stats(resid_um(ma, src, dst))
        for k in ("mean", "median", "rms", "max"):
            add(core, "1_keypoints", mod, "similarity_refit", f"residual_{k}", rs[k], "um")
            add(core, "1_keypoints", mod, "affine_refit", f"residual_{k}", ra[k], "um")
        red = 100.0 * (1.0 - ra["rms"] / rs["rms"])
        # SSR ratio expected if the extra 2 affine parameters only fit noise
        noise = 100.0 * (1.0 - np.sqrt((2 * n - 6) / (2 * n - 4)))
        add(core, "1_keypoints", mod, "affine_vs_similarity", "rms_reduction_pct", red, "%")
        add(core, "1_keypoints", mod, "affine_vs_similarity", "rms_reduction_expected_from_noise_pct", noise, "%", "expected if the 2 extra parameters only fit random error")
        u, s, vt = np.linalg.svd(ma[:2, :2])
        add(core, "1_keypoints", mod, "affine_refit", "anisotropy_pct", 100.0 * (s[0] / s[1] - 1.0), "%", "ratio of linear-part singular values minus 1")
        add(core, "1_keypoints", mod, "affine_refit", "scale_x_over_y_pct", 100.0 * (np.hypot(*ma[:2, 0]) / np.hypot(*ma[:2, 1]) - 1.0), "%")
        a1, a2 = ma[:2, 0], ma[:2, 1]
        ang = np.degrees(np.arccos(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)))) - 90.0
        add(core, "1_keypoints", mod, "affine_refit", "shear_deg", ang, "deg", "angle between fitted axes minus 90")
        sc = np.sqrt(abs(np.linalg.det(ms[:2, :2])))
        add(core, "1_keypoints", mod, "similarity_refit", "scale", sc)
        add(core, "1_keypoints", mod, "provided_matrix", "scale", np.sqrt(abs(np.linalg.det(m[:2, :2]))))


# ----------------------------------------------------------------------------- part 2
def decompose(m):
    a = m[:2, :2]
    det = np.linalg.det(a)
    scale = np.sqrt(abs(det))
    rot = np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1]))
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    nonsim = np.linalg.norm(a - scale * np.sign(det) * np.array([[c, -s], [s, c]])) / scale
    return {"scale": scale, "rotation_deg": rot, "tx_px": m[0, 2], "ty_px": m[1, 2], "det_sign": np.sign(det),
            "non_similarity": nonsim}


def part2(core, mats, px_he, px_comet):
    steps = {"he_to_fixed": ("he", FIXED_UM, px_he), "comet_to_fixed": ("comet", FIXED_UM, px_comet),
             "he_to_comet": ("he_comet", px_comet, px_he)}
    for step, (key, target_um, meta_um) in steps.items():
        for source in ("provided", "automatic"):
            m = mats[source][key]
            d = decompose(m)
            implied = d["scale"] * target_um
            for k, v in d.items():
                add(core, "2_decomposition", step, source, k, v)
            add(core, "2_decomposition", step, source, "tx_um", d["tx_px"] * target_um, "um")
            add(core, "2_decomposition", step, source, "ty_um", d["ty_px"] * target_um, "um")
            add(core, "2_decomposition", step, source, "implied_pixel_size_um", implied, "um", f"scale x {target_um:g} um (target frame pixel)")
            add(core, "2_decomposition", step, source, "metadata_pixel_size_um", meta_um, "um")
            add(core, "2_decomposition", step, source, "implied_vs_metadata_pct", 100.0 * (implied / meta_um - 1.0), "%")


# ----------------------------------------------------------------------------- part 3
def part3(core, auto, grids, prov, t_prov, px_comet):
    for key, grid, label in (("comet", grids["comet"], "comet_to_fixed"), ("he", grids["he"], "he_to_fixed")):
        d = pc.difference_um(auto[key], prov[key], grid)
        for k in ("mean_um", "median_um", "max_um"):
            add(core, "3_auto_vs_provided", label, "automatic_vs_provided", k, d[k], "um", f"n_grid_points={d['n_points']}")
    t_auto = np.linalg.inv(auto["comet"]) @ auto["he"]
    mean_um, median_um, max_um, n = ev.transform_difference_um(t_auto, t_prov, grids["he"], comet_px_um=px_comet)
    for k, v in (("mean_um", mean_um), ("median_um", median_um), ("max_um", max_um)):
        add(core, "3_auto_vs_provided", "he_to_comet", "automatic_vs_provided", k, v, "um", f"n_grid_points={n}")


# ----------------------------------------------------------------------------- part 4
def xenium_nuclei_um(core):
    df = pd.read_parquet(DATA / core / "xenium" / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    g = df.groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
    return g.to_numpy(float)


def pair_indicators(pred_um, target_um):
    """Mutual nearest neighbors within 4 um. Returns per-nucleus indicators of the smaller set for
    matched within 1 um and within 2 um, and the pair distances."""
    ip, it, d = nc.mutual_nearest_neighbors(pred_um, target_um, nc.MATCH_RADIUS_UM)
    small_is_pred = len(pred_um) <= len(target_um)
    n_small = len(pred_um) if small_is_pred else len(target_um)
    idx = ip if small_is_pred else it
    out = {}
    for r in (1.0, 2.0):
        ind = np.zeros(n_small, dtype=bool)
        ind[idx[d <= r]] = True
        out[r] = ind
    return out, d, n_small


def boot_median_ci(d, rng, b=1000):
    if len(d) == 0:
        return np.nan, np.nan
    meds = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(b)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_diff_ci(a, b, rng):
    """Bootstrap CI of mean(a) - mean(b) over nuclei, exact via multinomial resampling of the 4 joint cells."""
    n = len(a)
    cells = np.array([(a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()], dtype=float)
    draws = rng.multinomial(n, cells / n, size=N_BOOT)
    diff = (draws[:, 1] - draws[:, 2]) / n
    return float((a.mean() - b.mean())), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def binom_ci(ind, rng):
    n = len(ind)
    draws = rng.binomial(n, ind.mean(), size=N_BOOT) / n
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def nuclei_step(core, step, pred_prov_um, pred_auto_um, target_um, rng):
    res = {}
    for src, pred in (("provided", pred_prov_um), ("automatic", pred_auto_um)):
        ind, d, n_small = pair_indicators(pred, target_um)
        res[src] = (ind, d)
        add(core, "4_nuclei", step, src, "n_moving_nuclei", len(pred))
        add(core, "4_nuclei", step, src, "n_target_nuclei", len(target_um))
        add(core, "4_nuclei", step, src, "n_pairs_within_4um", len(d))
        for r in (1.0, 2.0):
            lo, hi = binom_ci(ind[r], rng)
            add(core, "4_nuclei", step, src, f"fraction_matched_within_{int(r)}um", ind[r].mean(), "",
                f"of the smaller nuclei set (n={n_small}); 95% CI {lo:.4f} to {hi:.4f}")
            add(core, "4_nuclei", step, src, f"fraction_matched_within_{int(r)}um_ci_lo", lo)
            add(core, "4_nuclei", step, src, f"fraction_matched_within_{int(r)}um_ci_hi", hi)
        med = float(np.median(d)) if len(d) else np.nan
        lo, hi = boot_median_ci(d, rng)
        add(core, "4_nuclei", step, src, "median_distance_um", med, "um", f"mutual pairs within 4 um; 95% CI {lo:.3f} to {hi:.3f}")
        add(core, "4_nuclei", step, src, "median_distance_um_ci_lo", lo, "um")
        add(core, "4_nuclei", step, src, "median_distance_um_ci_hi", hi, "um")
    signs = []
    for r in (1.0, 2.0):
        diff, lo, hi = paired_diff_ci(res["automatic"][0][r], res["provided"][0][r], rng)
        add(core, "4_nuclei", step, "automatic_minus_provided", f"fraction_within_{int(r)}um_diff", diff, "",
            f"paired bootstrap 95% CI {lo:.4f} to {hi:.4f}")
        add(core, "4_nuclei", step, "automatic_minus_provided", f"fraction_within_{int(r)}um_diff_ci_lo", lo)
        add(core, "4_nuclei", step, "automatic_minus_provided", f"fraction_within_{int(r)}um_diff_ci_hi", hi)
        signs.append(1 if lo > 0 else (-1 if hi < 0 else 0))
    if 1 in signs and -1 in signs:
        verdict = "mixed"
    elif 1 in signs:
        verdict = "better"
    elif -1 in signs:
        verdict = "worse"
    else:
        verdict = "equivalent"
    add(core, "4_nuclei", step, "automatic_vs_provided", "verdict", verdict, "",
        "better/worse only if the paired 95% CI of the matched-fraction difference excludes 0 at 1 um or 2 um (and neither is opposite); equivalent = CI includes 0, i.e. not distinguishable")
    return verdict


def part4(core, images, mats, rng):
    he_ws = nc.detect_nuclei_centroids_working(images["hema"], images["he_um"])
    co_ws = nc.detect_nuclei_centroids_working(images["dapi_norm"], images["comet_um"])
    px_he, px_co = images["px_he"], images["px_comet"]
    he_nat = nc._to_native(he_ws, images["he_um"], px_he)
    co_nat = nc._to_native(co_ws, images["comet_um"], px_co)
    add(core, "4_nuclei", "detection", "he", "n_nuclei_detected", len(he_nat))
    add(core, "4_nuclei", "detection", "comet", "n_nuclei_detected", len(co_nat))
    # H&E -> COMET, compared in COMET micrometers
    tgt = co_nat * px_co
    pp = apply_transform(mats["provided"]["he_comet"], he_nat) * px_co
    pa = apply_transform(mats["automatic"]["he_comet"], he_nat) * px_co
    v1 = nuclei_step(core, "he_to_comet", pp, pa, tgt, rng)
    # COMET -> Xenium nuclei, compared in Xenium micrometers
    xen = xenium_nuclei_um(core)
    add(core, "4_nuclei", "detection", "xenium", "n_nuclei_detected", len(xen), "", "nucleus_boundaries.parquet polygon vertex mean per cell")
    pp = apply_transform(mats["provided"]["comet"], co_nat) * FIXED_UM
    pa = apply_transform(mats["automatic"]["comet"], co_nat) * FIXED_UM
    v2 = nuclei_step(core, "comet_to_xenium", pp, pa, xen, rng)
    return v1, v2


# ----------------------------------------------------------------------------- main
def main():
    samples = {s["sample_id"]: s for s in sm.group_samples(sm.discover_samples(DATA))}
    summary = []
    for core in CORES:
        t0 = time.time()
        print(f"\n===== {core} =====", flush=True)
        part1(core)
        sample = samples[core]
        runs = [pl.run_sample(sample) for _ in range(2)]
        for r in runs:
            assert r["status"] == "ok", (core, r["status"], r.get("error"))
        a, b = runs[0]["matrices"], runs[1]["matrices"]
        for key in ("comet", "he"):
            add(core, "3_auto_vs_provided", f"{key}_to_fixed", "run1_vs_run2", "max_abs_matrix_difference",
                float(np.abs(a[key] - b[key]).max()), "", f"array_equal={np.array_equal(a[key], b[key])}")
        add(core, "3_auto_vs_provided", "confidence", "automatic", "confidence", runs[0]["confidence"])
        grids = pc.core_grids(DATA, core)
        images = grids["images"]
        px_he, px_co = images["px_he"], images["px_comet"]
        prov = {m: pc.provided_matrix(DATA, core, m) for m in ("comet", "he")}
        t_prov = ev.target_he_to_comet(DATA, core)
        auto = a
        mats = {"provided": {"comet": prov["comet"], "he": prov["he"], "he_comet": t_prov},
                "automatic": {"comet": auto["comet"], "he": auto["he"],
                              "he_comet": np.linalg.inv(auto["comet"]) @ auto["he"]}}
        part2(core, mats, px_he, px_co)
        part3(core, auto, grids, prov, t_prov, px_co)
        v1, v2 = part4(core, images, mats, np.random.default_rng(1))
        summary.append((core, v1, v2))
        print(f"  done in {time.time() - t0:.0f} s; verdicts he_to_comet={v1}, comet_to_xenium={v2}", flush=True)
        pd.DataFrame(ROWS).to_csv(OUT / "ground_truth_audit.csv", index=False)
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT / "ground_truth_audit.csv", index=False)
    print(f"\nwrote {OUT / 'ground_truth_audit.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    config.dataset_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    main()
