"""Task 28: verify the deformation check on all four cores (analysis only, no app or engine changes).

Usage (from the repository root, UNUMLOCALIA_DIR set for real and stress):
       python validation/scripts/task28_verify.py real      # run_sample on the 4 cores and demo A, B -> validation_rerun_output/task28_real.csv
       python validation/scripts/task28_verify.py stress [core ...]   # subset of the Task 25 cases -> validation_rerun_output/stress_test_cores_0204.csv
       python validation/scripts/task28_verify.py analyze    # tables -> validation_rerun_output/task28_tables.txt (needs the CSVs of the other two commands and of task25)

The stress runs use the Task 25 generators and one_run(), then deformation.analyze_step on the same
registration result. "overall" is the step's match confidence capped by the check (pipeline.cap_confidence),
the per-step counterpart of the sample-level rule. Everything is deterministic (fixed seeds, cv2 RANSAC has none).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deformation  # noqa: E402
import engine  # noqa: E402
import evaluation as ev  # noqa: E402
import pipeline as pl  # noqa: E402
import prep  # noqa: E402
import samples as sm  # noqa: E402
import task25_stress_test as t25  # noqa: E402

OUT = ROOT / "validation_rerun_output"
REAL_CSV = OUT / "task28_real.csv"
STRESS_CSV = OUT / "stress_test_cores_0204.csv"
TABLES = OUT / "task28_tables.txt"
CORES = ["core01_hcc", "core02_non_tumour", "core03_tonsil", "core04_hca"]
DEMO = ROOT / "test_data" / "batch_demo"
R2_T = deformation.DEFORMATION_CONFIG["r2_threshold"]
LIN_T = deformation.DEFORMATION_CONFIG["linear_threshold_um"]

SUBSET = {("a_translation", 100, 0), ("b_rotation", 45, 0), ("b_rotation", 180, 0),
          ("c_mirror", "horizontal", 0), ("d_scale", 5, 0), ("e_stretch_x", 1, 0), ("e_stretch_x", 2, 0),
          ("f_shear", 1, 0), ("g_elastic", 5, 0), ("g_elastic", 5, 1), ("g_elastic", 10, 0),
          ("g_elastic", 10, 1), ("h_rot30_stretch3", 30, 0)}


def short(c):
    return deformation.SHORT_LABELS.get(c, "")


def real():
    rows = []
    samples = {s["sample_id"]: s for s in sm.group_samples(sm.discover_samples(t25.DATA))}
    for s in sm.group_samples(sm.discover_samples(DEMO)):
        samples[s["sample_id"]] = s
    for sid in CORES + ["demo_A", "demo_B"]:
        if sid not in samples:
            print("missing", sid, flush=True)
            continue
        t0 = time.time()
        res = pl.run_sample(samples[sid])
        print(sid, res["status"], res.get("confidence"), res.get("match_confidence"), res.get("deformation_check"),
              f"{time.time() - t0:.0f}s", flush=True)
        for d in res.get("diagnostics", []):
            if d.get("derived"):
                continue
            rows.append({"sample": sid, "step": d["step"], "status": res["status"],
                         "check": short(deformation.step_check(d)),
                         "linear_mean_um": d.get("linear_deformation_mean_um"),
                         "linear_max_um": d.get("linear_deformation_max_um"),
                         "quad_r2_adj": d.get("quad_r2_adj"), "inlier_rms_um": d.get("inlier_rms_um"),
                         "n_inliers": d.get("n_inliers"), "inlier_ratio": d.get("inlier_ratio"),
                         "match_confidence": d.get("confidence"),
                         "overall_confidence": res.get("confidence"),
                         "overall_match_confidence": res.get("match_confidence"),
                         "sample_check": short(res.get("deformation_check"))})
        pd.DataFrame(rows).to_csv(REAL_CSV, index=False)


def stress(cores):
    rows = []
    for core in cores:
        for step in t25.STEPS:
            fixed, fixed_um, moving, moving_um = t25.load(core, step)
            mask = prep.tissue_mask(moving, moving_um)
            grid = ev.sample_grid_in_mask(mask, moving_um, 1.0, step_px_mask=8)
            ys, xs = np.nonzero(mask)
            c = np.array([xs.mean(), ys.mean()])
            base = engine.register_pair(fixed, moving, fixed_um, moving_um, method=pl.METHOD, mirror_test=True)
            print(core, step, "baseline", base["confidence"], base["n_inliers"], flush=True)
            cases = [("baseline_repeat", 0, "", 0, [])] + [
                k for k in t25.build_cases(c, moving.shape) if (k[0], k[1], k[3]) in SUBSET]
            for case, mag, unit, seed, ops in cases:
                t0 = time.time()
                rec, reg, img, _P = t25.one_run(fixed, fixed_um, moving, moving_um, base["M"], grid, ops)
                if reg is not None:
                    ex = deformation.analyze_step(fixed, img, reg, fixed_um, moving_um, pl.METHOD)
                    chk = deformation.classify(ex["linear_deformation_mean_um"], ex["quad_r2_adj"])
                    rec.update(linear_mean_um=ex["linear_deformation_mean_um"],
                               linear_max_um=ex["linear_deformation_max_um"], quad_r2_adj_check=ex["quad_r2_adj"],
                               inlier_rms_check=ex["inlier_rms_um"], check=short(chk),
                               overall=pl.cap_confidence(reg["confidence"], chk) if chk else reg["confidence"])
                rec.update(core=core, step=step, case=case, magnitude=mag, unit=unit, seed=seed,
                           seconds=time.time() - t0)
                rows.append(rec)
                pd.DataFrame(rows).to_csv(STRESS_CSV, index=False)
                print(f"  {case} {mag} s{seed}: {rec['confidence']} err={rec.get('err_mean_um', float('nan')):.2f} "
                      f"floor={rec.get('floor_mean_um', float('nan')):.2f} lin={rec.get('linear_mean_um')} "
                      f"r2={rec.get('quad_r2_adj_check')} {rec.get('check')} {rec['seconds']:.1f}s", flush=True)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)))
                                     for v in r) + " |")
    return "\n".join(out)


def analyze():
    lines = []
    real_df = pd.read_csv(REAL_CSV)
    cores = real_df[real_df["sample"].isin(CORES)].copy()
    lines += ["## 1. Real cores, per step", md(cores[["sample", "step", "check", "linear_mean_um", "linear_max_um",
                                                      "quad_r2_adj", "inlier_rms_um", "match_confidence",
                                                      "overall_confidence"]])]
    marg = cores[["sample", "step"]].copy()
    marg["linear_mean / 1 um"] = cores["linear_mean_um"] / LIN_T
    marg["linear_max / 1 um"] = cores["linear_max_um"] / LIN_T
    marg["adj R2 / 0.05"] = cores["quad_r2_adj"] / R2_T
    lines += ["", "## Margin table (value / threshold; below 1 means not flagged)", md(marg)]
    flagged = cores[cores["check"] != "none detected"]
    lines += ["", "Flagged steps: " + ("none" if flagged.empty else "\n" + md(flagged))]

    demo = real_df[real_df["sample"].isin(["demo_A", "demo_B"])]
    c1 = cores[cores["sample"] == "core01_hcc"].set_index("step")
    lines += ["", "## 2. Generated files vs core01"]
    drows = []
    for _, r in demo.iterrows():
        if r["step"] not in c1.index:
            continue
        b = c1.loc[r["step"]]
        drows.append({"sample": r["sample"], "step": r["step"], "check": r["check"],
                      "linear_mean_um": r["linear_mean_um"],
                      "d_linear_vs_core01": r["linear_mean_um"] - b["linear_mean_um"],
                      "quad_r2_adj": r["quad_r2_adj"], "d_r2_vs_core01": r["quad_r2_adj"] - b["quad_r2_adj"],
                      "overall": r["overall_confidence"]})
    lines += [md(pd.DataFrame(drows)) if drows else "no demo rows"]

    old = pd.read_csv(t25.CSV)
    new = pd.read_csv(STRESS_CSV)
    new["acc"] = new["err_mean_um"] <= t25.ACCURATE_UM
    old["acc"] = old["err_mean_um"] <= t25.ACCURATE_UM

    def conf_table(df, col="confidence"):
        d = df[df["returned"] == True]  # noqa: E712
        return pd.crosstab(d[col], d["acc"].map({True: "accurate", False: "not accurate"}))

    allconf = pd.concat([old[["core", "step", "confidence", "returned", "acc"]],
                         new[new["core"].isin(["core02_non_tumour", "core04_hca"])][
                             ["core", "step", "confidence", "returned", "acc"]]])
    lines += ["", "## 4a. Match confidence vs actual error (accurate = mean error <= 2 um)",
              "Four cores: core01 and core03 full Task 25 runs (180) plus core02 and core04 subset runs.",
              conf_table(allconf).to_string(),
              "", "Subset runs only, all four cores (same cases everywhere):", conf_table(new).to_string(),
              "", "Overall (capped) confidence vs error, subset runs, all four cores:",
              conf_table(new[new["overall"].notna()], "overall").to_string()]
    ret = new[(new["returned"] == True) & new["check"].notna()].copy()  # noqa: E712
    ret["deformed"] = ret["floor_mean_um"] >= 1.0
    ret["fired"] = ret["check"] != "none detected"
    und, dfm = ret[~ret["deformed"]], ret[ret["deformed"]]
    cols = ["core", "step", "case", "magnitude", "err_mean_um", "floor_mean_um", "confidence",
            "linear_mean_um", "quad_r2_adj_check", "check"]
    lines += ["", "## 4b. Check behavior on subset runs, all four cores",
              f"Runs with a result and a check: {len(ret)}. Undeformed (floor < 1 um): {len(und)}. "
              f"Deformed (floor >= 1 um): {len(dfm)}.",
              f"False alarms (fired on undeformed): {int(und['fired'].sum())} of {len(und)}",
              f"Misses (not fired on deformed): {int((~dfm['fired']).sum())} of {len(dfm)}"]
    if int(und["fired"].sum()):
        lines += ["False alarms:", md(und[und["fired"]][cols])]
    if int((~dfm["fired"]).sum()):
        lines += ["Misses:", md(dfm[~dfm["fired"]][cols])]
    hi = ret[(ret["confidence"] == "high") & ret["fired"]]
    lines += [f"Runs rated high on match confidence but flagged: {len(hi)} of "
              f"{int((ret['confidence'] == 'high').sum())} high runs"]
    lines += ["", "Check outcome by truth (False = undeformed):",
              ret.groupby("deformed")["check"].value_counts().unstack(fill_value=0).to_string()]
    lines += [f"Runs where the engine returned no matrix: {int((new['returned'] != True).sum())}"]  # noqa: E712
    bad = ret[(ret["err_mean_um"] > 2) & ~ret["deformed"]]
    lines += [f"Registration failures (error > 2 um, floor < 1 um): {len(bad)}"]
    if len(bad):
        lines += [md(bad[cols + ["overall"]])]

    lines += ["", "## 4c. Threshold search (subset runs; deformed = floor >= 1 um)"]
    best = []
    for lt in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for rt in (0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 1e9):
            fire = (ret["linear_mean_um"] >= lt) | (ret["quad_r2_adj_check"] >= rt)
            fa = int((fire & ~ret["deformed"]).sum())
            ms = int((~fire & ret["deformed"]).sum())
            best.append({"linear_um": lt, "r2": rt if rt < 1e8 else "off", "false_alarms": fa, "misses": ms,
                         "errors": fa + ms})
    lines += [md(pd.DataFrame(best).sort_values(["errors", "false_alarms"]).head(10))]
    cur = [b for b in best if b["linear_um"] == LIN_T and b["r2"] == R2_T][0]
    lines += [f"Current thresholds (1.0 um, 0.05): false alarms {cur['false_alarms']}, misses {cur['misses']}"]
    for nm, d in (("undeformed", und), ("deformed", dfm)):
        lines += [f"{nm}: linear mean um min/median/max = {d['linear_mean_um'].min():.2f} / "
                  f"{d['linear_mean_um'].median():.2f} / {d['linear_mean_um'].max():.2f}; adj R2 = "
                  f"{d['quad_r2_adj_check'].min():.3f} / {d['quad_r2_adj_check'].median():.3f} / "
                  f"{d['quad_r2_adj_check'].max():.3f}"]
    lines += ["", "Per case type, subset runs:"]
    g = ret.groupby("case").agg(n=("check", "size"), floor_mean=("floor_mean_um", "mean"),
                                err_mean=("err_mean_um", "mean"), fired=("fired", "sum")).reset_index()
    lines += [md(g)]
    TABLES.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("real", "stress"):
        t25.config.dataset_dir()
        OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "real":
        real()
    elif cmd == "stress":
        stress(sys.argv[2:] or CORES)
    elif cmd == "analyze":
        analyze()
