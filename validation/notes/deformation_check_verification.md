# Task 28: deformation check verification on four cores

Analysis only. No app, src logic or thresholds changed. Script: `src/task28_verify.py`. Raw tables: `outputs/task28_tables.txt`, `outputs/task28_real.csv`, `outputs/stress_test_cores_0204.csv`.

## 1. Real cores (run_sample, zarr inputs, Xenium reference)
All four cores: check "none detected" on both steps, match and overall confidence high. No step flagged.

| core | step | linear mean / max (um) | adj R2 | inlier RMS (um) | linear/1 um | R2/0.05 |
|---|---|---|---|---|---|---|
| core01 | H&E to COMET | 0.107 / 0.251 | 0.012 | 1.029 | 0.11 | 0.24 |
| core01 | COMET to Xenium | 0.102 / 0.234 | 0.005 | 1.181 | 0.10 | 0.10 |
| core02 | H&E to COMET | 0.036 / 0.087 | 0.020 | 0.826 | 0.04 | 0.40 |
| core02 | COMET to Xenium | 0.108 / 0.208 | 0.021 | 0.931 | 0.11 | 0.43 |
| core03 | H&E to COMET | 0.167 / 0.370 | 0.029 | 0.804 | 0.17 | 0.57 |
| core03 | COMET to Xenium | 0.589 / 1.540 | -0.005 | 1.312 | 0.59 | n/a |
| core04 | H&E to COMET | 0.320 / 0.930 | 0.022 | 1.042 | 0.32 | 0.45 |
| core04 | COMET to Xenium | 0.254 / 0.573 | 0.025 | 1.134 | 0.25 | 0.51 |

Tightest margins: core03 COMET to Xenium linear (0.59 of threshold, only 253 inliers) and core03 H&E to COMET R2 (0.57).

## 2. Generated files (batch_demo A and B vs core01)
Both "none detected", overall high. COMET to Xenium identical to core01 (diff 0.000). H&E to COMET: A differs by -0.009 um linear and -0.002 R2; B by +0.091 um and -0.003 R2. Small, same order as the run-to-run floor of generated-file conversion.

## 3. Stress subset
13 cases plus a repeat, both steps, all four cores (112 runs, seeds fixed, cv2 RANSAC has no seed control but runs are deterministic). Saved as `outputs/stress_test_cores_0204.csv`. Deviation: the file holds core01 and core03 too, because the Task 25 CSV has no check outcome, so those two cores were rerun on the same subset. Overall confidence in this CSV is per step (match confidence capped by the check).

By case: translation, rotation, mirror, scale, repeat (56 runs): check fired 0 times. Stretch x (16), shear (8), rotation 30 plus 3% stretch (8): fired 32 of 32, always labelled stretch or shear. Elastic (32): fired 31 of 32; the one miss is below.

## 4. Combined results (accurate = mean error <= 2 um)
Match confidence vs error, core01 and core03 full Task 25 runs plus core02 and core04 subset (n = 236):

| confidence | accurate | not accurate |
|---|---|---|
| high | 138 | 4 |
| medium | 18 | 36 |
| low | 0 | 40 |

Subset only, all four cores (like for like): high 62 / 3, medium 5 / 28, low 0 / 14.
Overall (capped) confidence, subset: high 46 / 0, medium 21 / 31, low 0 / 14. Capping removed all inaccurate runs from "high".

Check performance (deformed = floor >= 1 um; 51 undeformed, 61 deformed):
- False alarms: 3 of 51. All are 5 um elastic runs with floor 0.76 to 0.98 um (core02 and core03 COMET to Xenium, core04 H&E to COMET), labelled non-linear. They are small real deformations, so arguably not false. On pure undeformed runs (translation, rotation, mirror, scale, repeat) there were 0 false alarms in 48.
- Misses: 1 of 61: core03 COMET to Xenium, elastic 5 um seed 0 (floor 1.35 um, error 1.43 um, linear 0.36, R2 0.025, rated medium). Core03 COMET to Xenium has few inliers (253), so R2 is noisy.
- Runs rated high on match confidence but flagged: 19 of 65 high runs (subset). All are stretch 1% or elastic 5 um cases with errors of roughly 0.8 to 2.4 um. Three high-rated runs were inaccurate (error > 2 um): stretch 1% on core02 (both steps) and core04 COMET to Xenium. All three were flagged.
- Wrong label: elastic 10 um is often labelled "stretch or shear likely" (linear 1.1 to 4.9 um) although an affine cannot remove most of that error (floor 1.5 to 3.4 um).

Threshold search (grid, subset): current (1.0 um, 0.05) gives 3 false alarms, 1 miss. R2 0.08 with linear 0.75 gives 2 and 1; R2 0.08 with linear 1.0 gives 2 and 2. The differences are one run out of 112, not meaningful. Undeformed injected runs reach R2 0.044 (core04), so lowering R2 would risk false alarms. No threshold change suggested.

## Summary
- The check behaves on all four cores: no flags on real data, no false alarms on 48 undeformed injected runs, and all linear stretch or shear cases of 1% or 1 degree detected with the right label.
- It fails or is weak for: small elastic deformation (about 5 um peak) on low-inlier pairs (one miss), and the label for strong elastic deformation (10 um), which is often called stretch or shear.
- Margins on real data are at least 1.7x for linear and 1.7x for R2 (core03 H&E to COMET). Core03 COMET to Xenium linear (0.59 um mean, 1.54 um max) is the closest to the limit.
- Untested: real deformed tissue (all deformation is synthetic and the thresholds came from the same cores), other scanners and tissue types, other fields of view, and fewer-inlier pairs. Elastic cases used only two seeds per magnitude.
- Scratch logs `outputs/_task28_real.log` and `outputs/_task28_stress.log` were left in place.
