# Task 25: synthetic deformation stress test

Written as task25_notes.md because outputs/task24_notes.md already holds the Task 24 audit.

## Setup
- Cores: core01_hcc and core03_tonsil. Steps: H&E to COMET, and COMET to rendered Xenium (COMET fixed, render moving, as in the app). 180 runs (45 per core and step), about 4 to 7 s each, 18 min total.
- Images are the working images the app passes to engine.register_pair (1 um/px, so px = um). Script: src/task25_stress_test.py. Data: outputs/stress_test.csv. Figures: outputs/stress_test_he_to_comet.png, outputs/stress_test_comet_to_xenium.png.
- Injection is applied to the moving image about the tissue-mask centroid (warpAffine or remap, linear). The border is 0 for all three images. The H&E image the engine sees is the hematoxylin channel, where white background is about 0.01, so 0 is the white equivalent.
- Error = distance between M' f(G) and M_base G on the standard in-tissue grid (step 8 px of the tissue mask, 16000 points for core01 H&E). For affine injections this equals evaluation.transform_difference_um (cross-check max difference 2e-13 um). It is also defined for elastic fields.
- Baseline repeats were identical (error 0.00 in all 4 combinations). cv2 RANSAC has no seed argument. The elastic seeds are fixed (0, 1, 2). Everything else is deterministic.
- "Succeeded" = matrix returned and mean error <= 2 um. A matrix was returned in all 180 runs.
- Translation, rotation and mirror cases lose part of the field of view (same canvas); no case was excluded.
- Wrong pixel size: the image is unchanged and the declared moving pixel size is 2 or 5 percent too large (positive sign only).
- "Floor" = mean error of the best similarity (with or without reflection) fitted to the true displacement field, i.e. what no similarity can remove.

## What works (similarity-representable cases: a, b, c, d, i)
88 runs, all accurate. Mean error 0.04 to 0.47 um, worst point 0.87 um. This includes translation up to 500 um, all rotations including 90 to 270, both mirrors, scale -10 to +10 percent, and wrong declared pixel size 2 and 5 percent (mean up to 0.47 um at 5 percent in core01). Lowest inlier count among these: 122 (core03 COMET to Xenium); confidence was high or medium, never low.

## What fails
The error is almost entirely the floor, so the engine is not failing; a similarity cannot represent these cases.

| Case | Magnitude | Mean error (um, both cores and steps) | Floor (um) | Confidence seen |
|---|---|---|---|---|
| Stretch (x or 30 deg) | 1 % | 2.1 to 2.4 | 1.9 to 2.0 | medium (one high) |
| | 2 % | 4.7 to 5.6 | 3.8 to 3.9 | medium or low |
| | 5 % | 12.5 to 14.6 | 9.4 to 9.6 | low |
| Shear | 1 deg | 3.9 to 4.1 | 3.4 | low or medium |
| | 3 deg | 13.2 to 15.3 | 10.1 to 10.3 | low |
| Elastic (mean over seeds) | 2 um | 0.6 to 0.7 | 0.5 to 0.6 | medium or high |
| | 5 um | 1.4 to 1.6 | 1.3 to 1.6 | medium or high |
| | 10 um | 2.9 to 3.6 | 2.6 to 3.1 | low or medium |
| | 20 um | 6.4 to 7.0 | 5.2 to 6.2 | low |
| Rot 30 + 3 % stretch | | 7.2 to 9.5 | 5.6 to 6.0 | low |
| Rot 90 + 5 um elastic | | 1.5 to 1.7 | 1.3 to 1.6 | high (all four) |

The 2 um threshold is crossed at about 1 percent stretch, 1 deg shear and 7 um elastic displacement. The error is about 10 to 25 percent above the floor (worst point about 2 to 3 times the mean). Rotation does not interact with deformation here (rot 90 + elastic matches elastic alone).

## Confidence versus error (all 180 runs, accurate = mean error <= 2 um)

| Label | Accurate | Not accurate |
|---|---|---|
| high | 100 | 1 |
| medium | 18 | 21 |
| low | 0 | 36 |

- High with mean error above 2 um: 1 run (core01 H&E to COMET, 1 percent stretch along 30 deg, error 2.24 um, floor 1.87 um). Another 5 high runs (elastic 5 um) have errors of 1.4 to 1.8 um.
- Low but actually accurate: 0.
- Medium is mixed (18 accurate, 21 not). Low was always right to warn. High is reliable only up to about 2 um of unmodeled deformation: the label cannot see deformation below roughly 2 um mean error, and 5 um elastic displacement still gets high in 8 of 12 runs. Mean error by label (deformed runs): high 1.1 um, medium 2.8 um, low 8.7 um.

## How much an affine would help (cases d to g, raw matches)
Raw ratio-test matches are reachable without engine changes via auto_register.feature_based_similarity(return_points=True), the same matcher the engine calls, so I used it with cv2.estimateAffine2D (RANSAC, 3 px).
- Scale (d): similarity already exact (0.06 to 0.19 um). The affine is not better (0.11 to 0.34 um), slightly noisier.
- Stretch and shear (e, f): the affine removes about 97 percent of the error (mean 0.1 to 0.4 um versus 2 to 15 um), for both steps.
- Elastic (g): no help. Affine mean error is 109 percent of the similarity error (0.65 vs 0.69 um at 2 um, 7.4 vs 7.0 um at 20 um). Elastic fields of this smoothness are not affine.
- Combination h was not run through the affine experiment (spec: d to g only).

## Deformation detector (inlier residuals after the fit)
Groups: representable and accurate runs plus baseline repeats (n = 92) versus deformed runs e to h (n = 88).
- Quadratic-surface R squared (adjusted) of the inlier residual vectors: representable 0.00 to 0.04 (both steps); deformed median 0.15 to 0.18. AUC 1.00 (H&E to COMET), 0.99 (COMET to Xenium). Stretch and shear give 0.4 to 0.8, elastic 0.04 to 0.19, with overlap only for the weakest elastic runs (2 um, minimum 0.008 versus a representable maximum of 0.025 in COMET to Xenium).
- RMS of inlier residuals: representable 0.77 to 1.43 um, deformed 0.94 to 2.05 um. AUC 0.97 to 0.98, but it saturates near 2 um because RANSAC keeps only points within 3 px, so it cannot grade strong deformation.
- For deformed runs whose floor is at least 1 um (75 runs), both measures separate perfectly (AUC 0.99 to 1.00).
- In this test the R squared separates; a threshold of about 0.05 would have flagged every deformed run with a floor of at least 1 um and no representable run. It does not tell which kind of deformation.

## Not tested or limits
- Two cores, one random-field family (Gaussian sigma 80 um), one fixed random seed per elastic case. Thresholds above are from these runs and are not validated on real data or other tissue.
- Only the detector input from the inlier set was used; thresholds were not tuned or applied inside the app.
- Case i with negative sign, rotation about a point other than the tissue centroid, noise, intensity changes and partial overlap beyond translation and rotation of the same canvas were not tested.
- Cores 02 and 04 were not run.
