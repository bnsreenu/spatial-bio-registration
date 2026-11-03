# Task 27: deformation check and optional affine

## What was done
- `src/deformation.py` (new): thresholds in one place (`DEFORMATION_CONFIG`: R squared 0.05, linear 1 um, affine RANSAC 3 px). Each step also fits a full affine (`cv2.estimateAffine2D`, RANSAC) from the raw matches, and stores the affine matrix, linear estimate (mean, max, um), adjusted quadratic R squared and inlier residual RMS.
- `src/pipeline.py`: the extra numbers are stored per step; sample result gets `deformation_check`, `match_confidence` (original rule), `matrices_similarity`, and the capped `confidence` (medium at most when the check is not "none detected").
- `src/reference.py`: `with_transform_choice` recomposes matrices from the stored chain (no re-registration); identities hold within 1e-9 with affine steps.
- `src/session.py`: choice and affine data saved and reloaded; params.json has `deformation_check` and per-layer `transform_type`, `matrix_similarity`, `matrix_affine`; batch_summary.csv has `deformation_check`, `linear_deformation_um`, `transform_type`.
- Review screen: "Deformation check" line, capped wording, checkbox "Use affine (allows stretch and shear)" (off by default, shown only when the check says stretch or shear is likely), reset to unreviewed after asking for an accepted sample, the Explorer line in the save dialog when a layer is affine. Help topic added, export topic updated.

## Results
- Similarity matrices of core01 and core03 are bit-identical to the pre-change baseline (`np.array_equal`).
- Undeformed core01 and core03: "none detected", confidence high. Linear estimate 0.10 to 0.59 um mean, adjusted R squared -0.005 to 0.029.
- Synthetic 2 percent stretch of the core01 H&E: flagged "stretch or shear likely" (linear 6.3 um), confidence low, and the affine option cut the error to under half of the similarity error (asserted in the test).

## Design choices
- The affine choice is per sample and applies to every step flagged as stretch or shear. The cap stays when affine is on.
- Layers on a chain that has no flagged step show "similarity". `matrix_affine` is null when it would equal the similarity matrix.
- Older sessions have no affine data and show "not available".
- The affine is fitted with a second `feature_based_similarity` call per step (about 1 to 2 s); the engine is unchanged.

## Tests run
`tests/test_task27_deformation.py` (16 tests, new), `tests.test_review_status`, `tests.test_version_and_log`, `test_app_smoke ... test_14`, and one offscreen launch. The full suite was not run.

## Not tested
Non-linear case end to end, the checkbox for a COMET to Xenium step, real scanner data, and loading an affine matrix in Xenium Explorer (unverified). Thresholds come from two cores. The 2 percent stretch run came out low confidence, so the "capped from high" wording is tested on a modified result, not on a natural run.

Screenshots: `outputs/task27_screens/`.
