# Task 24 notes: ground-truth audit

Script: `src/task24_ground_truth_audit.py` (analysis only, not published). Data: `outputs/ground_truth_audit.csv` (764 rows: core, section, item, source, metric, value, unit, note). No app or src logic changed. No full test suite run. All four cores: pipeline status ok, confidence high.

Frames: fixed frame = Xenium Explorer DAPI pixels, 0.2125 um each. Metadata pixel sizes: H&E 0.203125 um, COMET 0.28 um.

## README table: Agreement with the dataset alignments

Automatic = current `pipeline.run_sample` on the zarr inputs, Xenium as reference. Differences are `transform_difference_um` on the standard in-tissue grid, automatic vs provided, mean / max in um. Two runs per core gave identical matrices (`np.array_equal` true for COMET and H&E, all cores).

| Core | COMET to fixed | H&E to fixed | H&E to COMET |
|---|---|---|---|
| core01_hcc | 0.55 / 0.74 | 0.32 / 0.53 | 0.28 / 0.32 |
| core02_non_tumour | 0.48 / 0.58 | 0.40 / 0.68 | 0.23 / 0.42 |
| core03_tonsil | 0.35 / 0.64 | 0.48 / 0.87 | 0.46 / 0.91 |
| core04_hca | 0.19 / 0.27 | 0.56 / 0.84 | 0.41 / 0.67 |

Independent nuclei check (blob_log nuclei, mutual nearest neighbors within 4 um). Fraction matched within 1 um / within 2 um (of the smaller nuclei set) and median distance, provided vs automatic:

| Core, step | Provided | Automatic | Verdict (see rule) |
|---|---|---|---|
| core01 H&E to COMET | 0.46 / 0.81 / 1.01 um | 0.42 / 0.79 / 1.09 um | worse |
| core01 COMET to Xenium | 0.35 / 0.71 / 1.21 um | 0.43 / 0.73 / 1.04 um | better |
| core02 H&E to COMET | 0.58 / 0.89 / 0.86 um | 0.60 / 0.89 / 0.83 um | better (1 um only) |
| core02 COMET to Xenium | 0.51 / 0.83 / 0.93 um | 0.60 / 0.85 / 0.80 um | better |
| core03 H&E to COMET | 0.48 / 0.85 / 0.99 um | 0.55 / 0.86 / 0.88 um | better |
| core03 COMET to Xenium | 0.37 / 0.76 / 1.23 um | 0.35 / 0.76 / 1.27 um | worse (1 um only) |
| core04 H&E to COMET | 0.48 / 0.85 / 1.01 um | 0.54 / 0.86 / 0.91 um | better |
| core04 COMET to Xenium | 0.60 / 0.88 / 0.82 um | 0.63 / 0.89 / 0.79 um | better (1 um only) |

Verdict rule: paired bootstrap (2000 draws over nuclei) 95% CI of the automatic-minus-provided matched fraction at 1 um and 2 um. Better or worse only if a CI excludes 0 and the other is not opposite; otherwise equivalent. No case came out equivalent or mixed. The effects are small: differences of 0.2 to 8 percentage points in matched fraction and at most 0.2 um in median distance. Statistical significance comes from thousands of nuclei per core, not from large effects. Detail and CIs are in the CSV.

Limits of this check: (a) both alignments are scored with the same nucleus detector, whose centroid error is not separated from alignment error, and the median distances of both (0.8 to 1.3 um) are probably dominated by it (not isolated); (b) for COMET to Xenium the automatic method was fit to the same Xenium nucleus polygons used as target here, so that check is not independent for the automatic side; (c) the bootstrap only reflects which nuclei were sampled.

## Provided keypoints (item 1)

Residual of keypoints under the provided matrix, um. R2 is a quadratic surface fit (6 terms per component) of the residual vectors against position; p is a permutation test (2000 shuffles). "Affine gain" is the RMS reduction of an affine refit over a similarity refit (the gain expected if the 2 extra parameters only fit noise is in brackets).

| Core, modality | n | mean | median | RMS | max | R2 (adj.) | p | Affine gain (noise) |
|---|---|---|---|---|---|---|---|---|
| core01 COMET | 40 | 0.59 | 0.53 | 0.66 | 1.41 | 0.28 (0.17) | 0.003 | 5.7% (1.3%) |
| core01 H&E | 20 | 0.60 | 0.51 | 0.70 | 1.76 | 0.16 (-0.14) | 0.85 | 1.1% (2.8%) |
| core02 COMET | 22 | 0.55 | 0.46 | 0.66 | 1.66 | 0.25 (0.02) | 0.41 | 7.7% (2.5%) |
| core02 H&E | 21 | 0.63 | 0.51 | 0.75 | 2.14 | 0.18 (-0.10) | 0.76 | 2.6% (2.7%) |
| core03 COMET | 30 | 0.43 | 0.26 | 0.59 | 1.84 | 0.11 (-0.08) | 0.77 | 3.5% (1.8%) |
| core03 H&E | 30 | 0.99 | 0.74 | 1.40 | 5.05 | 0.19 (0.02) | 0.39 | 0.5% (1.8%) |
| core04 COMET | 30 | 0.50 | 0.47 | 0.56 | 1.24 | 0.22 (0.06) | 0.21 | 0.5% (1.8%) |
| core04 H&E | 40 | 1.12 | 0.77 | 1.61 | 6.67 | 0.06 (-0.08) | 0.88 | 1.2% (1.3%) |

- The provided matrices are similarities; their residuals equal those of a least-squares similarity refit of the same keypoints to within 0.01 um RMS (they look like least-squares fits; how the keypoints were picked is not documented in the files).
- Structure: only core01 COMET shows a significant structured residual (p 0.003, R2 0.28, weak). The other seven show none. Affine refits give gains near the noise expectation, except COMET in core01, core02 and core03 (3.5% to 7.7% vs 1.3% to 2.5% expected), still small. Fitted anisotropy is at most 0.14% and shear at most 0.04 degrees in all cases. This dataset's own landmarks show no real stretch or shear.

## Scale, rotation, translation (item 2)

Implied pixel size = scale x target-frame pixel size (0.2125 um for "to fixed"; COMET 0.28 um for H&E to COMET, which gives the H&E pixel size). Provided / automatic, um, with percent difference from metadata:

| Step | Metadata | Provided (range over 4 cores) | Automatic (range) |
|---|---|---|---|
| COMET to fixed | 0.2800 | 0.27908 to 0.27921 (-0.33% to -0.28%) | 0.27913 to 0.27919 (-0.31% to -0.29%) |
| H&E to fixed | 0.20313 | 0.19999 to 0.20031 (-1.54% to -1.39%) | 0.20007 to 0.20021 (-1.51% to -1.43%) |
| H&E to COMET | 0.20313 | 0.20058 to 0.20090 (-1.25% to -1.10%) | 0.20067 to 0.20083 (-1.21% to -1.13%) |

- Rotation: COMET to fixed about -0.73 to -0.77 degrees, H&E to fixed about 0.52 to 0.59, H&E to COMET about 1.29 to 1.34; provided and automatic differ by at most 0.05 degrees. Translations agree within 0.6 um. All matrices are proper similarities (determinant positive, non-similarity part below 1e-15).
- Provided and automatic agree with each other on scale, so the H&E scale deficit relative to the 0.20313 metadata value (about 1.4%) is a property of the images and metadata, not of our pipeline. The cause is not isolated here. COMET differs from its metadata by about 0.3% in both.

## How reliable are the dataset alignments as ground truth?

They are a good similarity alignment but not exact. At their own landmarks, the provided matrices leave a mean error of 0.4 to 1.1 um (RMS 0.56 to 0.66 um for COMET, 0.70 to 1.61 um for H&E) and a worst case of 1.2 to 6.7 um, with H&E in core03 and core04 the loosest. The residuals are mostly random, apart from a weak structured component in core01 COMET, and the landmarks give no evidence of stretch or shear. The H&E to COMET target is built from two such matrices, so its uncertainty combines both. The automatic alignments differ from the provided ones by mean 0.19 to 0.56 um and max up to 0.91 um, which is smaller than the provided alignments' own landmark scatter, so differences of that size cannot be called errors of either side. The independent nuclei check does not rank them decisively: in 6 of 8 core-step cases the automatic alignment scores slightly better, in 2 slightly worse, always by small amounts and under a detection noise floor of about 1 um that we did not separate. Reasonable claim: the automatic alignments agree with the dataset alignments to within the dataset's own landmark uncertainty (below about 1 um); not a claim that either is correct to better than that.

## Not tested

Only four cores from one dataset. No check against landmarks the researchers did not use. No manual verification of any nucleus match.
