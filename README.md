# spatial-bio-registration

A free, open-source desktop app that registers images from different spatial biology modalities (H&E, COMET, Xenium, or other images) and exports the **transforms**, not registered images.

Automatic registration is the default. Manual landmark correction is not included in this version.

**Status: early release (alpha), version 0.2.0.** See [Validation so far](#validation-so-far) and [Limits](#limits) before relying on results.

## What it does

- Finds an automatic starting alignment between modalities (feature matching on nuclear signal: hematoxylin from H&E against DAPI from COMET; Xenium nuclei are drawn from the nucleus outlines (nucleus_boundaries.parquet) when there is no Xenium image).
- Reports a confidence level (high, medium, low) for every registration, with a plain-language reason. A second method (SIFT) is run as an independent check.
- Checks each registration step for tissue deformation (stretch, shear or non-linear deformation) by also fitting an affine transform to the same matches. When the check flags something, the confidence is capped at medium and the Review screen says why. If stretch or shear is likely, an optional checkbox "Use affine (allows stretch and shear)" (off by default) switches that sample to the affine transform.
- Works on one sample or a batch. A sample is a set of two or more linked images of the same tissue.
- Lets you choose the reference modality (Xenium, COMET or H&E) and recomputes the transforms for it without re-registering.
- Gives you a review step: view overlays (blend, checkerboard, side by side, flicker), accept or reject each sample, then save results.

## What it exports

For each saved sample, in `results/<sample>/`:

| File | Content |
|---|---|
| `<modality>_imagealignment.csv` or `<modality>_to_<reference>_matrix.csv` | 3x3 transform in pixels |
| `<modality>_keypoints.csv` | matched points used for the fit (thinned) |
| `params.json` | transforms, pixel sizes, reference, method details, confidence, deformation check, and per layer `transform_type` ("similarity" or "affine") with both matrices |

The Xenium Explorer alignment file is written only when Xenium is the reference. A `batch_summary.csv` lists every sample with its confidence, warnings, review status and tool version. It also has the columns `deformation_check` ("none detected", "stretch or shear likely" or "non-linear deformation possible"), `linear_deformation_um` (how much an affine fit would move the alignment across the tissue, in um) and `transform_type`.

Transforms are similarity transforms (rotation, uniform scale, translation) by default; the affine (adds stretch and shear) is used only when you choose it. No registered images are written. The save dialog notes that Xenium Explorer normally computes similarity alignments and that whether it accepts a non-similarity matrix has not been verified.

## Install

Requires Python 3.10 (tested). Then:

```bash
git clone https://github.com/bnsreenu/spatial-bio-registration.git
cd spatial-bio-registration
pip install -r requirements.txt
python -m app
```

Tested on Windows with Python 3.10; other platforms have not been tested. SVS and NDPI files are read with openslide-python and openslide-bin, which requirements.txt installs.

## Quick start

1. Open the app. In the sidebar, check the **Output folder** (sessions and results are saved there).
2. Add a sample (**Add sample**) or a folder of samples (**Add folder of samples**). Drop files or folders onto the cards, or use Browse. Check any pixel size warnings.
3. Choose the reference (Xenium is the default when present).
4. Click **Run auto-registration** (one sample) or **Auto-register all ready samples**. When a run finishes with at least one successful sample, the app opens **2 Review and save**.
5. Review the results (check the Deformation check line), accept or reject each sample, and save the accepted results with **Save accepted results...**. **Accept all high-confidence** can also save at the same time.

Registering does not write results. The **Output folder** holds `sessions/` (one session file per sample, written automatically so you can reopen work later) and `results/` (the saved matrices, keypoints, `params.json` and `batch_summary.csv`).

Open **Help > Quick guide** in the app for details.

## Input layout for folders of samples

To propose samples automatically from a folder, use one folder per sample with these subfolders:

```
my_samples/
  sample_01/
    he/       one .ome.zarr folder, or one .tif/.tiff/.svs/.ndpi file
    comet/    one .ome.zarr folder, or one .tif/.tiff/.svs/.ndpi file
    xenium/   the Xenium output folder
  sample_02/
    he/
    comet/
    xenium/
```

Pixel sizes are read from the files and never guessed. If a pixel size is missing, you are asked to enter it.

For Xenium, the app uses `nucleus_boundaries.parquet` from the output folder when there is no morphology image.

## Validation so far

Tested on the UnumLocalia dataset: four tissue cores (core01 to core04) with H&E, COMET and Xenium outputs; same-section data. All numbers are in micrometers (um). Frames: the fixed frame is the Xenium Explorer DAPI pixel grid (0.2125 um per pixel).

**Agreement with the alignments provided with the dataset.** "Automatic" is the app's registration run on the OME-Zarr inputs with Xenium as reference. Difference = mean / max distance in um between the automatic and the provided transform, over a grid of points inside the tissue:

| Core | COMET to fixed | H&E to fixed | H&E to COMET |
|---|---|---|---|
| core01_hcc | 0.55 / 0.74 | 0.32 / 0.53 | 0.28 / 0.32 |
| core02_non_tumour | 0.48 / 0.58 | 0.40 / 0.68 | 0.23 / 0.42 |
| core03_tonsil | 0.35 / 0.64 | 0.48 / 0.87 | 0.46 / 0.91 |
| core04_hca | 0.19 / 0.27 | 0.56 / 0.84 | 0.41 / 0.67 |

Across the four cores the mean differences are 0.19 to 0.56 um and the largest is 0.91 um. Running the registration twice gave identical matrices.

**How exact the provided alignments are.** The provided alignments are similarity fits to manually placed landmarks, so they are a reference, not ground truth. At their own landmarks they leave a mean error of 0.43 to 1.12 um and a worst case of 1.24 to 6.67 um (H&E in core03 and core04 is the loosest). The automatic-versus-provided differences above are of similar size to this scatter or smaller, so differences of that size cannot be attributed to either side. These landmarks show no sign of stretch or shear. In a nucleus-based check (nuclei detected in each image, matched within 4 um), the automatic alignment scored slightly better in 6 of 8 core and step cases and slightly worse in 2, with small effects (up to 0.2 um in median distance) under a detection noise level of about 1 um that was not separated from alignment error. For COMET to Xenium the automatic fit used the same Xenium nucleus outlines as the check, so that check is not independent. In short, the automatic alignments agree with the dataset alignments to within the scatter of the dataset's own landmarks.

**Synthetic stress tests** (a known transform applied to one image of core01 and core03, then registered again; 180 runs; error is the mean distance in um to the known result, accurate means 2 um or less):
- Accurate: translation up to 500 um, rotation at all angles tested (1 to 270 degrees), mirroring, uniform scale of -10 to +10 percent, and a wrong declared pixel size of 2 and 5 percent (declared too large). All 88 runs were accurate, mean error 0.04 to 0.47 um, worst point 0.87 um. A later check on core02 and core04 (translation 100 um, rotation 45 and 180 degrees, mirror, scale +5 percent) were not flagged for deformation.
- Error grows with unmodeled deformation: stretch 1, 2 and 5 percent gave mean error about 2.1 to 2.4, 4.7 to 5.6 and 12.5 to 14.6 um; shear of 1 and 3 degrees gave 3.9 to 4.1 and 13.2 to 15.3 um; smooth elastic deformation with peak displacement 5, 10 and 20 um gave 1.4 to 1.6, 2.9 to 3.6 and 6.4 to 7.0 um. The 2 um mark is crossed at about 1 percent stretch, 1 degree shear and 7 um elastic displacement.
- An affine fit removed about 97 percent of the error for stretch and shear and did not help for elastic deformation.
- Confidence labels versus error (180 runs): rated high, 100 accurate and 1 not; medium, 18 and 21; low, 0 and 36. A low rating was always right to warn. A high rating can still hide deformation of up to about 2 um mean error.
- Deformation check (four cores, 112 runs, thresholds: adjusted R squared 0.05, linear estimate 1 um): none detected on all eight real core-and-step registrations, and not triggered in 48 of 48 runs with translation, rotation, mirror or scale. Stretch, shear and rotation-plus-stretch runs were flagged in 32 of 32; elastic runs in 31 of 32. Three runs with a small elastic deformation (5 um peak) were flagged although their unavoidable similarity error was under 1 um. With the cap at medium, all 46 runs that ended up rated high were accurate. Elastic deformation of 10 um is often labelled "stretch or shear likely" even though an affine would not correct most of it.
- All deformation tests are synthetic. The check thresholds were set from the two-core stress test and then checked on four cores; they have not been validated on real deformed tissue.

## Limits

- Manual landmark correction is not included in this version.
- Only similarity and optional affine transforms are computed. There is no deformable registration.
- Elastic (non-linear) deformation is flagged by the deformation check but not corrected.
- The error from unmodeled stretch or shear grows with the field of view.
- The deformation check thresholds were set from synthetic deformations on two cores and checked on four cores (synthetic only). They have not been tested on real deformed tissue.
- Whether Xenium Explorer accepts a non-similarity (affine) matrix has not been verified.
- OME-Zarr input has been tested on real data (UnumLocalia). OME-TIFF, plain TIFF and SVS-style inputs have been tested only on files generated from that data, which test the readers and not the quirks of real files. Real SVS and NDPI files and real Xenium morphology images have not been tested.
- Serial sections have not been tested. All tested pairs show the same section.
- Pixel size metadata can disagree with the image. In UnumLocalia the provided H&E alignment implies a pixel size about 1.4 percent below the stated value (cause not determined), and the core01 COMET OME-XML says 0.2125 um while the image pyramid says 0.28 um (the alignment agrees with 0.28 um). The app uses the pyramid value for OME-Zarr.
- Confidence thresholds were set from a small number of tests and should be treated as a guide.

## Background

This tool grew out of the registration step in the DigitalSreeni multimodal spatial biology video series.

## License

MIT License (see LICENSE).

This application depends on PyQt5, which is licensed under GPL v3 or a commercial license; if you redistribute the app bundled with PyQt5 (for example as a packaged executable), PyQt5's terms apply.

## Author

Developed by Dr. Sreenivas Bhattiprolu (DigitalSreeni LLC).
