# spatial-bio-registration

A free, open-source desktop app that registers images from different spatial biology modalities (H&E, COMET, Xenium, or other images) and exports the **transforms**, not registered images.

Automatic registration is the default. Landmark correction is planned for cases where automatic results are not good enough.

**Status: early release (alpha).** See [Validation so far](#validation-so-far) and [Limitations](#limitations) before relying on results.

## What it does

- Finds an automatic starting alignment between modalities (feature matching on nuclear signal: hematoxylin from H&E against DAPI from COMET; Xenium nuclei are drawn from the cell boundary outlines when there is no Xenium image).
- Reports a confidence level (high, medium, low) for every registration, with a plain-language reason. A second method (SIFT) is run as an independent check.
- Works on one sample or a batch. A sample is a set of two or more linked images of the same tissue.
- Lets you choose the reference modality (Xenium, COMET or H&E) and recomputes the transforms for it without re-registering.
- Gives you a review step: view overlays (blend, checkerboard, side by side, flicker), accept or reject each sample, then save results.

## What it exports

For each saved sample, in `results/<sample>/`:

| File | Content |
|---|---|
| `<modality>_imagealignment.csv` or `<modality>_to_<reference>_matrix.csv` | 3x3 transform in pixels |
| `<modality>_keypoints.csv` | matched points used for the fit (thinned) |
| `params.json` | transforms, pixel sizes, reference, method details, confidence |

The Xenium Explorer alignment file is written only when Xenium is the reference. A `batch_summary.csv` lists every sample with its confidence, warnings, review status and tool version.

Transforms are similarity transforms (rotation, uniform scale, translation). No registered images are written.

## Install

Requires Python TODO (version). Then:

```bash
git clone https://github.com/bnsreenu/spatial-bio-registration.git
cd spatial-bio-registration
pip install -r requirements.txt
python -m app
```

TODO: confirm the launch command and the OpenSlide installation note for Windows, macOS and Linux.

## Quick start

1. Open the app. In the sidebar, check the **Output folder** (sessions and results are saved there).
2. Add a sample (**Add sample**) or a folder of samples (**Add folder of samples**). Drop files or folders onto the cards, or use Browse. Check any pixel size warnings.
3. Choose the reference (Xenium is the default when present).
4. Click **Run auto-registration** (one sample) or **Auto-register all ready samples**.
5. Review the results, accept or reject each sample, and save the accepted results.

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

Tested on the UnumLocalia dataset (four tissue cores with H&E, COMET and Xenium outputs; same-section data):

- TODO: replace with the measured numbers from the validation notes: how close the automatic transforms are to the alignments provided with the dataset, per core and overall, with units and how it was measured. The provided alignments are least-squares fits to manually placed landmarks, so they are a reference, not ground truth.
- A deliberately wrong pairing (Xenium from a different core) is flagged as failed with very few matches.

## Limitations

- OME-Zarr input has been tested on real data (UnumLocalia). OME-TIFF, TIFF and SVS-style inputs have been tested only on files generated from that data. Real SVS and NDPI files and real Xenium morphology images have not been tested.
- Only same-section pairs have been tested. Serial sections and large tissue deformations have not.
- Only similarity transforms are computed. There is no deformable registration.
- Landmark correction is not available yet.
- Confidence thresholds were set from a small number of tests and should be treated as a guide.

## Background

This tool grew out of the registration step in the DigitalSreeni multimodal spatial biology video series (TODO: add link to video).

## License

TODO: choose a license (see note in the repository issues or the project owner).

## Author

Developed by Dr. Sreenivas Bhattiprolu (DigitalSreeni LLC).
