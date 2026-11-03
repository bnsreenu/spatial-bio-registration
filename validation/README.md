# Validation of version 0.2.0

This folder holds the analysis scripts, result tables and notes behind the numbers in the validation of version 0.2.0. The scripts import modules from the repository's `src/` folder, so they must be run from the repository root. They were written for this validation and are not part of the application.

## Contents

| File | What it contains | Produced by | Supports |
|---|---|---|---|
| `results/agreement_with_dataset_alignments.csv` | Long-format table (core, section, item, source, metric, value, unit, note): provided alignments, landmark scatter, automatic versus provided matrices, nucleus-based check, four cores | `scripts/agreement_with_dataset_alignments.py` | Agreement with the dataset alignments |
| `results/stress_test_core01_core03.csv` | One row per synthetic transform run on core01 and core03, both registration steps: confidence, inliers, error, best-similarity floor, affine and residual measures | `scripts/stress_test_known_perturbations.py` | Rigid stress test (also the deformation cases of the stress test) |
| `results/stress_test_core02_core04.csv` | Subset of the same cases on core02 and core04, with deformation check outcome | `scripts/deformation_check_verification.py stress` | Rigid stress test, deformation check |
| `results/deformation_check_real_steps.csv` | Deformation check values per step on the four cores (and demo samples A and B when present) | `scripts/deformation_check_verification.py real` | Deformation check |
| `results/deformation_check_summary_tables.txt` | Combined tables for the four cores (confidence versus error, false alarms, misses, margins) as plain text, not CSV | `scripts/deformation_check_verification.py analyze` | Deformation check |
| `notes/agreement_with_dataset_alignments.md` | Notes on the audit of the dataset's provided alignments | not a script output (text note) | Agreement with the dataset alignments |
| `notes/pixel_size_metadata_check.md` | Notes on pixel size metadata (OME-XML, multiscales, alignment-implied). Read-only check, no script in this folder | not a script output (text note) | Agreement with the dataset alignments |
| `notes/stress_test_known_perturbations.md` | Notes on the synthetic stress test | not a script output (text note) | Rigid stress test |
| `notes/deformation_check_implementation.md` | Notes on the deformation check and optional affine. The tests for it are in the repository's `tests/` and `app/` code, not here | not a script output (text note) | Deformation check |
| `notes/deformation_check_verification.md` | Notes on the four-core verification of the deformation check | not a script output (text note) | Deformation check |
| `scripts/agreement_with_dataset_alignments.py` | Audit of the dataset's provided alignments | | Agreement with the dataset alignments |
| `scripts/stress_test_known_perturbations.py` | Synthetic transform generator and stress-test runner, plot and summary | | Rigid stress test |
| `scripts/deformation_check_verification.py` | Four-core run, stress subset on core02 and core04, and tables | | Deformation check |
| `scripts/example_figure_images.py` | Draws images (modalities, working images, matches, overlays) for core01. Script only, the images are not included | | Example images |
| `scripts/config.py` | Helper: reads `UNUMLOCALIA_DIR`, core ids | | Used by the scripts above |
| `scripts/auto_baselines.py` | Helper: loads core images, older automatic baselines | | Used by `agreement_with_dataset_alignments.py` |
| `scripts/chain.py` | Helper: provided-matrix chaining | | Used by `agreement_with_dataset_alignments.py` |
| `scripts/nuclei_check.py` | Helper: nucleus-based check | | Used by `agreement_with_dataset_alignments.py` |
| `scripts/pipeline_compare.py` | Helper: grids and provided matrices | | Used by `agreement_with_dataset_alignments.py` |
| `README.md` | This file | | |

The notes were written during development and cite the original file names. The table in "File names in the notes" below maps them to the files in this folder.

## Running the scripts

Set `UNUMLOCALIA_DIR` to the folder that contains the core folders (`core01_hcc`, `core02_non_tumour`, and so on). A script that reads the dataset stops with a clear message if it is not set. Run from the repository root. Outputs go to `validation_rerun_output/` in the repository root, so the result tables in `results/` are not overwritten.

PowerShell:

```
$env:UNUMLOCALIA_DIR = "path\to\UnumLocalia"
python validation/scripts/agreement_with_dataset_alignments.py
python validation/scripts/stress_test_known_perturbations.py run
python validation/scripts/stress_test_known_perturbations.py plot
python validation/scripts/deformation_check_verification.py real
python validation/scripts/deformation_check_verification.py stress core02_non_tumour core04_hca
python validation/scripts/deformation_check_verification.py analyze
python validation/scripts/example_figure_images.py --core core01_hcc
```

Status of these commands: none of them has been run from this folder. The original versions of the scripts were run in the development folder, and the result tables here come from those runs. The copies here were checked only for syntax, for import, and for the message shown when `UNUMLOCALIA_DIR` is missing. The `analyze` command needs the outputs of `stress_test_known_perturbations.py run`, `deformation_check_verification.py real` and `deformation_check_verification.py stress` in `validation_rerun_output/`. `deformation_check_verification.py real` also looks for the demo samples in `test_data/batch_demo`, which is not included. When they are missing the script prints "missing" and continues with the four cores. `plot` and `example_figure_images.py` need matplotlib, which is not listed in `requirements.txt`.

## File names in the notes

The notes were written during development and cite the original file names.

| Name cited in the notes | File in this folder |
|---|---|
| `src/task24_ground_truth_audit.py` | `scripts/agreement_with_dataset_alignments.py` |
| `src/task25_stress_test.py` | `scripts/stress_test_known_perturbations.py` |
| `src/task28_verify.py` | `scripts/deformation_check_verification.py` |
| `src/task29_video2b_images.py` | `scripts/example_figure_images.py` |
| `outputs/ground_truth_audit.csv` | `results/agreement_with_dataset_alignments.csv` |
| `outputs/stress_test.csv` | `results/stress_test_core01_core03.csv` |
| `outputs/stress_test_cores_0204.csv` | `results/stress_test_core02_core04.csv` |
| `outputs/task28_real.csv` | `results/deformation_check_real_steps.csv` |
| `outputs/task28_tables.txt` | `results/deformation_check_summary_tables.txt` |
| `outputs/task24_notes.md` (task24_notes.md) | `notes/agreement_with_dataset_alignments.md` |
| `outputs/task24b_notes.md` (task24b_notes.md) | `notes/pixel_size_metadata_check.md` |
| `outputs/task25_notes.md` (task25_notes.md) | `notes/stress_test_known_perturbations.md` |
| `outputs/task27_notes.md` (task27_notes.md) | `notes/deformation_check_implementation.md` |
| `outputs/task28_notes.md` (task28_notes.md) | `notes/deformation_check_verification.md` |

## Dataset

The scripts read the public UnumLocalia dataset. The dataset is not included here. See the dataset paper: Duchini et al., bioRxiv 2026, doi 10.64898/2026.08.17.742355.

## Known limits

- Results are from four cores of one dataset.
- The deformation tests are synthetic.
- The scripts were run on Windows with Python 3.10.
- The scripts were written for this validation and are not part of the application.
- Repeated runs on the same machine gave identical matrices. Reproducibility on other machines has not been tested.
