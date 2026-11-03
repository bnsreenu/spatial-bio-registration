# Validation of version 0.2.0

This folder holds the analysis scripts, result tables and notes behind the numbers in the validation of version 0.2.0. The scripts import modules from the repository's `src/` folder, so they must be run from the repository root. They were written for this validation and are not part of the application.

## Contents

| File | What it contains | Produced by | Supports |
|---|---|---|---|
| `results/ground_truth_audit.csv` | Long-format table (core, section, item, source, metric, value, unit, note): provided alignments, landmark scatter, automatic versus provided matrices, nucleus-based check, four cores | `scripts/task24_ground_truth_audit.py` | Agreement with the dataset alignments |
| `results/stress_test.csv` | One row per synthetic transform run on core01 and core03, both registration steps: confidence, inliers, error, best-similarity floor, affine and residual measures | `scripts/task25_stress_test.py` | Rigid stress test (also the deformation cases of the stress test) |
| `results/stress_test_cores_0204.csv` | Subset of the same cases on core02 and core04, with deformation check outcome | `scripts/task28_verify.py stress` | Rigid stress test, deformation check |
| `results/task28_real.csv` | Deformation check values per step on the four cores (and demo samples A and B when present) | `scripts/task28_verify.py real` | Deformation check |
| `results/task28_tables.txt` | Combined tables for the four cores (confidence versus error, false alarms, misses, margins) as plain text, not CSV | `scripts/task28_verify.py analyze` | Deformation check |
| `notes/task24_notes.md` | Notes on the ground-truth audit | not a script output (text note) | Agreement with the dataset alignments |
| `notes/task24b_notes.md` | Notes on pixel size metadata (OME-XML, multiscales, alignment-implied). Read-only check, no script in this folder | not a script output (text note) | Agreement with the dataset alignments |
| `notes/task25_notes.md` | Notes on the synthetic stress test | not a script output (text note) | Rigid stress test |
| `notes/task27_notes.md` | Notes on the deformation check and optional affine. The tests for it are in the repository's `tests/` and `app/` code, not here | not a script output (text note) | Deformation check |
| `notes/task28_notes.md` | Notes on the four-core verification of the deformation check | not a script output (text note) | Deformation check |
| `scripts/task24_ground_truth_audit.py` | Ground-truth audit | | Agreement with the dataset alignments |
| `scripts/task25_stress_test.py` | Synthetic transform generator and stress-test runner, plot and summary | | Rigid stress test |
| `scripts/task28_verify.py` | Four-core run, stress subset on core02 and core04, and tables | | Deformation check |
| `scripts/task29_video2b_images.py` | Draws images (modalities, working images, matches, overlays) for core01. Script only, the images are not included | | Example images |
| `scripts/config.py` | Helper: reads `UNUMLOCALIA_DIR`, core ids | | Used by the scripts above |
| `scripts/auto_baselines.py` | Helper: loads core images, older automatic baselines | | Used by task24 |
| `scripts/chain.py` | Helper: provided-matrix chaining | | Used by task24 |
| `scripts/nuclei_check.py` | Helper: nucleus-based check | | Used by task24 |
| `scripts/pipeline_compare.py` | Helper: grids and provided matrices | | Used by task24 |
| `README.md` | This file | | |

The notes were copied unchanged. They name the original working-folder paths (for example `src/task25_stress_test.py` and `outputs/stress_test.csv`). In this folder those files are under `scripts/` and `results/`.

## Running the scripts

Set `UNUMLOCALIA_DIR` to the folder that contains the core folders (`core01_hcc`, `core02_non_tumour`, and so on). A script that reads the dataset stops with a clear message if it is not set. Run from the repository root. Outputs go to `validation_rerun_output/` in the repository root, so the result tables in `results/` are not overwritten.

PowerShell:

```
$env:UNUMLOCALIA_DIR = "path\to\UnumLocalia"
python validation/scripts/task24_ground_truth_audit.py
python validation/scripts/task25_stress_test.py run
python validation/scripts/task25_stress_test.py plot
python validation/scripts/task28_verify.py real
python validation/scripts/task28_verify.py stress core02_non_tumour core04_hca
python validation/scripts/task28_verify.py analyze
python validation/scripts/task29_video2b_images.py --core core01_hcc
```

Status of these commands: none of them has been run from this folder. The original versions of the scripts were run in the development folder, and the result tables here come from those runs. The copies here were checked only for syntax, for import, and for the message shown when `UNUMLOCALIA_DIR` is missing. The `analyze` command needs the outputs of `task25 run`, `task28 real` and `task28 stress` in `validation_rerun_output/`. `task28 real` also looks for the demo samples in `test_data/batch_demo`, which is not included. When they are missing the script prints "missing" and continues with the four cores. `plot` and `task29` need matplotlib, which is not listed in `requirements.txt`.

## Dataset

The scripts read the public UnumLocalia dataset. The dataset is not included here. See the dataset paper: Duchini et al., bioRxiv 2026, doi 10.64898/2026.08.17.742355.

## Known limits

- Results are from four cores of one dataset.
- The deformation tests are synthetic.
- The scripts were run on Windows with Python 3.10.
- The scripts were written for this validation and are not part of the application.
- Repeated runs on the same machine gave identical matrices. Reproducibility on other machines has not been tested.
