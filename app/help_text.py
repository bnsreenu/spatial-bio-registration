"""Text of Help > Quick guide and Help > About, as plain data so it is easy
to edit. Each topic is (title, text). The text describes what the code does
(see src/samples.py, src/io_layers.py, src/engine.py and src/session.py);
change it together with those rules."""

from version import VERSION

APP_NAME = "Spatial registration"
LICENSE_NAME = "MIT License"

ABOUT_TEXT = (
    f"{APP_NAME}\n"
    f"Version {VERSION}\n\n"
    "This tool is free and open source.\n"
    "Developed by: Dr. Sreenivas Bhattiprolu (DigitalSreeni LLC)\n"
    f"License: {LICENSE_NAME}\n\n"
    "It computes alignments between H&E, COMET, Xenium and other images and exports transforms. "
    "It does not export registered images."
)

TOPICS = [
    ("What this tool does",
     "This tool computes how images of the same tissue line up: H&E, COMET, Xenium or other images. "
     "It finds matching structures in pairs of images, fits an alignment, and gives you a confidence "
     "level for each alignment. It exports the transforms (3 x 3 matrices, keypoints and a run record), "
     "not registered images. You can look at the result in the Review step, where the registered image "
     "is drawn in memory only."),

    ("Quick start in 5 steps",
     "1. Add data. Use \"Add folder of samples\" if your folders follow the naming rules (see \"How to "
     "organize files\"), or \"Add sample\" and drop or browse for each image on its card.\n"
     "2. Check each card. A green check means the input is readable. Fix anything shown in orange or "
     "red, for example enter a missing pixel size.\n"
     "3. Choose the reference under \"Register everything onto\". The default is Xenium if you have it.\n"
     "4. Press \"Run auto-registration\" (selected sample) or \"Auto-register all ready samples\".\n"
     "5. When a run finishes, the program opens \"2 Review and save\". Check the result (blend, "
     "checkerboard, side by side, flicker), accept or reject it, then save. Low-confidence samples are "
     "marked clearly and should be checked before use."),

    ("Supported inputs",
     "H&E: an RGB image. Formats: OME-Zarr (a folder ending in .ome.zarr), OME-TIFF, plain TIFF, and "
     "Aperio SVS or Hamamatsu NDPI when OpenSlide is installed. Give a file, or the zarr folder.\n\n"
     "COMET: a multichannel fluorescence image in the same formats. Give a file, or the zarr folder. "
     "Registration uses the one channel whose name contains \"DAPI\". If there is none, or more than "
     "one, that image cannot be used yet, because this version has no way to pick a channel.\n\n"
     "When you drop a file without choosing a card, it is placed by its content: an RGB image goes to "
     "H&E, a multichannel image to COMET, a Xenium folder to Xenium; anything else goes to Other.\n\n"
     "Xenium: the Xenium output folder (not a file). The tool recognizes it by files such as cells.csv, "
     "nucleus_boundaries.parquet and cell_boundaries.parquet. It uses a morphology image if the folder "
     "contains a file or folder whose name starts with \"morphology\". Without one (the case for the "
     "Xenium folders we have), nuclei are drawn from the cell nucleus outlines in "
     "nucleus_boundaries.parquet, and that works fine. If neither is present, the Xenium folder cannot "
     "be used. The transcripts file is never read.\n\n"
     "Other (optional): any additional image in the formats above."),

    ("How to organize files",
     "\"Add folder of samples\" looks at the folders directly inside the folder you choose. Each of "
     "them is one sample, and the folder name becomes the sample name.\n\n"
     "Inside a sample folder, the tool looks for subfolders named he, comet and xenium (capital "
     "letters do not matter).\n"
     "- In he and comet it takes the one .ome.zarr folder. If there is none, it takes the one .tif, "
     ".tiff, .svs or .ndpi file. The file name does not matter. If there are several candidates, it "
     "does not guess: the empty card says \"3 H&E candidates found: pick one\" and offers them in a menu.\n"
     "- xenium is linked as a folder.\n"
     "- If a subfolder is missing, the tool searches the sample folder (including its subfolders) for a "
     "file or folder whose name contains he, comet or xenium anywhere in the name. If several match at "
     "the same depth, you choose from a menu in the same way. Matches are marked \"proposed\" so you can "
     "check and change them. Put image files directly in a folder named he or comet to avoid this.\n"
     "- Image files placed directly in the folder you choose are not samples; put each sample in its own "
     "folder.\n\n"
     "Example:\n"
     "  my_data\\\n"
     "    core01\\\n"
     "      he\\core01_he.ome.zarr\n"
     "      comet\\core01_comet.ome.zarr\n"
     "      xenium\\ (the Xenium output folder)\n"
     "    core02\\\n"
     "      he\\slide_he.svs\n"
     "      comet\\comet.ome.tiff\n"
     "      xenium\\ (the Xenium output folder)"),

    ("Pixel size",
     "The pixel size (micrometers per pixel) is read from the file's metadata and is never guessed. "
     "Values outside 0.05 to 5 um per pixel are treated as implausible. If the pixel size is missing, "
     "the card says so, shows a field to enter it, and the sample cannot run until you do. Type the "
     "value from your scanner or your acquisition notes and press Use. An entered value is used for "
     "that input only.\n\n"
     "If the lower-resolution levels of a pyramid differ slightly from exact powers of two, the card "
     "shows a gray information note when the difference is above 0.1 percent. The exact figures are in "
     "the card's Details."),

    ("Choosing the reference",
     "The reference is the image that every other image is registered onto. The default is Xenium if "
     "it is loaded, otherwise COMET, otherwise H&E. Choose Xenium if you will load the result in Xenium "
     "Explorer. The matrices for the other choices are computed from the same registration by "
     "combining and inverting the steps, so the registration itself is not repeated.\n\n"
     "The Review step shows the reference image as the base and the chosen layer warped onto it. Use "
     "the Change link there to go back and pick another reference."),

    ("Reading confidence",
     "Each alignment step is rated from the matched points that support it:\n"
     "- High: at least 100 matching points and at least 70 percent of the candidate matches agree.\n"
     "- Low: fewer than 30 matching points, or fewer than 40 percent agree.\n"
     "- Medium: anything in between.\n"
     "A layer that goes through two steps (for example H&E to COMET to Xenium) gets the weakest of the "
     "steps that matter for the layer you are looking at. The overall sample confidence is shown "
     "separately. A deformation check that is not \"none detected\" caps the confidence at medium.\n\n"
     "For a low-confidence sample: check that the right images are linked (for example the right "
     "section and the right channel), and look at the overlay. Manual landmark correction is not "
          "included in this version."),

    ("Where files go: output folder, sessions, results",
     "The Data screen shows one Output folder (change it with \"Change...\"; the choice is remembered "
     "between launches). Inside it:\n"
     "- sessions\\ holds one <sample>_session.json per registered sample. They are written automatically "
     "after each sample, and again when you accept or reject a sample or change its reference. They hold "
     "the result, so you can close the program and open a session later (File > Open session).\n"
     "- results\\ is where saved results are offered by default. The save dialog lets you choose another "
     "place.\n\n"
     "Registering does not save any results. A sample is saved only when you use \"Save accepted "
     "results\", \"Save this sample's results\", or the save box of \"Accept all high-confidence\". The "
     "Review screen shows, for the selected sample, \"Not saved yet\" or "
     "\"Saved to <folder> at <time>\", and the time is stored in the session file. "
     "Changing the reference after a run does not re-register anything: the matrices are recomputed from "
     "the stored result, and the review status is kept."),

    ("Working through a batch",
     "1. Press \"Auto-register all ready samples\". The run uses the output folder shown under \"Output "
     "folder\" on the Data screen (the default is registration_projects in your Documents folder; the "
     "first time, you are asked once to confirm creating it). Results stay in the session, and a session "
     "file (<sample>_session.json) is saved in its sessions folder as each sample finishes. Nothing is "
     "written to results during the run. When at least one sample was registered, the program opens "
     "\"2 Review and save\" by itself; if every sample failed or the run was cancelled it stays on the "
     "Data screen and the footer says why. After a run, the run buttons become outline buttons and "
     "\"Review results\" is the main button. If a sample already has a result, you are asked before it is re-run, "
     "because re-running replaces the result and resets its review status.\n"
     "2. On the Review screen every sample starts as unreviewed. Samples with low confidence (or a failed "
     "registration) start as \"needs attention\" with a red dot.\n"
     "3. Look at each sample and press Accept or Reject. The status is saved in the session file.\n"
     "4. \"Accept all high-confidence\" accepts every unreviewed sample with high confidence after you "
     "confirm the count. They are recorded as \"accepted (bulk)\". Rejected samples are not changed. "
     "The confirmation has a box \"Also save their results to <output folder>/results\" (ticked by "
     "default): \"Change...\" picks another folder and \"Show files\" lists what will be written. If it is "
     "ticked, the results and batch_summary.csv are written at once (you are asked before anything is "
     "overwritten); if not, the samples are only marked as accepted.\n"
     "5. \"Save accepted results...\" lists the files and the destination first. Unreviewed and rejected "
     "samples are listed but not written unless you tick \"Include unreviewed and rejected samples\". "
     "Samples with no matrices (failed, blocked or not run) are listed as \"Not written\"; that option "
     "cannot write them. "
     "\"Save this sample's results...\" writes the selected sample whatever its status.\n"
     "6. Every save also writes batch_summary.csv in the destination folder (see \"What is saved\")."),

    ("What is saved",
     "Saving writes, for each registered image, a matrix file and a keypoints file (about 40 points), "
     "plus one params.json with the reference, pixel sizes, confidence and warnings, in a folder named "
     "after the sample. No registered images are written.\n\n"
     "If Xenium is the reference: {name}_imagealignment.csv (the Xenium Explorer alignment file, with "
     "the Xenium DAPI image as the fixed image) and {name}_keypoints.csv.\n"
     "If another image is the reference: {name}_to_{reference}_matrix.csv and "
     "{name}_to_{reference}_keypoints.csv. The Xenium Explorer file is not written in that case, and "
     "the save list says so before anything is written.\n\n"
     "batch_summary.csv, written in the destination folder every time, has one row per sample in the "
     "session, including samples that were blocked or never run (their confidence is blank and the "
     "review status says why, for example \"not run: needs at least two linked layers\"). Columns: "
     "sample, confidence, inliers_weakest_step, inlier_ratio_weakest_step, sift_diff_um_worst_step, "
     "deformation_check, linear_deformation_um, transform_type, "
     "warnings, reference, review_status, export_time and tool_version.\n"
     "- deformation_check: \"none detected\", \"stretch or shear likely\" or \"non-linear deformation "
     "possible\" (see \"Tissue deformation and the affine option\"); blank when not computed.\n"
     "- linear_deformation_um: the largest estimate over the steps of how much an affine fit would move "
     "the alignment across the tissue, in micrometers.\n"
     "- transform_type: \"affine\" when you chose the affine for any layer, otherwise \"similarity\".\n"
     "- inliers_weakest_step: the number of agreeing matching points in the alignment step that has the "
     "fewest (a sample has two or three steps).\n"
     "- inlier_ratio_weakest_step: the lowest share of matching points that agree, over the steps.\n"
     "- sift_diff_um_worst_step: the largest difference in micrometers between the main method and the "
     "second method, over the steps.\n"
     "Any of these is blank when not available. export_time is blank for samples that were not written.\n\n"
     "The save list is shown first, and existing files are never overwritten without asking. "
     "Sessions (File > Save session) store the inputs, settings, reference, review status and results "
     "in one file."),

    ("Tissue deformation and the affine option",
     "Each alignment step is fitted as a similarity: rotation, shift and one uniform scale. If the tissue "
     "is also stretched or sheared between the two images, a similarity cannot follow it. The program "
     "therefore also fits an affine (which allows stretch and shear) to the same matching points and "
     "reports a \"Deformation check\" on the confidence card:\n"
     "- None detected: an affine would change the alignment by less than 1 um across the tissue, and the "
     "leftover point errors show no smooth pattern.\n"
     "- Stretch or shear likely: an affine would change the alignment by 1 um or more. A checkbox \"Use "
     "affine (allows stretch and shear)\" appears (off by default). Turning it on uses the affine for "
     "those steps, recomputes the matrices and overlays, and records transform_type and both matrices "
     "in params.json. If the sample was accepted, you are asked first and it goes back to unreviewed.\n"
     "- Non-linear deformation possible: the leftover errors have a smooth pattern but an affine would "
     "not correct it. This is flagged only; it is not corrected.\n"
     "When the check is not \"none detected\", the confidence is capped at medium, even if the match "
     "itself is strong. Look at the overlay.\n\n"
     "The error scales with the field of view. A 1 percent stretch gave about 2 um of error across a "
     "1.2 mm core (measured). The figure of about 10 um across 6 mm is a calculation (the measured error "
     "scaled up to 6 mm), not a measurement. In our synthetic tests on two cores, 1 percent "
     "stretch or 1 degree shear gave about 2 to 4 um of error with a similarity, and 5 percent stretch or "
     "3 degrees shear gave 12 to 15 um. The thresholds (1 um and an R squared of 0.05) were set from "
     "two cores and checked on four cores, all with synthetic deformation; they have not been "
     "validated on real data.\n\n"
     "Xenium Explorer normally computes similarity alignments; whether it accepts a non-similarity "
     "matrix has not been verified."),

    ("Known limits",
     "- Manual landmark correction is not included in this version.\n"
     "- Only a similarity (rotation, shift, one uniform scale) is computed, plus the optional affine "
     "(allows stretch and shear). Nothing else is fitted.\n"
     "- Elastic (non-linear) deformation is flagged by the deformation check but not corrected.\n"
     "- The error from stretch or shear that is not modeled grows with the field of view.\n"
     "- The deformation check thresholds were set from synthetic deformations (stretch, shear, smooth "
     "elastic fields) on two cores and checked on four cores. They have not been tested on real "
     "deformed tissue.\n"
     "- Whether Xenium Explorer accepts a non-similarity (affine) matrix has not been verified.\n"
     "- Real SVS and NDPI files and real Xenium morphology images have not been tested. Only files "
     "generated from the UnumLocalia data were used (OME-TIFF, plain TIFF and an Aperio-style SVS "
     "file), and those test the readers, not the quirks of real scanner files.\n"
     "- Serial sections have not been tested. The tested images all show the same section.\n"
     "- Pixel size metadata can disagree with the image. In the UnumLocalia data, the provided H&E "
     "alignment implies a pixel size about 1.4 percent below the stated value (cause not determined). "
     "The core01 COMET OME-XML states 0.2125 um while the image pyramid says 0.28 um, and the alignment "
     "agrees with 0.28 um. For OME-Zarr the app uses the pyramid value. Check the pixel size shown on "
     "each card."),
]
