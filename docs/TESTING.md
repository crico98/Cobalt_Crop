# Testing and validation

## Automated unit tests

Run from the repository root with the project virtual environment active:

~~~powershell
python -m unittest discover -s tests -v
~~~

The current suite covers:

- Explicit angle_index mapping for Isometric, Top, and Face.
- Rejection of missing, malformed, unsupported, and duplicate manifest records.
- Stable Isometric → Top → Face grouping and order preservation within each group.
- Detection-region settings for all three views in landscape and portrait orientations.
- Centered-region extraction and translation of detection coordinates to full-image coordinates.
- Square crop geometry, image-edge padding, and Isometric target placement.
- Detection candidate selection and confidence-weighted box fusion.
- EXIF orientation normalization.
- Derived training-box expansion and quarter-turn orientation handling.
- Crop-test batch preparation behavior.

The GitHub Actions workflow runs the unit-test suite on pushes to configured branches and on pull requests. It does not download the private/local model checkpoints or run full image inference.

## What unit tests do not prove

A passing unit-test suite does not establish that:

- The local Face, Top, and Isometric model checkpoints are present or load successfully.
- The models detect every real-world cutter correctly.
- A full mixed-view batch completes successfully.
- Every resulting crop is visually acceptable.
- Runtime behavior is correct on every collaborator's GPU/driver configuration.

Those require local inference and visual review with the approved model files and representative source images.

## Current visual-validation record

The implementation plan records the following user-reviewed results:

- **Top:** crop target coverage 0.70 was accepted after a previous 0.80 framing clipped part of the cutter.
- **Isometric:** crop target coverage 0.80 and diamond-table center at 70% of output height were accepted after comparing alternative vertical placements.
- **Top / Isometric detector selection:** the user reviewed a 106-image comparison (70 Isometric and 36 Top). WBF produced selected crops for 102 images. Four Top images had no detections in any compared method and retained the existing no-crop behavior. The user preferred the WBF results over baseline and NMS.
- **Face:** one representative Face image, DSC_0560.JPG, was visually reviewed in the current validation session and the crop was accepted. This is a single-image acceptance, not a full-batch validation.

These records are specific to the current baseline. They do not authorize changing model weights, crop geometry, training annotations, or starting BitHub integration.

## Pending validation

The next planned validation is a full-photo batch using the explicit JSON manifest contract. The user will request that manifest after the developer handoff documentation is complete.

For the manifest contract and example records, see the root-level manifest_template.json and README.md.

## Suggested review checklist for a new batch

1. Confirm the manifest contains one record for every uploaded photo.
2. Confirm every angle_index is correct: 0 Isometric, 1 Top, 2 Face.
3. Confirm the upload is rejected if a photo is missing from the manifest or appears more than once.
4. Confirm processing order is Isometric, then Top, then Face.
5. Review representative outputs from each view, including portrait/landscape and EXIF-rotated examples.
6. Record misses, false positives, clipped targets, excess background, and inconsistent framing.
7. Keep source images, annotations, and model files unchanged during the validation run.
8. Save the batch summary and visual findings with the project validation record.
