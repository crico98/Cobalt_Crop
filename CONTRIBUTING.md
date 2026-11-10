# Contributing

## Start from the active development branch

The current crop-pipeline implementation is on:

~~~text
feature/view-specific-crop-pipeline
~~~

The repository default branch is main, which does not yet represent this implementation. Do not assume that cloning the default branch gives you the current project state.

Before starting work:

~~~powershell
git switch feature/view-specific-crop-pipeline
git pull
~~~

## Before changing code

1. Read docs/ARCHITECTURE.md.
2. Read docs/PROJECT_STATUS.md.
3. Read the relevant section of CROP_PIPELINE_PLAN.md.
4. Confirm whether the change affects view routing, model selection, crop geometry, training data, or the Face baseline.
5. Open an issue or discuss the intended change before changing accepted crop behavior.

## Protected behavior and assets

Do not change these without explicit review and approval:

- The explicit mapping: 0 Isometric, 1 Top, 2 Face.
- Processing order: Isometric → Top → Face.
- Face V2 model selection and established crop behavior.
- Accepted Top and Isometric crop geometry.
- The original annotation manifest and baseline YOLO export.
- Trained model checkpoints and source training images.
- The boundary between the StandAlone validation harness and future BitHub integration.

Do not silently add filename-based view inference or infer a view from EXIF, image orientation, image pixels, or upload order.

## Data and model handling

The following are local or separately distributed assets and should not be committed without explicit approval:

- Proprietary cutter source photos.
- Annotation manifests.
- Generated training datasets and derived presentation datasets.
- Top and Isometric model checkpoints.
- Batch uploads and generated crop outputs under work/.

The Face V2 checkpoint is currently committed in this repository. Repository collaborators can access committed files, including model binaries. Confirm that a person is authorized to receive repository contents before inviting them.

## Validation expectations

Run the unit tests before submitting a code change:

~~~powershell
python -m unittest discover -s tests -v
~~~

If a change affects inference, also run the relevant view with the approved local checkpoint and visually inspect representative crops. Unit tests alone are not sufficient for model or crop acceptance.

Document any changed behavior, the images/views used for validation, known misses, and whether an accepted baseline has changed.

## Pull requests

- Keep changes focused and explain the reason for the change.
- Identify affected views (Face, Top, Isometric).
- State whether model files, source images, annotation data, or crop geometry changed.
- Include unit-test results and visual-validation evidence where relevant.
- Do not merge to main or begin BitHub integration without explicit project-owner approval.
