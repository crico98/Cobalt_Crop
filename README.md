# Cropper StandAlone

Cropper StandAlone is a local development and validation harness for the view-specific cutter crop pipeline intended for eventual integration into BitHub. It is not a production multi-user service.

The current implementation is developed on **feature/view-specific-crop-pipeline**. The repository default branch is still **main**; use the feature branch when reviewing or running the current pipeline. Do not merge it into main or begin BitHub migration until the validation gates in the crop pipeline plan are complete and explicitly approved.

## What the application does

The Flask app accepts a batch of cutter photos and a JSON manifest. The manifest explicitly assigns each photo to a view:

| angle_index | View | Processing order |
|---:|---|---:|
| 0 | Isometric | First |
| 1 | Top | Second |
| 2 | Face | Third |

The filename is used only to match a manifest record to an uploaded photo. **The filename, EXIF metadata, image orientation, image pixels, and upload order do not determine the view.** Images are grouped Isometric → Top → Face while preserving their order within each group.

A starter manifest is available at manifest_template.json. Replace the example filenames with the exact photo filenames being uploaded.

## Current crop paths

- **Face:** existing Face V2 segmentation path and square-crop geometry. The Face model and crop behavior are intentionally kept independent from the Top and Isometric detectors.
- **Top:** independent YOLO object detector for the complete visible cutter. Accepted target coverage: 0.70.
- **Isometric:** independent YOLO object detector for the visible diamond table. Accepted target coverage: 0.80; the diamond-table center is positioned at 70% of output height.

Top and Isometric outputs are square 1200 × 1200 presentation crops. See docs/ARCHITECTURE.md and CROP_PIPELINE_PLAN.md for the detailed processing contract and validation history.

## Quick start

The current development workstation baseline uses Python 3.12 and a CUDA-enabled PyTorch installation. See docs/DEVELOPER_SETUP.md before installing dependencies, especially the separate PyTorch/CUDA step.

From the repository root in PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

Install the CUDA-compatible PyTorch and torchvision build appropriate for the machine, then install the project dependencies:

~~~powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python app.py
~~~

The PyTorch command is for a CUDA 12.8-capable NVIDIA setup. For CPU-only machines or a different CUDA environment, use the official PyTorch install selector and install a matching build. Do not replace a working CUDA-enabled PyTorch installation with a CPU-only build.

Open http://127.0.0.1:5000.

## Run a batch

1. Prepare a JSON array using file_path and angle_index for every image.
2. On the upload page, select all batch photos.
3. Select the matching JSON manifest.
4. Submit the batch and review the view label and crop result for each photo.

Example record:

~~~json
[
  {
    "file_path": "DSC_0560.JPG",
    "angle_index": 2
  }
]
~~~

The manifest must include every uploaded image exactly once. The app rejects missing, duplicate, unsupported, or unmatched records rather than guessing a view.

## Documentation

- Architecture and processing flow: docs/ARCHITECTURE.md
- Developer setup and model requirements: docs/DEVELOPER_SETUP.md
- Testing and validation: docs/TESTING.md
- Project status and handoff checklist: docs/PROJECT_STATUS.md
- Contributor workflow and protected assets: CONTRIBUTING.md
- Detailed implementation plan and validation record: CROP_PIPELINE_PLAN.md
- Top / Isometric model training plan: TRAINING_PLAN.md
- Preparing the Top + Isometric crop-test batch: CROP_TEST_BATCH.md

## Important project boundaries

- This is a local, single-user prototype. Batch/job metadata is held in memory; restarting Flask clears active batch metadata.
- Uploaded working files and generated crops are stored under work/, which is excluded from Git. They are not automatically removed when a batch finishes.
- Top and Isometric checkpoints are local model artifacts and are excluded from Git. They must be supplied separately to a collaborator who needs to run those views.
- The Face V2 checkpoint is currently present in the repository under models/face_combined_v2_finetune/weights/best.pt. Anyone granted repository access can access committed files, including this checkpoint.
- Proprietary source photos, annotation manifests, and generated training datasets should remain local/untracked unless there is explicit approval to share them.
- Automated unit tests do not replace inference tests with the actual trained checkpoints or visual review of full-batch results.

See docs/PROJECT_STATUS.md for what has and has not yet been validated.
