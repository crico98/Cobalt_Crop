# Cutter Cropper — standalone batch prototype

Separate Flask app for manually oriented cutter photos and V2 YOLO face cropping. It does not modify or import the existing Gradio or Plotly Dash applications.

## Features
- Upload multiple cutter photos in one batch.
- Rotate each photo manually by 90 degrees before processing.
- Process all photos sequentially with one **Auto Crop / Center Batch** action.
- Review each result and confidence/review indicators.
- Download successful crops individually or together as a ZIP.
- EXIF orientation is applied once during upload; additional manual 90-degree rotation remains available. No orientation classifier or deskew.
- Square crop target uses approximately 87.5% face coverage and pads beyond image edges.

## Configurable combined upload limit

Default combined request limit: **50000 MB**. Change it before starting the app with `CROPPER_MAX_UPLOAD_MB`.

PowerShell:
```powershell
$env:CROPPER_MAX_UPLOAD_MB = "750"
python app.py
```

This is the total request size for all selected files plus multipart overhead, not a per-image limit. Default maximum is 50 images per batch; change it with `CROPPER_MAX_FILES`:
```powershell
$env:CROPPER_MAX_FILES = "100"
```

## Run on Windows
Activate the virtual environment, navigate to this project folder, and run:
```powershell
python app.py
```
Open http://127.0.0.1:5000

Install requirements with `python -m pip install -r requirements.txt`. Preserve the CPU PyTorch build already installed for Intel integrated graphics.

## Orientation-aware center detection regions

The batch page has separate adjustable width/height settings for each working-image orientation:
- Landscape (width >= height): default **40% width × 70% height**.
- Portrait (height > width): default **80% width × 55% height**.

EXIF orientation is normalized during upload. Landscape/portrait is determined from the normalized working image after any additional manual 90-degree rotation. Each image in a mixed-orientation batch receives the matching settings. The preview rectangle updates per image, and the model receives only that centered rectangle. Detection coordinates are translated back to the full manually oriented image before the final square crop is made.

## Batch workflow
1. Select multiple JPG/PNG/WEBP/BMP/TIFF images.
2. Upload the batch; EXIF orientation is applied automatically once.
3. Rotate an image manually only if its orientation still needs correction after EXIF normalization.
4. Adjust the landscape and portrait width/height settings if needed, then click **Auto Crop / Center Batch**.
5. Review each result. A failed detection on one image does not stop the others.
6. Re-run with adjusted detection regions to compare, then download successful crops individually or as one ZIP.

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for the independent Top and Isometric diamond-face detection models.


## EXIF orientation and Top / Isometric training preparation

The upload pipeline applies the image's EXIF orientation once with Pillow's `ImageOps.exif_transpose()`, then saves normalized RGB pixels as `working.png`. The original EXIF orientation value is retained in the in-memory job metadata for diagnostics. Manual rotation is an additional correction relative to this normalized image. Detection-region selection, model coordinates, and final crop coordinates all use the normalized working-image pixel space.

The existing annotation manifest and `training_presentation/` export are the baseline and are not changed by training preparation. Build a separate derived dataset:

```powershell
python -m tools.prepare_top_isometric_training
```

The preparation script reads each exported image's EXIF orientation, applies EXIF normalization once to the pixels, and uses EXIF as the source of truth for the expansion group. Orientations 5, 6, 7, and 8 use the rotated-image expansion; orientation 1 and other non-quarter-turn orientations use the standard expansion. The existing labels are already in the EXIF-corrected coordinate convention, so the script does not rotate those boxes a second time. PNG output removes EXIF ambiguity.

The derived dataset is written to `training_presentation_exif_normalized/` and keeps the existing train/validation split. Derived labels use:
- EXIF-normal/non-rotated group: +5% total width and +5% total height.
- EXIF quarter-turn group: unchanged width and +20% total height.

Expansion is symmetric around the original box center and is limited symmetrically at image boundaries. The preparation summary records EXIF orientation counts. Original annotations and the baseline export remain read-only. Review the derived images and labels before training.

Train the independent detectors only after that visual review:
```powershell
python -m tools.train_top_isometric --view all
```
Defaults are `yolo11n.pt`, 1024 image size, 100 epochs, batch 4, patience 25, GPU 0, and zero data-loader workers. Use `--view top` or `--view isometric` to train one model. Checkpoints are written under `models/top_view/` and `models/isometric_view/`.

## Automated checks

Run the unit tests from the project root after installing the requirements:
```powershell
python -m unittest discover -s tests
```

## Top and Isometric cutter annotation

The Flask app includes a browser-based editor for separate YOLO **object-detection** datasets for manually assigned **Top** and **Isometric** views. These experimental datasets do not modify or replace the production Face model or crop workflow.

Start the app from the project root:
```powershell
python app.py
```
Open http://127.0.0.1:5000 and choose **Open Top / Isometric Annotation Editor**, or go directly to http://127.0.0.1:5000/annotate.

The editor reads images recursively from `dataset/train/images`. It stores annotations in `dataset/top_isometric_annotations.json`, using the version-3 manifest and normalized bounding-box format as the desktop annotation tool. Top annotations carry a target-version marker so old face-only Top boxes are excluded until redrawn; existing Isometric diamond-table boxes remain usable. Existing saved annotations are loaded when the editor opens; saving an image updates only that image's record.

Editor workflow:
- Drag on the image to draw a tight rectangle around each visible diamond face. Multiple faces can be boxed in one image.
- Drag inside an existing box to reposition it.
- Select a box and use **Delete selected box** (or the keyboard Delete key) to remove only that box. **Clear all boxes** removes all boxes for the current image.
- Assign the image as **Top** or **Isometric**.
- **Save annotation** saves the current image. Previous/Next also saves pending changes before navigating.
- **Export YOLO datasets** runs the existing exporter using the saved manifest.

Annotation targets are view-specific. Existing legacy **Top** annotations from the earlier diamond-face-for-both-views workflow are treated as stale and must be redrawn; they are not exported. Existing Isometric diamond-face boxes remain valid as diamond-table targets and are retained.
- **Top:** box the entire visible cutter, including all carbide and the diamond table. Exclude unrelated background and pocket material outside the cutter.
- **Isometric:** box the visible diamond table only. Exclude the surrounding carbide body and pocket; the downstream presentation crop will expand around the table to retain useful carbide context.

Include worn, chipped, or spalled targets when their boundaries remain locatable. Skip broken, obscured, cut-off, or unclear targets when the required boundary cannot be located reliably.

The exporter writes separate datasets to `training_presentation/top` and `training_presentation/isometric`, each with `images/train`, `images/val`, `labels/train`, `labels/val`, and `data.yaml`. Both use standard YOLO object-detection labels (class ID 0 and normalized center X/Y, width, and height), with distinct class names: `full_cutter` for Top and `diamond_table` for Isometric. The browser editor calls the same export function as the existing desktop script; the desktop script remains available for users who prefer it.

The default validation fraction is 20% (seed 42). If source images are organized in subfolders, each folder is kept together in one split. If all images are in one flat folder, images are split individually; group related acquisition sessions into subfolders first if you need to prevent near-duplicate leakage. Use the desktop export command with `--val-fraction 0` to put all exported images in train.

**Data protection:** keep proprietary source photos, the annotation manifest, and generated training datasets local/untracked unless there is an explicit reason to share them. The browser editor does not upload these images to GitHub. The existing desktop editor and exporter remain available:
```powershell
python tools/edit_top_isometric_labels.py
python tools/edit_top_isometric_labels.py --export-only
```

## Notes
- This is a local single-user prototype. Batch/job metadata is held in memory; restarting Flask clears active sessions. Uploaded images remain under `work/` until manually deleted.
- Processing is sequential, suitable for the current CPU-based workstation but potentially slow for large batches.
- Detection and geometry reconstruction should be visually validated on real samples.
- Keep `FaceFinder` and crop logic independent for eventual Dash integration.
#   S t a n d a l o n e - C r o p 
 
 