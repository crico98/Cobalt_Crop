# Cropper StandAlone — view-specific crop pipeline prototype

Cropper StandAlone is a development and validation harness for the crop-processing tool intended for eventual use inside BitHub. It is not a separate end-user product. This branch validates view routing, independent model inference, detection regions, and output crop geometry before any BitHub integration work.

## Current batch workflow

1. Select a mixed batch of cutter images. Each filename must contain exactly one delimited angle token: `angle_0`, `angle_1`, or `angle_2`.
2. The filename token determines the view; image orientation never determines the view:
   - `angle_0` → Isometric
   - `angle_1` → Top
   - `angle_2` → Face
3. Upload validation rejects missing, unsupported, or ambiguous angle tokens before saving the batch.
4. The batch is stably grouped and processed Isometric first, Top second, and Face last. Original order is retained within each group.
5. EXIF orientation is applied once during upload. Additional manual 90-degree rotation remains available as a correction.
6. Each view uses its own inference path. A missing detector checkpoint is reported for that view; no image is silently routed to another model.
7. Review each output. Successful crops can be downloaded individually or as a ZIP.

## Model and detection-region configuration

| View | Model target | Checkpoint | Landscape region | Portrait region |
|---|---|---|---:|---:|
| Isometric | Diamond table | `models/isometric_view/weights/best.pt` | 55% × 80% | 65% × 55% |
| Top | Full visible cutter | `models/top_view/weights/best.pt` | 50% × 68% | 63% × 62% |
| Face | Cutter-face segmentation | Existing Face V2 checkpoint | 55% × 80% | 60% × 45% |

The Face path retains its existing V2 segmentation-to-circle geometry, crop framing, review flags, and interactive detection-region behavior. Top and Isometric use independent Ultralytics object detectors and translate detections from the centered model-input region back into full working-image coordinates.

Top and Isometric outputs are square, 1200 × 1200 presentation crops. They preserve aspect ratio and pad beyond source-image edges rather than stretching the cutter. The current preview coverage factors are **provisional only**: Top 0.80 and Isometric 0.65. They are exposed in `crop_pipeline.py` and must be visually reviewed against representative images before being treated as production framing values.

## Model files

The Top and Isometric checkpoint directories are intentionally excluded by `.gitignore` because trained model binaries are local artifacts. Restore or train these files locally before inference:
- `models/top_view/weights/best.pt`
- `models/isometric_view/weights/best.pt`

The upload page reports checkpoint readiness. A missing checkpoint does not trigger fallback to Face or the other view's detector.

## Running locally

Activate the project's virtual environment and install dependencies while preserving the locally installed PyTorch build:
```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

Combined upload size defaults to 50000 MB and can be changed with `CROPPER_MAX_UPLOAD_MB`. The default maximum batch size is 500 images and can be changed with `CROPPER_MAX_FILES`.

## Automated checks

Run the crop-pipeline unit tests:
```powershell
python -m unittest discover -s tests
```

These tests cover filename routing, invalid/missing/ambiguous angle identifiers, stable group order, all six view/orientation detection-region combinations, coordinate translation, square crop geometry, and confidence-based detector selection. They do not replace inference tests with the locally stored trained checkpoints or visual crop acceptance.

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