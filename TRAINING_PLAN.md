# Top and Isometric cutter detection and presentation crop development

## Objective

Train two independent object-detection models for different presentation requirements:

- **Top:** locate the complete visible cutter, including the full carbide body and diamond table. Its predicted box is the reference for a normalized crop that retains the entire cutter.
- **Isometric:** locate the visible diamond table. Its predicted box is the reference for a normalized crop in which the diamond table occupies a large portion of the image while retaining enough carbide to preserve the isometric context.

These are intentionally different targets. Do not use one shared annotation definition for both views.

## Model layout

Keep the existing production Face-view model and its inference/cropping logic unchanged. Develop two independent detection checkpoints:

- `models/top_view/weights/best.pt`
- `models/isometric_view/weights/best.pt`

The workflow supplies the view assignment. Do not train a view classifier or infer Top/Isometric from image orientation.

Each detector has one class:

- Top class 0: `full_cutter`
- Isometric class 0: `diamond_table`

## Annotation rules

### Top

Draw one tight bounding box around the **entire visible cutter**:
- Include the complete outer carbide body and the diamond table.
- Keep all visible carbide inside the box; do not crop the annotation to the diamond table.
- Exclude unrelated background and drill-bit pocket material outside the cutter.
- If the cutter is cut off by the photo edge or part of its outer boundary cannot be located, skip it unless the complete cutter target is still unambiguous.

### Isometric

Draw one tight bounding box around the **visible diamond table**:
- Include the visible diamond material/table, including wear, chips, and spalling when its boundary remains locatable.
- Exclude the carbide body and pocket from the annotation.
- The final presentation crop will expand beyond this box to retain useful carbide context while making the diamond table visually prominent.

Skip broken, obscured, blurry, or ambiguous targets rather than guessing. Multiple visible targets may be annotated individually.

## Presentation crop requirements

The detection box is a locator, not itself the final image crop.

- **Top output:** square, fixed-resolution presentation image; the entire cutter must remain visible, centered, and surrounded by consistent margins. Never clip the carbide body.
- **Isometric output:** square, fixed-resolution presentation image; center the diamond-table target and expand the crop enough that the table occupies a large, consistent portion of the frame while retaining some carbide context.
- Normalize Top and Isometric independently. Do not force the apparent diamond-table scale in Isometric to match the full-cutter scale in Top.
- Preserve the source orientation and aspect ratio. Use padding when a square crop extends beyond the source image rather than stretching the cutter.
- Keep crop expansion/coverage settings explicit and shared between training validation and inference. Do not choose arbitrary final margins until representative predictions have been reviewed.

## Dataset structure

```text
training_presentation/
  top/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
  isometric/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
```

Top and Isometric are separate one-class datasets. The annotation manifest stores normalized boxes against the original source images. The exporter must preserve the view-specific class names in each dataset YAML. Split by physical cutter/sample or acquisition session when possible—not random near-duplicate frames—to reduce leakage.

## Orientation-normalized training dataset

The app normalizes EXIF orientation once during upload and then treats the resulting pixels as the working-image coordinate system. For training, create a separate derived dataset from the current `training_presentation/` export. Do not overwrite the original annotation manifest or baseline export.

The preparation tool requires an explicit per-image `manual_rotation_degrees` value (0, 90, 180, or 270 clockwise) describing any additional rotation needed **after EXIF normalization**. EXIF values are not a substitute for this classification. The tool physically applies EXIF and declared manual rotations to image pixels, applies the same transform to the YOLO boxes, and writes PNGs without orientation metadata.

Derived box expansion:
- No additional manual rotation: 1.05× width and 1.05× height (5% total expansion in each dimension).
- Additional manual rotation: 1.00× width and 1.20× height (10% of original height on each side).

Expansion remains centered on the original box and is symmetrically limited at image edges. The derived output is `training_presentation_exif_normalized/`; keep it separate from the original `training_presentation/` baseline. Visually inspect representative and edge-limited boxes before training.

Training entry point: `python -m tools.train_top_isometric --view all`. Defaults: pretrained `yolo11n.pt`, 1024 image size, 100 epochs, batch 4, patience 25, GPU 0, and Windows-friendly workers=0. Top and Isometric are trained independently; outputs are stored under `models/top_view/` and `models/isometric_view/`.

## Training and acceptance sequence

1. Re-annotate Top images using the complete-cutter target; the exporter deliberately excludes legacy Top boxes.
2. Review existing Isometric diamond-face boxes to confirm they tightly represent the visible diamond table; redraw only inconsistent examples.
3. Review samples from each view for consistent target boundaries before exporting.
4. Export separate baseline datasets and audit every image/label pair, normalized box, class ID, and train/validation split.
5. Complete the per-image post-EXIF rotation manifest, then build the separate EXIF-normalized dataset with the agreed derived-box expansions. Do not infer manual rotation from EXIF.
6. Visually review the derived images and labels, including rotated examples and boxes constrained by image edges.
7. Train Top and Isometric independently using the documented pretrained YOLO detection checkpoint. Record model version, dataset revision, image size, epochs, and metrics.
8. Evaluate held-out images visually, measuring missed detections, false positives, box tightness, and final presentation crop consistency.
9. Implement deterministic view-specific crop expansion/normalization only after reviewing predictions on representative images.
10. Only after evaluation, wire each checkpoint into the cropper behind the explicit view supplied by the workflow.

## Scope boundaries

- No Top/Isometric view classifier.
- No cutter-body segmentation or diamond-face segmentation mask.
- No changes to the production Face-view model or crop behavior.
- No automatic view inference from portrait/landscape orientation.
- Preserve existing baseline model weights and the original source images/annotation manifest.
