# Top and Isometric diamond-face detector development

## Objective

Train two independent object-detection models to locate the **visible diamond face** in their assigned view. The model does not classify view type and does not need to trace the cutter body or produce a precise segmentation mask. Its bounding box is used as the region of interest for the downstream crop.

## Model layout

Keep the existing production Face-view model and its inference/cropping logic unchanged. Develop two independent detection checkpoints:

- `models/top_view/weights/best.pt`
- `models/isometric_view/weights/best.pt`

The application or upstream image organization supplies the view assignment. Do not train a view classifier or infer Top/Isometric from image orientation.

Each detector has one object-detection class:

- Class 0: `diamond_face`

The target box should tightly enclose the visible diamond material/face. Exclude the surrounding carbide body, pocket, and unrelated material. The annotation is a crop locator, not a measurement of the exact diamond boundary.

## Annotation inclusion rules

Include intact, worn, chipped, or spalled cutters when the visible diamond face can still be located reliably. Exclude broken, obscured, blurry, or incomplete examples when the diamond face cannot be identified with confidence. Do not label an uncertain guess.

The annotation tool's **T** and **I** assignments are used only to route each image to the corresponding training dataset. They are not model targets.

## Dataset structure

```text
training/
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

Each image has a matching YOLO detection label with normalized `class x_center y_center width height` values. Split by physical cutter/sample or acquisition session when possible—not random near-duplicate frames—to reduce leakage. Keep source photos and the annotation manifest separate from generated training exports.

## Training and acceptance sequence

1. Gather representative Top and Isometric photos, including lighting variation, wear, chips/spalling, partial occlusion, and edge-of-frame examples.
2. Annotate visible diamond faces with bounding boxes and manually assign each image to Top or Isometric.
3. Review a sample of boxes for consistent target boundaries before labeling the full set.
4. Split train/validation by physical sample or acquisition session where possible.
5. Train Top and Isometric independently using a documented pretrained YOLO object-detection checkpoint. Record model version, dataset revision, image size, epochs, and metrics.
6. Evaluate held-out images and inspect missed detections, false positives, box tightness, and final crop results.
7. Only after evaluation, wire each checkpoint into the cropper behind the explicit view supplied by the workflow. Use the predicted diamond-face box as the crop region; preserve the existing Face-view path.

## Scope boundaries

- No Top/Isometric view classifier.
- No cutter-body polygon or diamond-face segmentation mask.
- No changes to the production Face model or its crop behavior during this dataset-building phase.
- No automatic view inference from portrait/landscape orientation.

No Top or Isometric model weights are included yet. Models can only be trained after representative images have been annotated and exported.
