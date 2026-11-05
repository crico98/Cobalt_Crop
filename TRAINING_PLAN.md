# Top and Isometric detector development

## Current state

The production baseline in this repository is the V2 **Face segmentation** model. Its inference code consumes YOLO segmentation polygons/masks and fits cutter-face geometry from those masks. Top and Isometric views should be treated as separate view-specific models until evaluation shows that a shared model is reliable.

## Proposed model layout

Keep the existing Face model unchanged. Add two independent segmentation checkpoints:

- `models/top_view/weights/best.pt`
- `models/isometric_view/weights/best.pt`

The app should eventually offer an explicit view choice per image: Face, Top, or Isometric. Do not infer view from orientation: a manually rotated photo's portrait/landscape dimensions do not tell us whether it is a Top or Isometric view.

Each model should initially have one segmentation class, `cutter_face`, with a polygon around the visible cutter face. This matches the current geometry pipeline better than bounding boxes. If Top or Isometric images require a different target boundary, document that before annotation rather than mixing incompatible label definitions.

## Dataset structure

Keep source images and labels separated by view, and split by physical cutter/sample or acquisition session—not random near-duplicate frames—to reduce leakage:

```text
training/
  face/
    images/train/
    images/val/
    labels/train/
    labels/val/
  top/
    images/train/
    images/val/
    labels/train/
    labels/val/
  isometric/
    images/train/
    images/val/
    labels/train/
    labels/val/
```

For Ultralytics segmentation, each image needs a matching YOLO polygon label file. Preserve a copy of original photos and annotation exports. Do not overwrite source images during resizing or augmentation.

## Training and acceptance sequence

1. Gather representative Top and Isometric examples, including partial occlusions, edge-of-frame cutters, lighting variation, and the range of cutter conditions expected in production.
2. Agree on the exact polygon boundary for each view.
3. Annotate polygons and review a sample together before labeling the full set.
4. Split train/validation by physical sample or acquisition session.
5. Train Top and Isometric independently from a documented pretrained segmentation checkpoint; record model version, dataset revision, image size, epochs, and metrics.
6. Evaluate on a held-out set and visually inspect masks, center, radius, reconstructed geometry, and final crops.
7. Only then wire each checkpoint into the app behind an explicit per-image view selector. Keep Face as the default and preserve a way to compare model output before replacing any existing workflow.

## Information needed before training

- The actual Top and Isometric image datasets (or a first representative sample set).
- Confirmation that the desired annotation is the visible cutter-face boundary as a polygon, consistent with Face V2.
- Whether Top and Isometric should return the same square crop/87.5% face-coverage output as Face V2, or whether either view needs different crop geometry.

No Top or Isometric weights are included yet. A detector cannot be trained meaningfully from code alone; it needs representative labeled images.
