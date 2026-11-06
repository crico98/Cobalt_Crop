# View-specific crop pipeline implementation plan

## Purpose

Add Top and Isometric detection/cropping to the existing Cropper StandAlone application while preserving the established Face pipeline. This branch is a development branch; do not merge it into `main` until the complete crop workflow has been tested and accepted.

## Existing baseline

- Face inference uses the existing segmentation checkpoint through `FaceFinder`.
- Face geometry and square crop behavior are established and must remain unchanged.
- Top detector checkpoint: `models/top_view/weights/best.pt`, class 0 `full_cutter`.
- Isometric detector checkpoint: `models/isometric_view/weights/best.pt`, class 0 `diamond_table`.
- The workflow must supply the view explicitly. Do not infer view from image orientation.
- EXIF normalization occurs once at upload. Detection and crop coordinates use the normalized working-image coordinate system.
- Original annotation manifests and baseline YOLO exports are read-only.

## Detection-region configuration

These are centered source-image regions passed to the model, not final output crop dimensions.

| View | Working orientation | Width | Height |
|---|---|---:|---:|
| Top | Landscape | 50% | 68% |
| Top | Portrait | 63% | 62% |
| Isometric | Landscape | 55% | 80% |
| Isometric | Portrait | 65% | 55% |
| Face | Landscape | 55% | 80% |
| Face | Portrait | 60% | 45% |

Apply the selected region to the normalized working image based on the explicit view and actual working-image orientation. Translate detector coordinates back into full working-image coordinates before constructing an output crop. Do not apply training-label expansion rules to inference coordinates.

## Detection targets

- Top: use the complete visible cutter bounding box. The final crop must retain the complete cutter.
- Isometric: use the visible diamond-table bounding box. The final crop must retain useful carbide context while making the table prominent.
- Face: continue using the existing segmentation-to-circle geometry and current crop behavior without changes.

## Output crop requirements

- Top and Isometric outputs are independently normalized square presentation crops at a consistent fixed output resolution.
- Preserve source aspect ratio; never stretch the cutter.
- Pad beyond source-image edges when the square crop extends outside the image.
- Top framing must retain the entire visible cutter with consistent surrounding margin.
- Isometric framing must keep the diamond table prominent while retaining carbide context.
- The final Top and Isometric target-coverage / margin factors are intentionally unresolved until representative model predictions are run through a preview implementation and reviewed. Do not silently choose production crop margins.

## Implementation sequence

1. Add explicit per-view inference adapters and load only the checkpoint associated with the supplied view.
2. Add view-aware detection-region selection using the six settings above.
3. Map predicted boxes from detection-region coordinates back to normalized working-image coordinates.
4. Add a previewable crop-geometry layer for Top and Isometric, keeping final target-coverage factors configurable and visible.
5. Preserve the Face code path and its defaults.
6. Add unit tests for view selection, orientation settings, coordinate translation, square crop geometry, image-edge padding, and Face-path regression.
7. Run local application tests and review representative Top and Isometric output crops before finalizing crop factors.
8. Only after crop validation, benchmark alternative detector architectures on the same held-out data.

## Merge gate

Do not merge this branch into `main`, and do not begin BitHub migration, until:
- Top and Isometric inference both work in the app.
- All six view/orientation detection-region combinations are verified.
- Final Top and Isometric crop framing is visually accepted.
- Face behavior remains unchanged.
- Tests pass and the cropper baseline is documented.
