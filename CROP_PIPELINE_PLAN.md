# View-specific crop pipeline implementation plan

## Purpose

Validate the production crop-processing behavior in Cropper StandAlone before implementing the tool inside BitHub. StandAlone is a development/validation harness, not a separate end-user product. Do not merge this branch into `main` or begin BitHub migration until the complete crop workflow has been tested and accepted.

## Production input and processing contract

- BitHub supplies a JSON manifest for each uploaded batch. The known source field is `file_path`; the view assignment is carried separately in `angle_index`.
- StandAlone creates/uses the same explicit mapping in its own JSON manifest. Each record contains the photo filename/path in `file_path` and its numeric `angle_index`. The filename is used only to match the uploaded image; it is not expected to contain an angle token or EXIF angle metadata.
- Required mapping and processing order:
  1. `angle_index: 0` → Isometric — process every image in this group first.
  2. `angle_index: 1` → Top — process every image in this group second.
  3. `angle_index: 2` → Face — process every image in this group last.
- The actual BitHub JSON envelope is unavailable, so the StandAlone adapter uses a documented JSON array of records with the known `file_path` and `angle_index` fields. Only the adapter should need adjustment if the external envelope differs.
- Preserve the incoming order of images within each angle group unless BitHub's manifest defines a more specific ordering requirement.
- Validate angle identifiers before inference. Missing, malformed, or unsupported identifiers must be reported as input errors; never silently guess a view or route an image to a default model.
- Keep the batch's source paths and angle metadata attached to each processing item so outputs can be traced back to their source records.
- StandAlone's current direct-upload interface may remain a testing harness, but it must exercise the same angle-to-view mapping and group-processing order as production. Do not introduce a separate manual view-selection workflow.

## Implementation status on this branch

The following pipeline components are implemented:

- Explicit `angle_index` resolver for values 0, 1, and 2.
- Strict rejection of missing, malformed, unsupported, duplicate, or unmatched manifest/photo records.
- Stable Isometric → Top → Face grouping, preserving manifest order within each group.
- Direct-upload harness requires a JSON array of `{file_path, angle_index}` records and matches each record to an uploaded photo by filename. No view is inferred from filename, EXIF, image pixels, or upload position.
- Independent lazy-loaded Top and Isometric YOLO detectors, plus the unchanged Face V2 path.
- Six view/orientation detection-region settings, centered region extraction, and translation of detector boxes back to full working-image coordinates.
- Square Top/Isometric output geometry, fixed 1200-pixel output, aspect-ratio preservation, and edge padding.
- Top/Isometric permissive candidate inference and confidence-weighted box fusion, with unchanged single-member clusters and existing no-detection behavior.
- Batch preview labels the view, angle, model input region, detector confidence, and crop framing.
- Unit tests cover routing, stable ordering, invalid inputs, all six region settings, coordinate mapping, square crop geometry, detection selection, and WBF fusion behavior.

The Top and Isometric model directories are excluded by `.gitignore`; their locally trained `best.pt` files must be present on the workstation for inference. The application reports missing checkpoints and does not substitute another view's model.

**Top crop framing is visually accepted at target coverage 0.70.** This value was lowered from 0.80 after the initial Top runtime crop clipped part of the cutter; the user reviewed the same-image comparison and approved the 0.70 framing. The application treats Top framing as accepted in its preview metadata. **Isometric crop framing is now accepted for the current StandAlone baseline** at target coverage 0.80, with the detected diamond-table center positioned at 70% of output height. The user approved this position after comparing the 55%, 65%, and 70% placements. The crop geometry keeps the complete detected box inside the crop while retaining surrounding carbide.

## Top/Isometric detection fusion

Top and Isometric production inference now uses the validated single-model,
confidence-weighted box-fusion path in `crop_pipeline.py`.

- Run the detector with `BOX_FUSION_CANDIDATE_IOU = 0.99` to retain overlapping
  candidates that normal NMS would otherwise suppress at its usual threshold.
- Cluster candidates using `BOX_FUSION_CLUSTER_IOU = 0.60`.
- Fuse coordinates within each overlapping cluster using confidence-weighted
  averages; the fused cluster confidence is the mean member confidence.
- A single-member cluster is returned unchanged. Multiple non-overlapping
  detections remain separate; they are never averaged together.
- Select among resulting clusters using the established 3-percentage-point
  confidence band, then detector-input center proximity, then confidence.
- If no candidates are returned, preserve the existing no-crop/error behavior.
- Face segmentation and its crop path are unchanged.
- Record whether fusion occurred, the source-candidate count, and source IDs
  in the per-image detection metadata.

### Full-batch validation record

The user-reviewed comparison covered 106 images: 70 Isometric and 36 Top.
The comparison CSV contains one result per method (baseline, NMS, WBF) for
each image. WBF produced a selected crop for 102 images; the other four Top
images had no detections in any of the three methods and therefore retain the
existing no-crop behavior. Every selected WBF cluster in the successful
images combined multiple candidates (3–20 source boxes). The user visually
reviewed the WBF outputs and confirmed that they looked better than both
baseline and NMS. This supports integrating the validated WBF approach into
the StandAlone Top/Isometric crop path; it does not change the accepted Isometric target-coverage setting or authorize BitHub migration.

## Existing model baseline

- Face inference uses the existing segmentation checkpoint through `FaceFinder`.
- Face geometry and square crop behavior are established and must remain unchanged.
- Top detector checkpoint: `models/top_view/weights/best.pt`, class 0 `full_cutter`.
- Isometric detector checkpoint: `models/isometric_view/weights/best.pt`, class 0 `diamond_table`.
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

Apply the selected region to the normalized working image based on the resolved view and actual working-image orientation. Translate detector coordinates back into full working-image coordinates before constructing an output crop. Do not apply training-label expansion rules to inference coordinates.

## Detection targets

- Top: use the complete visible cutter bounding box. The final crop must retain the complete cutter.
- Isometric: use the visible diamond-table bounding box. The final crop must retain useful carbide context while making the table prominent.
- Face: continue using the existing segmentation-to-circle geometry and current crop behavior without changes.

## Output crop requirements

- Top and Isometric outputs are independently normalized square presentation crops at a consistent fixed output resolution.
- Preserve source aspect ratio; never stretch the cutter.
- Pad beyond source-image edges when the square crop extends outside the image.
- Top framing is accepted at target coverage 0.70; it must retain the entire visible cutter with consistent surrounding margin.
- Isometric framing is accepted at target coverage 0.80. The detected diamond-table center is placed at 70% of output height, and the crop geometry contains the complete detected box while retaining surrounding carbide.

## Implementation sequence

1. Implement and test the explicit `angle_index` metadata resolver and stable group ordering.
2. Add per-view inference adapters and load only the checkpoint associated with the resolved view.
3. Add view-aware detection-region selection using the six settings above.
4. Map predicted boxes from detection-region coordinates back to normalized working-image coordinates.
5. Add a previewable crop-geometry layer for Top and Isometric, keeping target-coverage factors configurable and visible; Top is accepted at 0.70 and Isometric is accepted at target coverage 0.80 and 70% vertical placement.
6. Preserve the Face code path and its defaults.
7. Add unit tests for angle mapping, invalid/missing identifiers, stable group ordering, orientation settings, coordinate translation, square crop geometry, image-edge padding, and Face-path regression.
8. Run local application tests and review representative Top and Isometric output crops before finalizing crop factors.
9. Only after crop validation, benchmark alternative detector architectures on the same held-out data.

## Merge gate

Do not merge this branch into `main`, and do not begin BitHub migration, until:
- Production angle mapping and Isometric → Top → Face group order are verified.
- Top and Isometric inference both work in the app.
- All six view/orientation detection-region combinations are verified.
- Top crop framing is visually accepted at 0.70.
- Isometric crop framing is visually accepted after reviewing the 0.80-coverage, 70%-vertical-position test output.
- Face behavior remains unchanged.
- Tests pass and the cropper baseline is documented.
