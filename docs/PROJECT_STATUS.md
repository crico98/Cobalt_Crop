# Project status and handoff

## Current project identity

- Repository: crico98/Standalone-Crop
- Active development branch: feature/view-specific-crop-pipeline
- Default branch: main
- Purpose: local validation harness for a view-specific crop workflow intended for eventual BitHub integration.
- Product status: prototype / validation harness; not a production multi-user service.

Use the active feature branch when reviewing the current pipeline. The default branch has not been updated to this implementation. Do not merge or migrate to BitHub until the user explicitly approves the complete validation result.

## Current implementation

- JSON manifest routing uses file_path and angle_index.
- View mapping is explicit: 0 Isometric, 1 Top, 2 Face.
- Processing order is Isometric → Top → Face, preserving order within each view.
- Uploaded images are EXIF-normalized once and saved as RGB working images.
- Top and Isometric use separate YOLO object-detection models and separate crop geometry.
- Face uses the existing Face V2 segmentation and crop path.
- Top and Isometric inference use confidence-weighted box fusion for overlapping detections.
- The browser-based Top / Isometric annotation editor and training preparation utilities remain in the repository.
- Unit tests cover routing, geometry, image orientation, and training preparation.

## Accepted baseline decisions

| Area | Current accepted setting |
|---|---|
| Face | Existing Face V2 model and crop geometry remain unchanged |
| Face sample | DSC_0560.JPG visually accepted as a representative single-image crop |
| Top target | Complete visible cutter |
| Top crop target coverage | 0.70 |
| Isometric target | Visible diamond table |
| Isometric crop target coverage | 0.80 |
| Isometric vertical placement | Diamond-table center at 70% of output height |
| Top / Isometric output | Square 1200 × 1200, aspect ratio preserved, edge padding |
| Training-label preparation | Standard group: +5% total width and height; EXIF quarter-turn group: width unchanged, +20% total height |

## Model availability and sharing

- Face V2 checkpoint is committed at models/face_combined_v2_finetune/weights/best.pt.
- Top checkpoint is approved for sharing and is configured for Git LFS at models/top_view/weights/best.pt.
- Isometric checkpoint is approved for sharing and is configured for Git LFS at models/isometric_view/weights/best.pt.
- The .gitattributes rule tracks PyTorch .pt files with Git LFS; .gitignore no longer excludes the Top or Isometric model directories.
- The two local Top and Isometric checkpoint files still need to be added and pushed from the repository owner's workstation. Until that upload is verified on GitHub, the branch does not yet contain all three model binaries.
- Authorized collaborators need Git LFS installed to retrieve the two large checkpoints.
- Proprietary training photos, annotation manifests, and generated datasets remain local/untracked.

## Known limitations

- Batch and job metadata are held in memory and are lost when the Flask process restarts.
- Working images and crops remain under work/ until manually cleaned up.
- Processing is sequential.
- The app is designed for local, single-user validation, not concurrent production use.
- The actual BitHub JSON envelope is not available in this repository context. StandAlone uses a documented JSON array adapter with file_path and angle_index; only that adapter should need adjustment if the production envelope differs.
- Automated tests do not perform model inference or replace visual review with the local checkpoints.

## Pending work before full handoff

- Complete the full-photo JSON-manifest batch test.
- Review and record mixed-view batch results and any exceptions.
- Confirm a clean collaborator setup using the documented dependencies and separately supplied model files.
- Review documentation against the result of that full test.
- Decide whether the feature branch is ready to become the shared development baseline. This is a separate approval; it does not automatically authorize merging to main or BitHub migration.

## Handoff readiness checklist

- [x] Current manifest routing documented.
- [x] Architecture and module responsibilities documented.
- [x] Local setup and model locations documented.
- [x] Automated test command documented.
- [x] Protected data and model boundaries documented.
- [ ] Full-photo batch validation completed.
- [ ] Clean collaborator setup verified.
- [ ] Final handoff approved by repository owner.
