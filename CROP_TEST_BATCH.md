# Preparing the Top + Isometric crop-test batch

The crop-test batch builder copies the training images into a disposable upload folder and writes the StandAlone JSON manifest. It does not modify source photos, annotations, labels, or model files.

## Default workstation paths

- Isometric: `C:\PDC_Training\training\isometric\images\train`
- Top: `C:\PDC_Training\training\top\images\train`
- Output: `work\crop_test_batch\`

## Run

From the repository root, with the project virtual environment active:

```powershell
python tools\prepare_crop_test_batch.py
```

The output folder contains:

- The copied JPG/other supported image files.
- `manifest.json`, with `angle_index: 0` for Isometric and `angle_index: 1` for Top.
- `batch_summary.json`, recording the copied counts and output location.

The script matches images by filename only. It does not infer a view from filenames, image pixels, EXIF, or upload order. The manifest controls view assignment.

The default output is under `work\`, which is excluded from Git. The source folders remain untouched. If the output already exists, the script stops rather than replacing it. To intentionally rebuild the generated output, run:

```powershell
python tools\prepare_crop_test_batch.py --overwrite
```

## Upload to the StandAlone app

On the upload page, select all images in `work\crop_test_batch\` and select `work\crop_test_batch\manifest.json` as the JSON manifest. Do not upload `batch_summary.json` as the manifest.

Expected source counts for the current workstation dataset are 70 Isometric and 36 Top images (106 total). Counts are reported from the actual folders at run time; the script does not hard-code those expected totals.
