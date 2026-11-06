from __future__ import annotations

import io
import os
import uuid
import zipfile
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image
from image_orientation import normalize_uploaded_image, read_exif_orientation
from werkzeug.utils import secure_filename

from face_model import FaceFinder
from top_isometric_web import top_isometric

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Combined request-size limit, configurable before starting Flask.
# PowerShell: $env:CROPPER_MAX_UPLOAD_MB = "750"
MAX_UPLOAD_MB = max(1, int(os.environ.get("CROPPER_MAX_UPLOAD_MB", "50000")))
MAX_FILES_PER_BATCH = max(1, int(os.environ.get("CROPPER_MAX_FILES", "500")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
finder = FaceFinder()
app.register_blueprint(top_isometric)

# Local single-user prototype: metadata is held in memory until Flask restarts.
batches: dict[str, dict] = {}
jobs: dict[str, dict] = {}


def batch_dir(batch_id: str) -> Path:
    return WORK / batch_id


def item_dir(batch_id: str, job_id: str) -> Path:
    return batch_dir(batch_id) / job_id


def load_rgb(path: Path) -> Image.Image:
    """Load a working image; EXIF was normalized exactly once during upload."""
    with Image.open(path) as source:
        return source.convert("RGB")


def save_rgb(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG")


def padded_square_crop(image: Image.Image, face: dict, max_size: int = 1200):
    """Square crop using V2 face coverage, padding at image edges."""
    cx, cy, radius = float(face["cx"]), float(face["cy"]), float(face["r"])
    half = max(1, int(round(radius / 0.875)))
    left, top = int(round(cx)) - half, int(round(cy)) - half
    right, bottom = int(round(cx)) + half + 1, int(round(cy)) + half + 1
    side = max(2, right - left, bottom - top)
    right, bottom = left + side, top + side

    canvas = Image.new("RGB", (side, side), (24, 24, 24))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, right), min(image.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        region = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(region, (src_left - left, src_top - top))
    if side > max_size:
        canvas = canvas.resize((max_size, max_size), Image.Resampling.LANCZOS)
    return canvas, (left, top, right, bottom)


def adjusted_detection_region(width: int, height: int, manually_rotated: bool) -> tuple[int, int]:
    """Expand the centered model-input region without changing any annotations."""
    if manually_rotated:
        return width, min(100, int(round(height * 1.20)))
    return (
        min(100, int(round(width * 1.05))),
        min(100, int(round(height * 1.05))),
    )


def require_batch(batch_id: str) -> dict:
    batch = batches.get(batch_id)
    if batch is None:
        abort(404)
    return batch


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        model_available=finder.available,
        max_upload_mb=MAX_UPLOAD_MB,
        max_files=MAX_FILES_PER_BATCH,
        error=f"The combined upload exceeds the current {MAX_UPLOAD_MB} MB limit. You can change it with CROPPER_MAX_UPLOAD_MB.",
    ), 413


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_available=finder.available,
        max_upload_mb=MAX_UPLOAD_MB,
        max_files=MAX_FILES_PER_BATCH,
        error=None,
    )


@app.post("/upload")
def upload():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        abort(400, "Choose at least one image.")
    if len(files) > MAX_FILES_PER_BATCH:
        abort(400, f"Too many images. Maximum per batch: {MAX_FILES_PER_BATCH}.")

    batch_id = uuid.uuid4().hex
    folder = batch_dir(batch_id)
    folder.mkdir(parents=True, exist_ok=True)
    item_ids = []
    used_names: set[str] = set()

    try:
        for file in files:
            original_name = secure_filename(file.filename) or "cutter_image"
            suffix = Path(original_name).suffix.lower()
            if suffix not in ALLOWED:
                raise ValueError(f"Unsupported image type: {original_name}")

            stem = Path(original_name).stem or "cutter_image"
            output_stem = stem
            counter = 2
            while output_stem.lower() in used_names:
                output_stem = f"{stem}_{counter}"
                counter += 1
            used_names.add(output_stem.lower())

            job_id = uuid.uuid4().hex
            one_dir = item_dir(batch_id, job_id)
            one_dir.mkdir()
            try:
                with Image.open(file.stream) as source:
                    source_exif_orientation = read_exif_orientation(source)
                    normalized = normalize_uploaded_image(source)
                    save_rgb(normalized, one_dir / "working.png")
            except Exception as exc:
                raise ValueError(f"Could not read {original_name} as an image.") from exc

            jobs[job_id] = {
                "id": job_id,
                "filename": original_name,
                "output_name": output_stem,
                "source_exif_orientation": source_exif_orientation,
                "rotation": 0,
                "crop": False,
                "face": None,
                "error": None,
            }
            item_ids.append(job_id)

        batches[batch_id] = {"items": item_ids}
    except Exception as exc:
        for job_id in item_ids:
            jobs.pop(job_id, None)
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        abort(400, str(exc))

    return redirect(url_for("batch_view", batch_id=batch_id))


@app.get("/batch/<batch_id>")
def batch_view(batch_id):
    batch = require_batch(batch_id)
    items = [jobs[job_id] for job_id in batch["items"]]
    completed = sum(bool(item["crop"]) for item in items)
    failed = sum(bool(item["error"]) for item in items)
    return render_template(
        "batch.html", batch_id=batch_id, items=items,
        completed=completed, failed=failed,
        total=len(items),
        landscape_width=batch.get("landscape_width", 55),
        landscape_height=batch.get("landscape_height", 80),
        portrait_width=batch.get("portrait_width", 60),
        portrait_height=batch.get("portrait_height", 45),
    )


@app.post("/rotate/<batch_id>/<job_id>/<direction>")
def rotate(batch_id, job_id, direction):
    batch = require_batch(batch_id)
    if job_id not in batch["items"] or direction not in {"left", "right"}:
        abort(404)
    item = jobs[job_id]
    path = item_dir(batch_id, job_id) / "working.png"
    image = load_rgb(path)
    angle = 90 if direction == "left" else -90
    save_rgb(image.rotate(angle, expand=True), path)
    item["rotation"] = (item["rotation"] + angle) % 360
    item["crop"] = False
    item["face"] = None
    item["error"] = None
    (item_dir(batch_id, job_id) / "crop.png").unlink(missing_ok=True)
    return redirect(url_for("batch_view", batch_id=batch_id))


@app.post("/autocrop/<batch_id>")
def autocrop_batch(batch_id):
    batch = require_batch(batch_id)
    try:
        settings = {
            "landscape_width": int(request.form.get("landscape_width", "55")),
            "landscape_height": int(request.form.get("landscape_height", "80")),
            "portrait_width": int(request.form.get("portrait_width", "60")),
            "portrait_height": int(request.form.get("portrait_height", "45")),
        }
    except (TypeError, ValueError):
        abort(400, "All detection-region settings must be whole-number percentages.")
    if any(value < 25 or value > 100 for value in settings.values()):
        abort(400, "Detection-region settings must each be between 25% and 100%.")
    batch.update(settings)

    for job_id in batch["items"]:
        item = jobs[job_id]
        item["crop"], item["face"], item["error"] = False, None, None
        try:
            image = load_rgb(item_dir(batch_id, job_id) / "working.png")
            # Choose settings from the actual saved working image after manual rotation.
            is_landscape = image.width >= image.height
            base_width = settings["landscape_width" if is_landscape else "portrait_width"]
            base_height = settings["landscape_height" if is_landscape else "portrait_height"]
            manually_rotated = bool(item["rotation"] % 360)
            region_width, region_height = adjusted_detection_region(
                base_width, base_height, manually_rotated
            )
            orientation_name = "landscape" if is_landscape else "portrait"
            rotation_note = "manual-rotation expansion" if manually_rotated else "standard expansion"

            print(
                f"\n[Face detector] {item['filename']} | "
                f"working image={image.width}x{image.height} ({orientation_name}) | "
                f"base={base_width}%W x {base_height}%H | "
                f"effective region={region_width}%W x {region_height}%H "
                f"({rotation_note})",
                flush=True,
            )

            # Width and height are independent percentages of the original.
            # Run the model only on this centered rectangle, then translate
            # its detection coordinates back to the full working image.
            crop_width = max(1, int(round(image.width * region_width / 100.0)))
            crop_height = max(1, int(round(image.height * region_height / 100.0)))
            offset_x = (image.width - crop_width) // 2
            offset_y = (image.height - crop_height) // 2
            model_image = image.crop((
                offset_x, offset_y,
                offset_x + crop_width, offset_y + crop_height,
            ))

            face = finder.find(model_image)
            if face is None:
                item["error"] = (
                    f"No cutter face detected in the centered {region_width}% "
                    f"width × {region_height}% height region ({orientation_name}, {rotation_note}). "
                    "Try adjusting the region or adding representative training images."
                )
                print(f"[Face detector] NO DETECTION: {item['filename']}", flush=True)
                continue

            print(
                f"[Face detector] DETECTED {item['filename']} | "
                f"confidence={float(face.get('conf', 0)):.3f} | "
                f"center-in-region=({float(face['cx']):.1f}, {float(face['cy']):.1f})",
                flush=True,
            )
            face["cx"] = float(face["cx"]) + offset_x
            face["cy"] = float(face["cy"]) + offset_y
            crop, box = padded_square_crop(image, face)
            save_rgb(crop, item_dir(batch_id, job_id) / "crop.png")
            item["crop"] = True
            item["face"] = {
                "confidence": round(float(face.get("conf", 0)), 3),
                "center": [round(float(face["cx"]), 1), round(float(face["cy"]), 1)],
                "radius": round(float(face["r"]), 1),
                "reconstructed": bool(face.get("geometry_reconstructed")),
                "review": bool(face.get("geometry_needs_review") or face.get("low_conf") or face.get("not_very_circular")),
                "box": box,
                "region_width": region_width,
                "region_height": region_height,
            }
        except Exception as exc:
            import traceback
            item["error"] = f"Processing error: {type(exc).__name__}: {exc}"
            print(f"[Face detector] ERROR on {item['filename']}: {exc}", flush=True)
            traceback.print_exc()
    return redirect(url_for("batch_view", batch_id=batch_id))


@app.get("/image/<batch_id>/<job_id>/<kind>")
def image_file(batch_id, job_id, kind):
    batch = require_batch(batch_id)
    if job_id not in batch["items"]:
        abort(404)
    name = {"working": "working.png", "crop": "crop.png"}.get(kind)
    if not name:
        abort(404)
    path = item_dir(batch_id, job_id) / name
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png", max_age=0)


@app.get("/download/<batch_id>/<job_id>")
def download_one(batch_id, job_id):
    batch = require_batch(batch_id)
    if job_id not in batch["items"]:
        abort(404)
    item = jobs[job_id]
    path = item_dir(batch_id, job_id) / "crop.png"
    if not item["crop"] or not path.exists():
        abort(404, "This image has not been cropped successfully.")
    return send_file(path, as_attachment=True, download_name=f"{item['output_name']}_centered.png")


@app.get("/download-batch/<batch_id>")
def download_batch(batch_id):
    batch = require_batch(batch_id)
    memory_file = io.BytesIO()
    added = 0
    with zipfile.ZipFile(memory_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job_id in batch["items"]:
            item = jobs[job_id]
            path = item_dir(batch_id, job_id) / "crop.png"
            if item["crop"] and path.exists():
                archive.write(path, arcname=f"{item['output_name']}_centered.png")
                added += 1
    if not added:
        abort(404, "There are no successfully cropped images to download.")
    memory_file.seek(0)
    return send_file(memory_file, mimetype="application/zip", as_attachment=True, download_name="centered_cutter_faces.zip")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
