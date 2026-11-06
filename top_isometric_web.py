from __future__ import annotations

import math
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request, send_file
from PIL import Image

from tools.edit_top_isometric_labels import (
    collect_images,
    export_datasets,
    load_manifest,
    relative_key,
    save_manifest,
)

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset" / "train" / "images"
MANIFEST_PATH = ROOT / "dataset" / "top_isometric_annotations.json"
OUTPUT_ROOT = ROOT / "training"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

top_isometric = Blueprint("top_isometric", __name__)


def _image_path(key: str) -> Path:
    if not isinstance(key, str) or not key:
        abort(400, "An image key is required.")
    path = (IMAGE_ROOT / key).resolve()
    try:
        path.relative_to(IMAGE_ROOT.resolve())
    except ValueError:
        abort(400, "Invalid image key.")
    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
        abort(404, "Image not found.")
    return path


def _validated_boxes(raw_boxes) -> list[list[float]]:
    if not isinstance(raw_boxes, list):
        abort(400, "Boxes must be a list.")
    boxes = []
    for raw in raw_boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            abort(400, "Each box must contain four normalized coordinates.")
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError):
            abort(400, "Box coordinates must be numeric.")
        if not all(math.isfinite(value) for value in values):
            abort(400, "Box coordinates must be finite.")
        x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in values)
        if x2 - x1 < 0.003 or y2 - y1 < 0.003:
            abort(400, "Each box must have a visible width and height.")
        boxes.append([x1, y1, x2, y2])
    return boxes


@top_isometric.get("/annotate")
def editor():
    return render_template("top_isometric.html")


@top_isometric.get("/api/top-isometric/images")
def image_index():
    if not IMAGE_ROOT.is_dir():
        return jsonify({"images": [], "error": f"Image folder not found: {IMAGE_ROOT}"})
    images = collect_images(IMAGE_ROOT)
    manifest = load_manifest(MANIFEST_PATH)
    results = []
    for path in images:
        key = relative_key(path, IMAGE_ROOT)
        record = manifest["annotations"].get(key, {})
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        boxes = record.get("boxes")
        if boxes is None:
            boxes = []
        results.append({
            "key": key,
            "name": path.name,
            "width": width,
            "height": height,
            "view": record.get("view", ""),
            "boxes": boxes,
            "annotated": bool(record.get("view") in {"top", "isometric"} and boxes),
        })
    return jsonify({"images": results})


@top_isometric.get("/api/top-isometric/image")
def image_file():
    path = _image_path(request.args.get("key", ""))
    return send_file(path, max_age=0)


@top_isometric.post("/api/top-isometric/save")
def save_annotation():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key")
    _image_path(key)
    boxes = _validated_boxes(payload.get("boxes", []))
    view = payload.get("view", "")
    manifest = load_manifest(MANIFEST_PATH)

    # An explicitly saved empty annotation removes only this image's record.
    if not boxes:
        manifest["annotations"].pop(key, None)
    else:
        if view not in {"top", "isometric"}:
            abort(400, "Assign this image as Top or Isometric before saving boxes.")
        manifest["annotations"][key] = {"view": view, "boxes": boxes}

    save_manifest(MANIFEST_PATH, manifest)
    return jsonify({
        "saved": True,
        "key": key,
        "annotated": bool(boxes),
        "annotation_count": len(manifest["annotations"]),
    })


@top_isometric.post("/api/top-isometric/export")
def export_annotation_datasets():
    manifest = load_manifest(MANIFEST_PATH)
    export_datasets(IMAGE_ROOT, OUTPUT_ROOT, manifest, MANIFEST_PATH)
    return jsonify({
        "exported": True,
        "output": str(OUTPUT_ROOT),
        "annotation_count": len(manifest["annotations"]),
    })
