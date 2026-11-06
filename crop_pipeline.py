from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


# Angle IDs are part of the source filename, not the image's aspect ratio.
ANGLE_VIEWS = {
    "0": ("isometric", 0),
    "1": ("top", 1),
    "2": ("face", 2),
}
VIEW_LABELS = {
    "isometric": "Isometric",
    "top": "Top",
    "face": "Face",
}
# Width and height are percentages of the full EXIF-normalized working image.
DETECTION_REGIONS = {
    ("top", "landscape"): (50, 68),
    ("top", "portrait"): (63, 62),
    ("isometric", "landscape"): (55, 80),
    ("isometric", "portrait"): (65, 55),
    ("face", "landscape"): (55, 80),
    ("face", "portrait"): (60, 45),
}
# Provisional preview framing only. These values are intentionally not a
# production acceptance decision; review representative crops before locking.
PROVISIONAL_CROP_COVERAGE = {
    "top": 0.80,
    "isometric": 0.65,
}
OUTPUT_SIZE = 1200


@dataclass(frozen=True)
class BatchItem:
    file_path: str
    angle_id: str
    view: str
    order: int


def resolve_angle_index(angle_index) -> tuple[str, str, int]:
    """Resolve the explicit JSON angle_index into (ID, view, processing order)."""
    if isinstance(angle_index, bool):
        raise ValueError("angle_index must be 0, 1, or 2.")
    if isinstance(angle_index, int):
        angle_id = str(angle_index)
    elif isinstance(angle_index, str) and angle_index.strip().isdigit():
        angle_id = str(int(angle_index.strip()))
    else:
        raise ValueError(f"Invalid angle_index: {angle_index!r}. Expected 0, 1, or 2.")
    if angle_id not in ANGLE_VIEWS:
        raise ValueError(f"Unsupported angle_index {angle_id}. Expected 0, 1, or 2.")
    view, order = ANGLE_VIEWS[angle_id]
    return angle_id, view, order


def normalize_manifest_records(records: Iterable[Mapping]) -> list[BatchItem]:
    """Validate explicit {file_path, angle_index} records and group stably by view."""
    items = []
    seen_paths = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Manifest record {index + 1} must be an object.")
        raw_path = record.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Manifest record {index + 1} is missing file_path.")
        normalized_path = Path(raw_path.replace("\\", "/")).name
        if not normalized_path:
            raise ValueError(f"Manifest record {index + 1} has an invalid file_path.")
        key = normalized_path.casefold()
        if key in seen_paths:
            raise ValueError(f"Manifest contains duplicate photo filename: {normalized_path}")
        seen_paths.add(key)
        if "angle_index" not in record:
            raise ValueError(f"Manifest record {index + 1} is missing angle_index.")
        angle_id, view, order = resolve_angle_index(record["angle_index"])
        items.append(BatchItem(raw_path, angle_id, view, order))
    return sorted(items, key=lambda item: item.order)


def load_manifest_records(path: str | Path) -> list[BatchItem]:
    """Load JSON records containing file_path and angle_index."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The StandAlone manifest must be a JSON array of photo records.")
    return normalize_manifest_records(payload)


def orientation_name(image: Image.Image) -> str:
    return "landscape" if image.width >= image.height else "portrait"


def detection_region(view: str, image: Image.Image) -> tuple[int, int]:
    try:
        return DETECTION_REGIONS[(view, orientation_name(image))]
    except KeyError as exc:
        raise ValueError(f"No detection-region configuration for view: {view}") from exc


def centered_detection_image(image: Image.Image, width_percent: int, height_percent: int):
    """Return the centered model input and its full-image (x, y) offset."""
    if not (1 <= width_percent <= 100 and 1 <= height_percent <= 100):
        raise ValueError("Detection-region percentages must be between 1 and 100.")
    crop_width = max(1, int(round(image.width * width_percent / 100.0)))
    crop_height = max(1, int(round(image.height * height_percent / 100.0)))
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)), (left, top)


def map_box_to_full_image(box, offset: tuple[int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    dx, dy = offset
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def square_crop_geometry(
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
    target_coverage: float,
) -> tuple[int, int, int, int]:
    """Build a square crop around a detected box; pad is applied by crop_square."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be greater than 0 and at most 1.")
    image_width, image_height = image_size
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Detection box coordinates must be finite.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box must have positive width and height.")
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(1, int(math.ceil(max(x2 - x1, y2 - y1) / (2.0 * target_coverage))))
    left, top = int(round(cx)) - half, int(round(cy)) - half
    side = 2 * half
    return left, top, left + side, top + side


def crop_square(image: Image.Image, box: tuple[int, int, int, int], output_size: int = OUTPUT_SIZE):
    """Square crop with dark padding outside the source; never stretch."""
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    if side <= 0:
        raise ValueError("Crop geometry must have positive dimensions.")
    right, bottom = left + side, top + side
    canvas = Image.new("RGB", (side, side), (24, 24, 24))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, right), min(image.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        canvas.paste(
            image.crop((src_left, src_top, src_right, src_bottom)),
            (src_left - left, src_top - top),
        )
    if output_size and canvas.size != (output_size, output_size):
        canvas = canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return canvas


def choose_detection(result):
    """Select the most confident valid detector box; return box and confidence."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    candidates = []
    for index, raw_box in enumerate(boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(float, raw_box)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(boxes.conf[index].item())
        candidates.append((confidence, (x1, y1, x2, y2)))
    if not candidates:
        return None
    confidence, box = max(candidates, key=lambda item: item[0])
    return {"box": box, "confidence": confidence}
