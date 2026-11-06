from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from crop_pipeline import (
    centered_detection_image,
    detection_region,
    map_box_to_full_image,
)
from image_orientation import normalize_uploaded_image


IMAGE_NAME = "DSC00019_22158e35.jpg"
COLORS = [(220, 55, 55), (30, 170, 80), (45, 115, 220)]


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(
        description="Draw Top/Isometric detector candidates and the detection-region boundary."
    )
    parser.add_argument("--image", default=IMAGE_NAME)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    batch_dir = root / "work" / "crop_test_batch"
    image_path = batch_dir / args.image
    manifest_path = batch_dir / "manifest.json"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        r for r in records
        if Path(r["file_path"].replace("\\", "/")).name.casefold()
        == args.image.casefold()
    )
    angle = int(record["angle_index"])
    view = {0: "isometric", 1: "top", 2: "face"}.get(angle)
    if view not in {"isometric", "top"}:
        raise ValueError(f"{args.image} is assigned to unsupported view: {view}")

    model_path = root / "models" / f"{view}_view" / "weights" / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    with Image.open(image_path) as source:
        image = normalize_uploaded_image(source)

    region_width, region_height = detection_region(view, image)
    model_image, offset = centered_detection_image(image, region_width, region_height)
    model = YOLO(str(model_path))
    result = model.predict(source=model_image, verbose=False)[0]
    boxes = result.boxes

    print(f"Image: {args.image} | view: {view} | normalized: {image.width}x{image.height}")
    print(f"Detection region: {model_image.width}x{model_image.height} at offset {offset}")

    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    large_font = load_font(max(28, image.width // 150))
    small_font = load_font(max(22, image.width // 190))

    left, top = offset
    region_box = (left, top, left + model_image.width, top + model_image.height)
    draw.rectangle(region_box, outline=(255, 190, 0), width=max(10, image.width // 600))
    draw.text((left + 18, top + 18), "MODEL DETECTION REGION", fill=(255, 190, 0), font=large_font)

    if boxes is None or len(boxes) == 0:
        print("NO DETECTIONS")
    else:
        image_cx, image_cy = image.width / 2, image.height / 2
        diagonal = (image.width**2 + image.height**2) ** 0.5
        for index, raw in enumerate(boxes.xyxy.tolist()):
            x1, y1, x2, y2 = map(float, raw)
            confidence = float(boxes.conf[index].item())
            full_box = map_box_to_full_image((x1, y1, x2, y2), offset)
            fx1, fy1, fx2, fy2 = full_box
            color = COLORS[index % len(COLORS)]
            width = max(12, image.width // 450)
            draw.rectangle(full_box, outline=color, width=width)
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            radius = max(18, image.width // 180)
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=width)
            center_distance = ((cx-image_cx)**2 + (cy-image_cy)**2)**0.5 / diagonal
            edge_margin = min(x1, y1, model_image.width-x2, model_image.height-y2)
            label = f"C{index+1}  conf={confidence:.3f}  center={center_distance:.4f}  edge={edge_margin:.0f}px"
            label_y = max(0, fy1 - large_font.size - 10)
            draw.rectangle(
                (fx1, label_y, fx1 + draw.textlength(label, font=small_font) + 24,
                 label_y + small_font.size + 16),
                fill=(20, 20, 20),
            )
            draw.text((fx1 + 10, label_y + 5), label, fill=color, font=small_font)
            print(
                f"Candidate {index+1}: confidence={confidence:.4f}, "
                f"box={[round(v, 1) for v in full_box]}, "
                f"center_distance={center_distance:.4f}, "
                f"detection_region_edge_margin={edge_margin:.1f}px"
            )

    output_path = Path(args.output) if args.output else (
        root / "work" / "diagnostics" / f"{Path(args.image).stem}_candidates.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, format="PNG")
    print(f"Overlay saved: {output_path}")


if __name__ == "__main__":
    main()
