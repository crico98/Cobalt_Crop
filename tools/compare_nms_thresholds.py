from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from crop_pipeline import centered_detection_image, detection_region
from image_orientation import normalize_uploaded_image


BATCH_DIR = PROJECT_ROOT / "work" / "crop_test_batch"
DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "diagnostics" / "nms_comparison"
BASELINE_IOU = 0.70
TEST_IOU = 0.60


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def extract_boxes(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    found = []
    for index, raw_box in enumerate(boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(float, raw_box)
        if x2 <= x1 or y2 <= y1:
            continue
        found.append({
            "candidate": f"C{len(found) + 1}",
            "confidence": float(boxes.conf[index].item()),
            "box": (x1, y1, x2, y2),
        })
    return found


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def save_overlay(image, offset, region_size, boxes, image_name, iou, output_dir):
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = load_font(max(24, image.width // 170))
    left, top = offset
    rw, rh = region_size
    line_width = max(8, image.width // 550)
    draw.rectangle((left, top, left + rw, top + rh),
                   outline=(255, 190, 0), width=line_width)
    draw.text((left + 16, top + 16),
              f"NMS IoU={iou:.2f} | detections={len(boxes)}",
              fill=(255, 255, 255), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        coords = (x1 + left, y1 + top, x2 + left, y2 + top)
        color = (30, 210, 100) if item["candidate"] == "C1" else (235, 80, 80)
        draw.rectangle(coords, outline=color, width=line_width)
        label = f"{item['candidate']} conf={item['confidence']:.3f}"
        label_y = max(0, int(coords[1] - font.size - 8))
        draw.rectangle((coords[0], label_y,
                        coords[0] + draw.textlength(label, font=font) + 12,
                        label_y + font.size + 8), fill=(20, 20, 20))
        draw.text((coords[0] + 6, label_y + 3), label, fill=color, font=font)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{Path(image_name).stem}_nms_{iou:.2f}.png"
    overlay.save(path, format="PNG")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic-only comparison of default and lower YOLO NMS IoU."
    )
    parser.add_argument("--images", nargs="*", default=None,
                        help="Image filenames; defaults to all Top/Isometric manifest images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-iou", type=float, default=TEST_IOU)
    args = parser.parse_args()
    if not 0.0 < args.test_iou <= 1.0:
        raise ValueError("--test-iou must be greater than 0 and at most 1.")
    if abs(args.test_iou - BASELINE_IOU) < 1e-9:
        raise ValueError("--test-iou must differ from the baseline IoU of 0.70.")

    records = json.loads((BATCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    requested = {name.casefold() for name in args.images} if args.images else None
    models = {}
    summary = []
    candidate_rows = []
    skipped = []

    for record in records:
        image_name = Path(record["file_path"].replace("\\", "/")).name
        if requested is not None and image_name.casefold() not in requested:
            continue
        view = {0: "isometric", 1: "top", 2: "face"}.get(int(record["angle_index"]))
        if view not in {"isometric", "top"}:
            skipped.append((image_name, f"unsupported view: {view}"))
            continue
        image_path = BATCH_DIR / image_name
        if not image_path.is_file():
            skipped.append((image_name, "image file missing"))
            continue

        model_path = PROJECT_ROOT / "models" / f"{view}_view" / "weights" / "best.pt"
        if view not in models:
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            models[view] = YOLO(str(model_path))

        with Image.open(image_path) as source:
            image = normalize_uploaded_image(source)
        width_pct, height_pct = detection_region(view, image)
        model_image, offset = centered_detection_image(image, width_pct, height_pct)

        per_threshold = {}
        for nms_iou in (BASELINE_IOU, args.test_iou):
            result = models[view].predict(
                source=model_image, verbose=False, iou=nms_iou
            )[0]
            boxes = extract_boxes(result)
            per_threshold[nms_iou] = boxes
            save_overlay(image, offset, model_image.size, boxes, image_name,
                         nms_iou, args.output_dir / "overlays")
            for item in boxes:
                x1, y1, x2, y2 = item["box"]
                candidate_rows.append({
                    "image": image_name,
                    "view": view,
                    "nms_iou": f"{nms_iou:.2f}",
                    "candidate": item["candidate"],
                    "confidence": f"{item['confidence']:.6f}",
                    "box_x1": f"{x1:.2f}",
                    "box_y1": f"{y1:.2f}",
                    "box_x2": f"{x2:.2f}",
                    "box_y2": f"{y2:.2f}",
                })

        base_boxes = per_threshold[BASELINE_IOU]
        test_boxes = per_threshold[args.test_iou]
        summary.append({
            "image": image_name,
            "view": view,
            "baseline_iou": f"{BASELINE_IOU:.2f}",
            "baseline_detection_count": len(base_boxes),
            "test_iou": f"{args.test_iou:.2f}",
            "test_detection_count": len(test_boxes),
            "detections_removed": len(base_boxes) - len(test_boxes),
            "highest_confidence_baseline": (
                f"{max((b['confidence'] for b in base_boxes), default=0.0):.6f}"
            ),
            "highest_confidence_test": (
                f"{max((b['confidence'] for b in test_boxes), default=0.0):.6f}"
            ),
            "best_baseline_test_box_iou": f"{max((
                box_iou(a['box'], b['box'])
                for a in base_boxes for b in test_boxes
            ), default=0.0):.6f}",
        })
        print(
            f"{image_name}: IoU {BASELINE_IOU:.2f} -> {len(base_boxes)} detections; "
            f"IoU {args.test_iou:.2f} -> {len(test_boxes)} detections"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "nms_summary.csv"
    candidates_path = args.output_dir / "nms_candidate_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(summary[0]) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(summary)
    with candidates_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image", "view", "nms_iou", "candidate", "confidence",
                  "box_x1", "box_y1", "box_x2", "box_y2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    print("\nNMS COMPARISON COMPLETE")
    print(f"Images analyzed: {len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Candidate metrics: {candidates_path}")
    print(f"Overlays: {args.output_dir / 'overlays'}")
    for name, reason in skipped:
        print(f"Skipped {name}: {reason}")
    if requested is not None:
        found = {row["image"].casefold() for row in summary}
        missing = requested - found
        if missing:
            print("Requested images not analyzed: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
