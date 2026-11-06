from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from crop_pipeline import (
    DETECTION_CONFIDENCE_TOLERANCE,
    centered_detection_image,
    detection_region,
)
from image_orientation import normalize_uploaded_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "work" / "crop_test_batch"
DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "diagnostics" / "selection_comparison"
WEIGHT_CENTER = 0.70
WEIGHT_CONFIDENCE_PENALTY = 0.30


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def choose(candidates, key):
    if not candidates:
        return None
    return min(candidates, key=key)["candidate"]


def analyze_candidates(result, image_size, region_size, confidence_floor):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    image_width, image_height = image_size
    region_width, region_height = region_size
    image_cx, image_cy = image_width / 2.0, image_height / 2.0
    diagonal = (image_width**2 + image_height**2) ** 0.5
    candidates = []

    for index, raw_box in enumerate(boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(float, raw_box)
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(boxes.conf[index].item())
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center_distance = (
            ((cx - region_width / 2.0) ** 2 + (cy - region_height / 2.0) ** 2) ** 0.5
            / diagonal
        )
        edge_margin = min(x1, y1, region_width - x2, region_height - y2)

        candidates.append({
            "candidate": f"C{index + 1}",
            "confidence": confidence,
            "center_distance": center_distance,
            "edge_margin_px": edge_margin,
            "weighted_score": (
                WEIGHT_CENTER * center_distance
                + WEIGHT_CONFIDENCE_PENALTY * (1.0 - confidence)
            ),
            "box_region": (x1, y1, x2, y2),
        })

    if not candidates:
        return []

    highest_confidence = max(item["confidence"] for item in candidates)
    within_current_tolerance = [
        item for item in candidates
        if highest_confidence - item["confidence"] <= DETECTION_CONFIDENCE_TOLERANCE
    ]
    above_floor = [
        item for item in candidates if item["confidence"] >= confidence_floor
    ]

    selections = {
        "current_3pp": choose(
            within_current_tolerance,
            lambda item: (item["center_distance"], -item["confidence"]),
        ),
        "center_first": choose(
            candidates,
            lambda item: (item["center_distance"], -item["confidence"]),
        ),
        "weighted_70_30": choose(
            candidates,
            lambda item: (item["weighted_score"], item["center_distance"], -item["confidence"]),
        ),
        "center_with_confidence_floor": choose(
            above_floor,
            lambda item: (item["center_distance"], -item["confidence"]),
        ),
    }

    for item in candidates:
        item.update({
            "selected_current_3pp": item["candidate"] == selections["current_3pp"],
            "selected_center_first": item["candidate"] == selections["center_first"],
            "selected_weighted_70_30": item["candidate"] == selections["weighted_70_30"],
            "selected_center_with_confidence_floor": (
                item["candidate"] == selections["center_with_confidence_floor"]
            ),
        })

    # Attach the per-image selections to every candidate row for convenient CSV review.
    for item in candidates:
        item["selections"] = selections

    return candidates


def save_disagreement_overlay(image, offset, region_size, candidates, image_name, output_dir):
    selections = candidates[0]["selections"] if candidates else {}
    selected_values = set(value for value in selections.values() if value is not None)
    if len(set(selections.values())) <= 1:
        return None

    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    large_font = load_font(max(28, image.width // 150))
    small_font = load_font(max(22, image.width // 190))

    left, top = offset
    region_width, region_height = region_size
    draw.rectangle(
        (left, top, left + region_width, top + region_height),
        outline=(255, 190, 0),
        width=max(10, image.width // 600),
    )

    legend = [
        f"Current 3pp: {selections['current_3pp'] or 'NONE'}",
        f"Center first: {selections['center_first'] or 'NONE'}",
        f"Weighted 70/30: {selections['weighted_70_30'] or 'NONE'}",
        f"Center + conf floor: {selections['center_with_confidence_floor'] or 'NONE'}",
    ]
    y = top + 18
    for line in legend:
        draw.text((left + 18, y), line, fill=(255, 255, 255), font=large_font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        y += large_font.size + 8

    for index, item in enumerate(candidates):
        x1, y1, x2, y2 = item["box_region"]
        full_box = (x1 + left, y1 + top, x2 + left, y2 + top)
        color = (30, 190, 90) if item["candidate"] in selected_values else (220, 70, 70)
        width = max(12, image.width // 450)
        draw.rectangle(full_box, outline=color, width=width)
        label = (
            f"{item['candidate']} conf={item['confidence']:.3f} "
            f"center={item['center_distance']:.4f} "
            f"edge={item['edge_margin_px']:.0f}px "
            f"score={item['weighted_score']:.4f}"
        )
        label_y = max(0, full_box[1] - small_font.size - 10)
        draw.rectangle(
            (full_box[0], label_y,
             full_box[0] + draw.textlength(label, font=small_font) + 20,
             label_y + small_font.size + 14),
            fill=(20, 20, 20),
        )
        draw.text((full_box[0] + 8, label_y + 4), label, fill=color, font=small_font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_name).stem}_selection_comparison.png"
    overlay.save(output_path, format="PNG")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare diagnostic-only Top/Isometric candidate selection strategies."
    )
    parser.add_argument("--confidence-floor", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Optional image filenames. Defaults to every image in the batch manifest.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence_floor <= 1.0:
        raise ValueError("--confidence-floor must be between 0 and 1.")

    manifest_path = BATCH_DIR / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = {name.casefold() for name in args.images} if args.images else None

    models = {}
    candidate_rows = []
    summary_rows = []
    skipped = []

    for record in records:
        image_name = Path(record["file_path"].replace("\\", "/")).name
        if requested is not None and image_name.casefold() not in requested:
            continue

        angle = int(record["angle_index"])
        view = {0: "isometric", 1: "top", 2: "face"}.get(angle)
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

        region_width_percent, region_height_percent = detection_region(view, image)
        model_image, offset = centered_detection_image(
            image, region_width_percent, region_height_percent
        )
        result = models[view].predict(source=model_image, verbose=False)[0]

        candidates = analyze_candidates(
            result,
            image.size,
            model_image.size,
            args.confidence_floor,
        )
        selections = candidates[0]["selections"] if candidates else {
            "current_3pp": None,
            "center_first": None,
            "weighted_70_30": None,
            "center_with_confidence_floor": None,
        }

        summary_rows.append({
            "image": image_name,
            "view": view,
            "candidate_count": len(candidates),
            "current_3pp": selections["current_3pp"] or "NONE",
            "center_first": selections["center_first"] or "NONE",
            "weighted_70_30": selections["weighted_70_30"] or "NONE",
            "center_with_confidence_floor": (
                selections["center_with_confidence_floor"] or "NONE"
            ),
            "strategies_disagree": len(set(selections.values())) > 1,
        })

        for item in candidates:
            candidate_rows.append({
                "image": image_name,
                "view": view,
                "candidate": item["candidate"],
                "confidence": f"{item['confidence']:.6f}",
                "center_distance": f"{item['center_distance']:.6f}",
                "edge_margin_px": f"{item['edge_margin_px']:.1f}",
                "weighted_score_70_30": f"{item['weighted_score']:.6f}",
                "selected_current_3pp": item["selected_current_3pp"],
                "selected_center_first": item["selected_center_first"],
                "selected_weighted_70_30": item["selected_weighted_70_30"],
                "selected_center_with_confidence_floor": (
                    item["selected_center_with_confidence_floor"]
                ),
            })

        if candidates:
            save_disagreement_overlay(
                image, offset, model_image.size, candidates, image_name,
                args.output_dir / "overlays",
            )

        print(
            f"{image_name}: {len(candidates)} candidates | "
            f"current={selections['current_3pp'] or 'NONE'} | "
            f"center={selections['center_first'] or 'NONE'} | "
            f"weighted={selections['weighted_70_30'] or 'NONE'} | "
            f"center+floor={selections['center_with_confidence_floor'] or 'NONE'}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_metrics.csv"
    summary_path = args.output_dir / "selection_summary.csv"

    candidate_fields = [
        "image", "view", "candidate", "confidence", "center_distance",
        "edge_margin_px", "weighted_score_70_30", "selected_current_3pp",
        "selected_center_first", "selected_weighted_70_30",
        "selected_center_with_confidence_floor",
    ]
    summary_fields = [
        "image", "view", "candidate_count", "current_3pp", "center_first",
        "weighted_70_30", "center_with_confidence_floor", "strategies_disagree",
    ]

    with candidate_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    disagreement_count = sum(row["strategies_disagree"] for row in summary_rows)
    print("\nCOMPARISON COMPLETE")
    print(f"Images analyzed: {len(summary_rows)}")
    print(f"Images with strategy disagreements: {disagreement_count}")
    print(f"Confidence floor (diagnostic only): {args.confidence_floor:.2f}")
    print(f"Candidate metrics: {candidate_path}")
    print(f"Selection summary: {summary_path}")
    print(f"Disagreement overlays: {args.output_dir / 'overlays'}")
    if skipped:
        print(f"Skipped records: {len(skipped)}")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    if requested is not None:
        found = {row["image"].casefold() for row in summary_rows}
        missing = requested - found
        if missing:
            print("Requested images not analyzed: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
