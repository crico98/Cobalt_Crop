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

from crop_pipeline import (
    PROVISIONAL_CROP_COVERAGE,
    centered_detection_image,
    crop_square,
    detection_region,
    map_box_to_full_image,
    square_crop_geometry,
)
from image_orientation import normalize_uploaded_image


BATCH_DIR = PROJECT_ROOT / "work" / "crop_test_batch"
DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "diagnostics" / "box_fusion_comparison"
BASELINE_IOU = 0.70
TEST_IOU = 0.60
FUSION_CANDIDATE_IOU = 0.99
FUSION_CLUSTER_IOU = 0.60


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


def extract_candidates(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    found = []
    for index, raw in enumerate(boxes.xyxy.tolist()):
        box = tuple(map(float, raw))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        found.append({
            "candidate": f"C{len(found) + 1}",
            "confidence": float(boxes.conf[index].item()),
            "box": box,
        })
    return found


def fuse_candidates(candidates, iou_threshold):
    """Single-model confidence-weighted fusion; greedy clusters by seed IoU."""
    ordered = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    clusters = []
    for candidate in ordered:
        best_cluster = None
        best_iou = 0.0
        for cluster in clusters:
            overlap = box_iou(candidate["box"], cluster["box"])
            if overlap >= iou_threshold and overlap > best_iou:
                best_cluster, best_iou = cluster, overlap
        if best_cluster is None:
            clusters.append({
                "members": [candidate],
                "box": candidate["box"],
            })
        else:
            best_cluster["members"].append(candidate)
            best_cluster["box"] = weighted_box(best_cluster["members"])

    fused = []
    for index, cluster in enumerate(clusters, start=1):
        members = cluster["members"]
        total_weight = sum(max(0.0, item["confidence"]) for item in members)
        if total_weight <= 0:
            continue
        fused.append({
            "candidate": f"F{index}",
            "confidence": sum(item["confidence"] for item in members) / len(members),
            "box": weighted_box(members),
            "member_count": len(members),
            "members": ",".join(item["candidate"] for item in members),
        })
    return fused


def weighted_box(members):
    weights = [max(0.0, item["confidence"]) for item in members]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(members)
        total = float(len(members))
    return tuple(
        sum(item["box"][axis] * weight for item, weight in zip(members, weights)) / total
        for axis in range(4)
    )


def choose_from_candidates(candidates, image_size):
    """Mirror crop_pipeline.choose_detection's 3pp confidence/center rule."""
    if not candidates:
        return None
    highest = max(item["confidence"] for item in candidates)
    tied = [
        item for item in candidates
        if highest - item["confidence"] <= 0.03
    ]
    image_width, image_height = image_size
    cx, cy = image_width / 2.0, image_height / 2.0

    def key(item):
        x1, y1, x2, y2 = item["box"]
        bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (bx - cx) ** 2 + (by - cy) ** 2, -item["confidence"]

    return min(tied, key=key)


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_method_overlay(image, offset, region_size, candidates, selected, method, output_path):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(24, image.width // 170))
    left, top = offset
    rw, rh = region_size
    width = max(8, image.width // 550)
    draw.rectangle((left, top, left + rw, top + rh), outline=(255, 190, 0), width=width)
    draw.text(
        (left + 16, top + 16),
        f"{method} | candidates={len(candidates)} | selected={selected['candidate'] if selected else 'NONE'}",
        fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0),
    )
    for item in candidates:
        x1, y1, x2, y2 = item["box"]
        coords = (x1 + left, y1 + top, x2 + left, y2 + top)
        is_selected = selected is not None and item is selected
        color = (30, 220, 100) if is_selected else (235, 80, 80)
        draw.rectangle(coords, outline=color, width=width)
        label = f"{item['candidate']} conf={item['confidence']:.3f}"
        if item.get("member_count"):
            label += f" n={item['member_count']} [{item['members']}]"
        label_y = max(0, int(coords[1] - font.size - 8))
        draw.rectangle(
            (coords[0], label_y, coords[0] + draw.textlength(label, font=font) + 12,
             label_y + font.size + 8), fill=(20, 20, 20),
        )
        draw.text((coords[0] + 6, label_y + 3), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def save_crop(image, offset, selected, view, method, output_path):
    if selected is None:
        return None
    full_box = map_box_to_full_image(selected["box"], offset)
    crop_box = square_crop_geometry(
        image.size, full_box, PROVISIONAL_CROP_COVERAGE[view]
    )
    crop = crop_square(image, crop_box, OUTPUT_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="PNG")
    return full_box, crop_box


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic-only comparison: current selection, standard NMS, and confidence-weighted box fusion."
    )
    parser.add_argument("--images", nargs="*", default=None,
                        help="Optional image filenames; defaults to all Top/Isometric manifest images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nms-iou", type=float, default=TEST_IOU)
    parser.add_argument("--fusion-iou", type=float, default=FUSION_CLUSTER_IOU)
    args = parser.parse_args()

    for name, value in (("nms-iou", args.nms_iou), ("fusion-iou", args.fusion_iou)):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name} must be greater than 0 and at most 1.")
    records = json.loads((BATCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    requested = {name.casefold() for name in args.images} if args.images else None
    models = {}
    summary_rows, candidate_rows = [], []
    skipped = []

    for record in records:
        image_name = Path(record["file_path"].replace("\\", "/")).name
        if requested is not None and image_name.casefold() not in requested:
            continue
        view = {0: "isometric", 1: "top", 2: "face"}.get(int(record["angle_index"]))
        if view not in {"top", "isometric"}:
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
        rw_pct, rh_pct = detection_region(view, image)
        model_image, offset = centered_detection_image(image, rw_pct, rh_pct)

        # Independent inference calls preserve each method's own NMS behavior.
        baseline_result = models[view].predict(
            source=model_image, verbose=False, iou=BASELINE_IOU
        )[0]
        nms_result = models[view].predict(
            source=model_image, verbose=False, iou=args.nms_iou
        )[0]
        fusion_pool_result = models[view].predict(
            source=model_image, verbose=False, iou=FUSION_CANDIDATE_IOU
        )[0]

        baseline_candidates = extract_candidates(baseline_result)
        nms_candidates = extract_candidates(nms_result)
        fusion_pool = extract_candidates(fusion_pool_result)
        fused_candidates = fuse_candidates(fusion_pool, args.fusion_iou)

        methods = {
            "baseline": (baseline_candidates, choose_from_candidates(baseline_candidates, model_image.size)),
            "nms": (nms_candidates, choose_from_candidates(nms_candidates, model_image.size)),
            "wbf": (fused_candidates, choose_from_candidates(fused_candidates, model_image.size)),
        }
        selected_by_method = {}
        for method, (candidates, selected) in methods.items():
            selected_by_method[method] = selected
            method_dir = args.output_dir / method
            draw_method_overlay(
                image, offset, model_image.size, candidates, selected, method.upper(),
                method_dir / "overlays" / f"{Path(image_name).stem}_{method}.png",
            )
            crop_info = save_crop(
                image, offset, selected, view, method,
                method_dir / "crops" / f"{Path(image_name).stem}_{method}_crop.png",
            )
            full_box = crop_info[0] if crop_info else None
            crop_box = crop_info[1] if crop_info else None
            summary_rows.append({
                "image": image_name,
                "view": view,
                "method": method,
                "candidate_count": len(candidates),
                "selected_candidate": selected["candidate"] if selected else "NONE",
                "selected_confidence": f"{selected['confidence']:.6f}" if selected else "",
                "selected_box_x1": f"{full_box[0]:.2f}" if full_box else "",
                "selected_box_y1": f"{full_box[1]:.2f}" if full_box else "",
                "selected_box_x2": f"{full_box[2]:.2f}" if full_box else "",
                "selected_box_y2": f"{full_box[3]:.2f}" if full_box else "",
                "crop_left": crop_box[0] if crop_box else "",
                "crop_top": crop_box[1] if crop_box else "",
                "crop_right": crop_box[2] if crop_box else "",
                "crop_bottom": crop_box[3] if crop_box else "",
                "fusion_source_candidates": selected.get("members", "") if selected else "",
                "fusion_source_count": selected.get("member_count", 0) if selected else 0,
            })
            for candidate in candidates:
                candidate_rows.append({
                    "image": image_name,
                    "view": view,
                    "method": method,
                    "candidate": candidate["candidate"],
                    "confidence": f"{candidate['confidence']:.6f}",
                    "box_x1": f"{candidate['box'][0]:.2f}",
                    "box_y1": f"{candidate['box'][1]:.2f}",
                    "box_x2": f"{candidate['box'][2]:.2f}",
                    "box_y2": f"{candidate['box'][3]:.2f}",
                    "fusion_members": candidate.get("members", ""),
                    "fusion_member_count": candidate.get("member_count", 0),
                })

        selections = selected_by_method
        base, nms, wbf = selections["baseline"], selections["nms"], selections["wbf"]
        print(
            f"{image_name}: candidates baseline={len(baseline_candidates)}, "
            f"NMS={len(nms_candidates)}, fusion pool={len(fusion_pool)}, "
            f"fused={len(fused_candidates)} | selected "
            f"baseline={base['candidate'] if base else 'NONE'}, "
            f"NMS={nms['candidate'] if nms else 'NONE'}, "
            f"WBF={wbf['candidate'] if wbf else 'NONE'}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "comparison_summary.csv"
    candidates_path = args.output_dir / "candidate_metrics.csv"
    summary_fields = list(dict.fromkeys(key for row in summary_rows for key in row))
    candidate_fields = [
        "image", "view", "method", "candidate", "confidence",
        "box_x1", "box_y1", "box_x2", "box_y2",
        "fusion_members", "fusion_member_count",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        if summary_fields:
            writer.writeheader()
            writer.writerows(summary_rows)
    with candidates_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    print("\nBOX FUSION COMPARISON COMPLETE")
    print(f"Images analyzed: {len({row['image'] for row in summary_rows})}")
    print(f"Summary: {summary_path}")
    print(f"Candidate metrics: {candidates_path}")
    print(f"Outputs: {args.output_dir}")
    for name, reason in skipped:
        print(f"Skipped {name}: {reason}")
    if requested is not None:
        found = {row["image"].casefold() for row in summary_rows}
        missing = requested - found
        if missing:
            print("Requested images not analyzed: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
