from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "training_presentation"
DEFAULT_OUTPUT = ROOT / "training_presentation_exif_normalized"
DEFAULT_ANNOTATIONS = ROOT / "dataset" / "top_isometric_annotations.json"
DEFAULT_ROTATIONS = ROOT / "dataset" / "training_rotation_manifest.json"
VALID_ROTATIONS = {0, 90, 180, 270}


def safe_stem(relative_path: str) -> str:
    path = Path(relative_path)
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", path.with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:90] or 'image'}_{digest}"


def annotation_records(manifest: dict) -> dict[str, str]:
    records = {}
    annotations = manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("Annotation manifest must contain an annotations object.")
    for key, record in annotations.items():
        view = record.get("view")
        if view == "top" and record.get("target_version") != 2:
            continue
        if view == "isometric" and record.get("target_version") not in (None, 1, 2):
            continue
        if view not in {"top", "isometric"} or not record.get("boxes"):
            continue
        records[key] = view
    if not records:
        raise ValueError("No current Top or Isometric box annotations were found.")
    return records


def write_rotation_template(records: dict[str, str], path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Rotation template already exists: {path}. Use --overwrite-template to replace it."
        )
    payload = {
        "version": 1,
        "rotation_convention": "Clockwise degrees applied after EXIF normalization; use 0, 90, 180, or 270.",
        "images": {
            key: {"manual_rotation_degrees": None}
            for key in sorted(records)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_rotation_manifest(path: Path, expected_keys: set[str]) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Rotation manifest not found: {path}. First run with --write-rotation-template."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if payload.get("version") != 1 or not isinstance(images, dict):
        raise ValueError("Rotation manifest must use version 1 and contain an images object.")

    missing = sorted(expected_keys - set(images))
    extra = sorted(set(images) - expected_keys)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing image classification(s): {missing[:8]}")
        if extra:
            parts.append(f"{len(extra)} unknown image key(s): {extra[:8]}")
        raise ValueError("Rotation manifest does not match the annotation set: " + "; ".join(parts))

    rotations = {}
    incomplete = []
    for key in sorted(expected_keys):
        entry = images[key]
        value = entry.get("manual_rotation_degrees") if isinstance(entry, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value not in VALID_ROTATIONS:
            incomplete.append(key)
        else:
            rotations[key] = value
    if incomplete:
        raise ValueError(
            "Set manual_rotation_degrees to 0, 90, 180, or 270 for every image. "
            f"Unclassified examples: {incomplete[:10]}"
        )
    return rotations


def rotate_box(box: tuple[float, float, float, float], degrees: int):
    x1, y1, x2, y2 = box
    if degrees == 0:
        return x1, y1, x2, y2
    if degrees == 90:
        return 1.0 - y2, x1, 1.0 - y1, x2
    if degrees == 180:
        return 1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1
    if degrees == 270:
        return y1, 1.0 - x2, y2, 1.0 - x1
    raise ValueError(f"Unsupported clockwise rotation: {degrees}")


def expand_box(box: tuple[float, float, float, float], rotated: bool):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = x2 - x1, y2 - y1
    width_factor = 1.0 if rotated else 1.05
    height_factor = 1.20 if rotated else 1.05

    # Limit expansion symmetrically so the original center never shifts.
    max_width = 2.0 * min(cx, 1.0 - cx)
    max_height = 2.0 * min(cy, 1.0 - cy)
    width = min(width * width_factor, max_width)
    height = min(height * height_factor, max_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot expand box while preserving its center: {box}")
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


def read_yolo_labels(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO detection fields.")
        class_id = int(parts[0])
        xc, yc, width, height = map(float, parts[1:])
        if class_id != 0 or not all(map(lambda value: 0.0 <= value <= 1.0, (xc, yc, width, height))):
            raise ValueError(f"{path}:{line_number}: invalid class or normalized coordinates.")
        rows.append((class_id, (xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2)))
    if not rows:
        raise ValueError(f"Empty YOLO label file: {path}")
    return rows


def find_export_pair(source_root: Path, view: str, stem: str):
    matches = []
    for split in ("train", "val"):
        image_dir = source_root / view / "images" / split
        matches.extend((split, path) for path in image_dir.glob(f"{stem}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one exported image for {view}/{stem}; found {len(matches)}."
        )
    split, image_path = matches[0]
    label_path = source_root / view / "labels" / split / f"{stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Matching YOLO label is missing: {label_path}")
    return split, image_path, label_path


def prepare_dataset(source_root: Path, output_root: Path, annotations_path: Path,
                    rotations_path: Path, overwrite: bool = False):
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source YOLO export not found: {source_root}")
    annotation_manifest = json.loads(annotations_path.read_text(encoding="utf-8"))
    records = annotation_records(annotation_manifest)
    rotations = load_rotation_manifest(rotations_path, set(records))

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite to rebuild this derived dataset."
            )
        shutil.rmtree(output_root)

    summary = {
        "source": str(source_root),
        "annotations": str(annotations_path),
        "rotation_manifest": str(rotations_path),
        "rotation_degrees": dict(Counter(map(str, rotations.values()))),
        "expansion": {
            "not_rotated": {"width_total": 1.05, "height_total": 1.05},
            "rotated_after_exif": {"width_total": 1.0, "height_total": 1.20},
        },
        "views": {},
    }

    for view in ("top", "isometric"):
        view_keys = [key for key, assigned_view in records.items() if assigned_view == view]
        exported_count = 0
        for key in sorted(view_keys):
            stem = safe_stem(key)
            split, source_image_path, source_label_path = find_export_pair(source_root, view, stem)
            rotation = rotations[key]
            label_rows = read_yolo_labels(source_label_path)

            with Image.open(source_image_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
            if rotation:
                image = image.rotate(-rotation, expand=True)

            output_image_dir = output_root / view / "images" / split
            output_label_dir = output_root / view / "labels" / split
            output_image_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir.mkdir(parents=True, exist_ok=True)
            image.save(output_image_dir / f"{stem}.png", format="PNG", optimize=True)

            lines = []
            for class_id, original_box in label_rows:
                rotated_box = rotate_box(original_box, rotation)
                x1, y1, x2, y2 = expand_box(rotated_box, rotated=rotation != 0)
                xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                width, height = x2 - x1, y2 - y1
                lines.append(f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
            (output_label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            exported_count += 1

        target_name = "full_cutter" if view == "top" else "diamond_table"
        yaml = (
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"  0: {target_name}\n"
        )
        view_dir = output_root / view
        view_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "data.yaml").write_text(yaml, encoding="utf-8")
        summary["views"][view] = {"images": exported_count, "class_0": target_name}

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared orientation-normalized datasets: {output_root.resolve()}")
    for view, details in summary["views"].items():
        print(f"  {view}: {details['images']} images")
    print("Original annotations and training_presentation exports were read only.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a separate EXIF-normalized Top/Isometric YOLO dataset with derived box expansion."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--rotations", type=Path, default=DEFAULT_ROTATIONS)
    parser.add_argument("--write-rotation-template", action="store_true")
    parser.add_argument("--overwrite-template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.write_rotation_template:
        manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
        records = annotation_records(manifest)
        write_rotation_template(records, args.rotations, overwrite=args.overwrite_template)
        print(f"Created rotation template for {len(records)} annotated images: {args.rotations}")
        print("Classify every image with 0, 90, 180, or 270 clockwise degrees after EXIF normalization.")
        return

    prepare_dataset(
        args.source, args.output, args.annotations, args.rotations,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
