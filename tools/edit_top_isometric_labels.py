from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = ROOT / "dataset" / "train" / "images"
DEFAULT_OUTPUT_DIR = ROOT / "training"
MANIFEST_PATH = ROOT / "dataset" / "top_isometric_annotations.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_FACE_CLASS_ID = 0
WINDOW_PREFIX = "Top / Isometric Diamond Face Box Editor"


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 2, "annotations": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read annotation manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), dict):
        raise RuntimeError(f"Invalid annotation manifest: {path}")
    return data


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        (p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: relative_key(p, image_dir).lower(),
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return images


def legacy_polygons_to_boxes(polygons):
    """Convert older saved polygon annotations to enclosing boxes for continuity."""
    boxes = []
    for polygon in polygons or []:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        if x2 > x1 and y2 > y1:
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


class Editor:
    def __init__(self, image, saved_boxes, view):
        self.original = image
        self.saved_boxes = [list(map(float, box)) for box in saved_boxes]
        self.initial_boxes = [box.copy() for box in self.saved_boxes]
        self.view = view if view in {"top", "isometric"} else ""
        self.h, self.w = image.shape[:2]
        scale = min(1400 / self.w, 900 / self.h, 1.0)
        self.display_w = max(1, int(round(self.w * scale)))
        self.display_h = max(1, int(round(self.h * scale)))
        self.drag_start = None
        self.drag_end = None
        self.drawing = False

    def to_normalized(self, x, y):
        return [float(np.clip(x / self.display_w, 0, 1)),
                float(np.clip(y / self.display_h, 0, 1))]

    def to_display(self, box):
        x1, y1, x2, y2 = box
        return (int(round(x1 * self.display_w)), int(round(y1 * self.display_h)),
                int(round(x2 * self.display_w)), int(round(y2 * self.display_h)))

    def add_box(self, start, end):
        x1, y1 = self.to_normalized(*start)
        x2, y2 = self.to_normalized(*end)
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if box[2] - box[0] >= 0.003 and box[3] - box[1] >= 0.003:
            self.saved_boxes.append(box)

    def delete_nearest_box(self, x, y):
        if not self.saved_boxes:
            return
        candidates = []
        for index, box in enumerate(self.saved_boxes):
            x1, y1, x2, y2 = self.to_display(box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            candidates.append(((cx - x) ** 2 + (cy - y) ** 2, index, (x1, y1, x2, y2)))
        distance, index, rect = min(candidates)
        x1, y1, x2, y2 = rect
        inside = x1 <= x <= x2 and y1 <= y <= y2
        if inside or distance <= 28 ** 2:
            self.saved_boxes.pop(index)

    def render(self, number, total, image_name):
        image = cv2.resize(
            self.original, (self.display_w, self.display_h),
            interpolation=cv2.INTER_AREA if self.display_w < self.w else cv2.INTER_LINEAR,
        )
        for box in self.saved_boxes:
            x1, y1, x2, y2 = self.to_display(box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(image, ((x1 + x2) // 2, (y1 + y2) // 2), 4, (0, 0, 255), -1)
        if self.drawing and self.drag_start and self.drag_end:
            cv2.rectangle(image, self.drag_start, self.drag_end, (0, 255, 255), 2, cv2.LINE_AA)

        view_label = self.view.upper() if self.view else "UNASSIGNED"
        lines = [
            f"{number}/{total}  {image_name}  |  VIEW: {view_label}  |  BOXES: {len(self.saved_boxes)}",
            ("Top: box the entire visible cutter, including all carbide and diamond table"
             if self.view == "top" else
             "Isometric: box the diamond table only; exclude carbide body and pocket"
             if self.view == "isometric" else
             "Assign T or I, then draw the target box using that view's rules"),
            "Right-click = delete nearest box | U=undo last box | C=clear | R=reset",
            "T=Top | I=Isometric | S=save+next | N=skip | P=previous | Q/Esc=quit",
        ]
        header_h = 88
        cv2.rectangle(image, (0, 0), (self.display_w, header_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(image, line[:180], (10, 21 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48 if i else 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return image


def edit_one(image_path, image_root, manifest, manifest_path, number, total):
    key = relative_key(image_path, image_root)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: could not read {image_path}; skipping.")
        return "next"

    record = manifest["annotations"].get(key, {})
    boxes = record.get("boxes")
    if boxes is None:
        boxes = legacy_polygons_to_boxes(record.get("polygons", []))
        if boxes:
            print("Converted older polygon annotation(s) to enclosing bounding box(es).")

    editor = Editor(image, boxes, record.get("view", ""))
    window = f"{WINDOW_PREFIX} - {image_path.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, editor.display_w, editor.display_h)

    def mouse_callback(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.drag_start = (x, y)
            editor.drag_end = (x, y)
            editor.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and editor.drawing:
            editor.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and editor.drawing:
            editor.drag_end = (x, y)
            editor.add_box(editor.drag_start, editor.drag_end)
            editor.drag_start = editor.drag_end = None
            editor.drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.delete_nearest_box(x, y)

    cv2.setMouseCallback(window, mouse_callback)
    print(f"\n[{number}/{total}] {key}")
    outcome = "quit"
    try:
        while True:
            cv2.imshow(window, editor.render(number, total, image_path.name))
            keycode = cv2.waitKeyEx(20)
            if keycode == -1:
                continue
            keycode &= 0xFF
            if keycode in (ord("t"), ord("T")):
                editor.view = "top"
            elif keycode in (ord("i"), ord("I")):
                editor.view = "isometric"
            elif keycode in (ord("u"), ord("U")):
                if editor.saved_boxes:
                    editor.saved_boxes.pop()
            elif keycode in (ord("c"), ord("C")):
                editor.saved_boxes = []
            elif keycode in (ord("r"), ord("R")):
                editor.saved_boxes = [box.copy() for box in editor.initial_boxes]
                editor.view = record.get("view", "")
                editor.drawing = False
                editor.drag_start = editor.drag_end = None
            elif keycode in (ord("s"), ord("S")):
                if editor.drawing:
                    print("Finish drawing the box before saving.")
                    continue
                if not editor.saved_boxes:
                    print("No diamond-face box to save. Press N to skip this image.")
                    continue
                if editor.view not in {"top", "isometric"}:
                    print("Assign this image with T (Top) or I (Isometric) before saving.")
                    continue
                manifest["annotations"][key] = {
                    "view": editor.view,
                    "boxes": editor.saved_boxes,
                }
                save_manifest(manifest_path, manifest)
                print(f"Saved {len(editor.saved_boxes)} box(es): {key} ({editor.view})")
                outcome = "next"
                break
            elif keycode in (ord("n"), ord("N"), 32):
                outcome = "next"
                break
            elif keycode in (ord("p"), ord("P")):
                outcome = "previous"
                break
            elif keycode in (ord("q"), ord("Q"), 27):
                outcome = "quit"
                break
    finally:
        cv2.destroyWindow(window)
    return outcome


def safe_stem(relative_path: str) -> str:
    path = Path(relative_path)
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", path.with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:90] or 'image'}_{digest}"


def split_groups(records, val_fraction, seed):
    if not records or val_fraction <= 0:
        return set()
    groups = {}
    for record in records:
        parent = Path(record["key"]).parent.as_posix()
        group = parent if parent != "." else Path(record["key"]).stem
        groups.setdefault(group, []).append(record)
    if len(groups) < 2:
        print("Warning: only one source folder/group is available; assigning all images to train.")
        return set()
    names = list(groups)
    random.Random(seed).shuffle(names)
    target = max(1, round(len(records) * val_fraction))
    validation_groups, count = set(), 0
    for name in names:
        if count >= target:
            break
        validation_groups.add(name)
        count += len(groups[name])
    return validation_groups


def export_datasets(image_root, output_root, manifest, manifest_path, val_fraction=0.2, seed=42):
    records, missing = [], []
    for key, annotation in manifest["annotations"].items():
        view = annotation.get("view")
        boxes = annotation.get("boxes")
        if boxes is None:
            boxes = legacy_polygons_to_boxes(annotation.get("polygons", []))
        if view not in {"top", "isometric"} or not boxes:
            continue
        image_path = image_root / Path(key)
        if not image_path.is_file():
            missing.append(key)
            continue
        records.append({"key": key, "view": view, "boxes": boxes, "image": image_path})

    if missing:
        print(f"Warning: {len(missing)} annotated source images are missing and will be skipped.")
    if not records:
        print("Nothing to export yet. Annotate a visible diamond face, assign T or I, and press S.")
        print(f"Annotation manifest: {manifest_path}")
        return

    for view in ("top", "isometric"):
        view_records = [record for record in records if record["view"] == view]
        for generated in (output_root / view / "images", output_root / view / "labels"):
            if generated.exists():
                shutil.rmtree(generated)
        val_groups = split_groups(view_records, val_fraction, seed)
        for record in view_records:
            parent = Path(record["key"]).parent.as_posix()
            group = parent if parent != "." else Path(record["key"]).stem
            split = "val" if group in val_groups else "train"
            image_dir = output_root / view / "images" / split
            label_dir = output_root / view / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = safe_stem(record["key"])
            destination_image = image_dir / f"{stem}{record['image'].suffix.lower()}"
            destination_label = label_dir / f"{stem}.txt"
            shutil.copy2(record["image"], destination_image)

            lines = []
            for box in record["boxes"]:
                values = np.asarray(box, dtype=np.float32).reshape(-1)
                if len(values) != 4 or not np.isfinite(values).all():
                    continue
                x1, y1, x2, y2 = np.clip(values, 0, 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                lines.append(
                    f"{OUTPUT_FACE_CLASS_ID} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
                )
            if not lines:
                destination_image.unlink(missing_ok=True)
                continue
            destination_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

        target_name = "full_cutter" if view == "top" else "diamond_table"
        yaml = (
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"  0: {target_name}\n"
        )
        (output_root / view / "data.yaml").write_text(yaml, encoding="utf-8")
        print(f"{view}: exported {len(view_records)} annotated image(s); class 0 = {target_name}.")

    print(f"YOLO object-detection datasets exported to: {output_root.resolve()}")
    print("Source images and the annotation manifest were not modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Draw diamond-face bounding boxes and export separate Top/Isometric YOLO detection datasets."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--export-only", action="store_true",
                        help="Export saved bounding boxes without opening the editor.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1).")
    manifest = load_manifest(args.manifest)

    if not args.export_only:
        images = collect_images(args.images)
        print(f"Found {len(images)} images.")
        print(f"Saved annotations: {args.manifest}")
        index = 0
        while 0 <= index < len(images):
            outcome = edit_one(
                images[index], args.images, manifest, args.manifest,
                index + 1, len(images)
            )
            if outcome == "next":
                index += 1
            elif outcome == "previous":
                index = max(0, index - 1)
            else:
                break
        cv2.destroyAllWindows()

    export_datasets(args.images, args.output, manifest, args.manifest,
                    args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
