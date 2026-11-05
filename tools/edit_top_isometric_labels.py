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
DEFAULT_LABEL_DIR = ROOT / "dataset" / "train" / "labels"
DEFAULT_OUTPUT_DIR = ROOT / "training"
MANIFEST_PATH = ROOT / "dataset" / "top_isometric_annotations.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_FACE_CLASS_ID = 3
OUTPUT_FACE_CLASS_ID = 0
WINDOW_PREFIX = "Top / Isometric Face Label Editor"


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_polygons(label_path: Path) -> list[np.ndarray]:
    polygons = []
    if not label_path.is_file():
        return polygons
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        except ValueError:
            print(f"Warning: malformed label line {line_number}: {label_path}")
            continue
        if class_id != SOURCE_FACE_CLASS_ID or len(coords) < 6 or len(coords) % 2:
            continue
        polygon = coords.reshape(-1, 2)
        if np.isfinite(polygon).all():
            polygons.append(polygon)
    return polygons


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "annotations": {}}
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


def normalized_to_pixels(polygons: list[list[list[float]]], width: int, height: int):
    return [
        np.asarray(poly, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        for poly in polygons
    ]


def pixels_to_normalized(polygons: list[np.ndarray], width: int, height: int):
    result = []
    for polygon in polygons:
        clipped = polygon.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, 1.0)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, 1.0)
        if len(clipped) >= 3 and np.isfinite(clipped).all():
            result.append(clipped.astype(float).tolist())
    return result


def draw_polygon(image, polygon, color, thickness=2, show_points=False):
    h, w = image.shape[:2]
    points = np.rint(polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
    if len(points) >= 3:
        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
    if show_points:
        for point in points:
            cv2.circle(image, tuple(point), 5, color, -1, cv2.LINE_AA)


class Editor:
    def __init__(self, image, saved_polygons, source_polygons, view):
        self.original = image
        self.saved_polygons = [p.copy() for p in saved_polygons]
        self.initial_polygons = [p.copy() for p in saved_polygons]
        self.source_polygons = [p.copy() for p in source_polygons]
        self.current = []
        self.view = view if view in {"top", "isometric"} else ""
        self.show_source = True
        self.h, self.w = image.shape[:2]
        scale = min(1400 / self.w, 900 / self.h, 1.0)
        self.display_w = max(1, int(round(self.w * scale)))
        self.display_h = max(1, int(round(self.h * scale)))
        self.drag_target = None

    def to_normalized(self, x, y):
        return [float(np.clip(x / self.display_w, 0, 1)),
                float(np.clip(y / self.display_h, 0, 1))]

    def to_display(self, point):
        return (int(round(point[0] * self.display_w)),
                int(round(point[1] * self.display_h)))

    def nearest_vertex(self, x, y, radius=14):
        candidates = []
        for polygon_index, polygon in enumerate(self.saved_polygons):
            for point_index, point in enumerate(polygon):
                px, py = self.to_display(point)
                distance = (px - x) ** 2 + (py - y) ** 2
                if distance <= radius ** 2:
                    candidates.append((distance, polygon_index, point_index))
        if not candidates:
            return None
        _, polygon_index, point_index = min(candidates)
        return polygon_index, point_index

    def move_vertex(self, x, y):
        if self.drag_target is not None:
            polygon_index, point_index = self.drag_target
            self.saved_polygons[polygon_index][point_index] = self.to_normalized(x, y)

    def delete_nearest_vertex(self, x, y):
        target = self.nearest_vertex(x, y)
        if target is None:
            return
        polygon_index, point_index = target
        polygon = self.saved_polygons[polygon_index]
        if len(polygon) <= 3:
            del self.saved_polygons[polygon_index]
        else:
            self.saved_polygons[polygon_index] = np.delete(polygon, point_index, axis=0)

    def render(self, number, total, image_name):
        image = cv2.resize(
            self.original, (self.display_w, self.display_h),
            interpolation=cv2.INTER_AREA if self.display_w < self.w else cv2.INTER_LINEAR,
        )
        if self.show_source:
            for polygon in self.source_polygons:
                draw_polygon(image, polygon, (255, 180, 0), 2)
        for polygon in self.saved_polygons:
            draw_polygon(image, polygon, (0, 0, 255), 2, True)
        if self.current:
            pts = np.rint(np.asarray(self.current) *
                          np.array([self.display_w, self.display_h])).astype(np.int32)
            for point in pts:
                cv2.circle(image, tuple(point), 5, (255, 255, 0), -1, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.polylines(image, [pts], False, (255, 255, 0), 2, cv2.LINE_AA)

        view_label = self.view.upper() if self.view else "UNASSIGNED"
        lines = [
            f"{number}/{total}  {image_name}  |  VIEW: {view_label}",
            "Click=add point | Drag red vertex=move | Right-click red vertex=delete | Enter=finish",
            "T=Top | I=Isometric | U=undo point | C=clear | R=reset | O=source overlay",
            "S=save+next | N=next (no save) | P=previous (no save) | Q/Esc=quit",
        ]
        header_h = 88
        cv2.rectangle(image, (0, 0), (self.display_w, header_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(image, line[:180], (10, 21 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48 if i else 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return image


def edit_one(image_path, image_root, label_root, manifest, manifest_path, number, total):
    key = relative_key(image_path, image_root)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: could not read {image_path}; skipping.")
        return "next"
    label_path = label_root / f"{image_path.stem}.txt"
    source_polygons = load_polygons(label_path)
    record = manifest["annotations"].get(key, {})
    saved = record.get("polygons")
    if saved is None:
        saved_polygons = [p.copy() for p in source_polygons]
    else:
        saved_polygons = normalized_to_pixels(saved, image.shape[1], image.shape[0])
        saved_polygons = [p / np.array([image.shape[1], image.shape[0]], dtype=np.float32)
                          for p in saved_polygons]

    editor = Editor(image, saved_polygons, source_polygons, record.get("view", ""))
    window = f"{WINDOW_PREFIX} - {image_path.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, editor.display_w, editor.display_h)

    def mouse_callback(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            target = editor.nearest_vertex(x, y)
            if target is not None:
                editor.drag_target = target
            else:
                editor.current.append(editor.to_normalized(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and editor.drag_target is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                editor.move_vertex(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            editor.move_vertex(x, y)
            editor.drag_target = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.delete_nearest_vertex(x, y)

    cv2.setMouseCallback(window, mouse_callback)
    print(f"\n[{number}/{total}] {key} | source class-{SOURCE_FACE_CLASS_ID} polygons: {len(source_polygons)}")
    outcome = "quit"
    try:
        while True:
            cv2.imshow(window, editor.render(number, total, image_path.name))
            keycode = cv2.waitKeyEx(20)
            if keycode == -1:
                continue
            keycode &= 0xFF
            if keycode in (13, 10):
                if len(editor.current) >= 3:
                    editor.saved_polygons.append(np.asarray(editor.current, dtype=np.float32))
                    editor.current = []
                else:
                    print("A polygon needs at least 3 points.")
            elif keycode in (ord("t"), ord("T")):
                editor.view = "top"
            elif keycode in (ord("i"), ord("I")):
                editor.view = "isometric"
            elif keycode in (ord("u"), ord("U")):
                if editor.current:
                    editor.current.pop()
                elif editor.saved_polygons:
                    editor.saved_polygons[-1] = editor.initial_polygons[-1].copy() if (
                        len(editor.saved_polygons) < len(editor.initial_polygons)
                    ) else editor.saved_polygons[-1]
                    if len(editor.saved_polygons) > len(editor.initial_polygons):
                        editor.saved_polygons.pop()
            elif keycode in (ord("c"), ord("C")):
                editor.saved_polygons = []
                editor.current = []
            elif keycode in (ord("r"), ord("R")):
                editor.saved_polygons = [p.copy() for p in editor.initial_polygons]
                editor.current = []
                editor.view = record.get("view", "")
            elif keycode in (ord("o"), ord("O")):
                editor.show_source = not editor.show_source
            elif keycode in (ord("s"), ord("S")):
                if editor.current:
                    print("Finish the current polygon with Enter before saving.")
                    continue
                if editor.saved_polygons and editor.view not in {"top", "isometric"}:
                    print("Assign this image with T (Top) or I (Isometric) before saving.")
                    continue
                manifest["annotations"][key] = {
                    "view": editor.view,
                    "polygons": pixels_to_normalized(
                        editor.saved_polygons, editor.w, editor.h
                    ),
                }
                save_manifest(manifest_path, manifest)
                print(f"Saved annotation: {key} ({editor.view or 'unassigned'})")
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
    if not records:
        return set()
    # Preserve subfolder/acquisition groups when source images are organized into folders.
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


def export_datasets(image_root, label_root, output_root, manifest, val_fraction=0.2, seed=42):
    records = []
    missing = []
    for key, annotation in manifest["annotations"].items():
        view = annotation.get("view")
        polygons = annotation.get("polygons", [])
        if view not in {"top", "isometric"} or not polygons:
            continue
        image_path = image_root / Path(key)
        if not image_path.is_file():
            missing.append(key)
            continue
        records.append({"key": key, "view": view, "polygons": polygons, "image": image_path})

    if missing:
        print(f"Warning: {len(missing)} annotated source images are missing and will be skipped.")
    if not records:
        raise RuntimeError("No saved images with a view assignment and at least one polygon to export.")

    for view in ("top", "isometric"):
        view_records = [r for r in records if r["view"] == view]
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
            for polygon in record["polygons"]:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) < 3 or not np.isfinite(points).all():
                    continue
                values = [str(OUTPUT_FACE_CLASS_ID)]
                for x, y in points:
                    values.extend((f"{float(np.clip(x, 0, 1)):.6f}",
                                   f"{float(np.clip(y, 0, 1)):.6f}"))
                lines.append(" ".join(values))
            if not lines:
                destination_image.unlink(missing_ok=True)
                continue
            destination_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

        yaml = (
            f"path: {view}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: cutter_face\n"
        )
        (output_root / view / "data.yaml").write_text(yaml, encoding="utf-8")
        print(f"{view}: exported {len(view_records)} annotated image(s).")

    print(f"YOLO segmentation datasets exported to: {output_root.resolve()}")
    print("Source images and original class-3 labels were not modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Review/edit source class-3 face polygons and export independent Top/Isometric YOLO segmentation datasets."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--export-only", action="store_true",
                        help="Export using saved annotations without opening the editor.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1).")
    manifest = load_manifest(args.manifest)

    if not args.export_only:
        images = collect_images(args.images)
        print(f"Found {len(images)} images (including images without labels).")
        print(f"Source labels: {args.labels}")
        print(f"Saved annotations: {args.manifest}")
        index = 0
        while 0 <= index < len(images):
            outcome = edit_one(
                images[index], args.images, args.labels, manifest,
                args.manifest, index + 1, len(images)
            )
            if outcome == "next":
                index += 1
            elif outcome == "previous":
                index = max(0, index - 1)
            else:
                break
        cv2.destroyAllWindows()

    export_datasets(args.images, args.labels, args.output, manifest,
                    args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
