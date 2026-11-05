from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent

# Use V2 explicitly as the face-segmentation baseline. Do not silently fall
# back to V3: the V2/V3 comparison needs to be intentional and reproducible.
MODEL_PATH = ROOT / "models" / "face_combined_v2_finetune" / "weights" / "best.pt"

FACE_FRACTION = 0.875

MIN_CONF = 0.1
WARN_CONF = 0.80
MAX_AXIS_RATIO = 1.15


def fit_face_geometry(points):
    """
    Fit an ellipse to the YOLO segmentation polygon.

    Returns the geometry fields expected by app.py:

        cx
        cy
        r
        rx
        ry
        angle
        ratio
    """

    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    if len(points) < 5:
        return None

    (cx, cy), (axis_1, axis_2), angle = cv2.fitEllipse(points)

    rx = float(axis_1 / 2.0)
    ry = float(axis_2 / 2.0)

    r = (rx + ry) / 2.0

    ratio = max(rx, ry) / max(
        1e-6,
        min(rx, ry),
    )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "rx": rx,
        "ry": ry,
        "angle": float(angle),
        "ratio": float(ratio),
    }


def fit_face_geometry_fallback(points):
    """
    Fallback geometry when the segmentation has too few points
    for cv2.fitEllipse().
    """

    points = np.asarray(points, dtype=np.float32)

    if len(points) < 3:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(points)

    r = float(radius)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": r,
        "rx": r,
        "ry": r,
        "angle": 0.0,
        "ratio": 1.0,
    }


def _circle_from_three_points(a, b, c):
    """Return the circumcircle through three 2-D points, or None if degenerate."""
    ax, ay = map(float, a)
    bx, by = map(float, b)
    cx, cy = map(float, c)
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(determinant) < 1e-6:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    center_x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / determinant
    center_y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / determinant
    radius = float(np.hypot(ax - center_x, ay - center_y))
    if not np.isfinite([center_x, center_y, radius]).all() or radius <= 1.0:
        return None
    return float(center_x), float(center_y), radius


def _fit_circle_least_squares(points):
    """Algebraic least-squares circle fit used to refine RANSAC inliers."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center_x, center_y, constant = solution
    radius_squared = constant + center_x * center_x + center_y * center_y
    if radius_squared <= 1.0 or not np.isfinite(radius_squared):
        return None
    return float(center_x), float(center_y), float(np.sqrt(radius_squared))


def _angular_coverage_degrees(points, center_x, center_y):
    """Measure how much of a circle's circumference the inliers span."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    angles = np.sort(np.mod(np.degrees(np.arctan2(
        points[:, 1] - center_y, points[:, 0] - center_x
    )), 360.0))
    if len(angles) < 3:
        return 0.0
    gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
    return float(360.0 - np.max(gaps))


def reconstruct_round_face_from_fragment(contour, image_shape):
    """
    Infer a full circular reference from a broken cutter's visible outer arc.

    Returns None unless a broad arc supports the fit. The visible segmentation
    is never discarded; callers retain it separately from this full reference.
    """
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 12:
        return None

    # Resample uniformly by physical distance along the contour.
    MAX_POINTS = 1000

    # Remove consecutive duplicate points.
    if len(points) > 1:
        step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate(([True], step_lengths > 1e-6))
        points = points[keep]

    if len(points) < 12:
        return None

    # Include the closing segment if the contour is not already closed.
    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.vstack([points, points[0]])

    # Cumulative distance along the contour.
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]

    if total_length <= 1e-6:
        return None

    # Sample at equal physical-distance intervals.
    sample_count = min(MAX_POINTS, max(12, len(points) - 1))
    targets = np.linspace(0.0, total_length, sample_count, endpoint=False)

    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    points = np.column_stack((x, y))

    height, width = image_shape[:2]
    tolerance_floor = max(2.0, min(height, width) * 0.0015)
    rng = np.random.default_rng(1729)
    best = None

    for _ in range(min(900, max(250, len(points) * 3))):
        indices = rng.choice(len(points), size=3, replace=False)
        circle = _circle_from_three_points(points[indices[0]], points[indices[1]], points[indices[2]])
        if circle is None:
            continue
        center_x, center_y, radius = circle
        if radius < 8.0 or radius > max(height, width) * 0.75:
            continue
        if not (-0.25 * width <= center_x <= 1.25 * width):
            continue
        if not (-0.25 * height <= center_y <= 1.25 * height):
            continue

        distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
        inliers = points[np.abs(distances - radius) <= max(tolerance_floor, radius * 0.018)]
        if len(inliers) < max(12, int(len(points) * 0.12)):
            continue

        refined = _fit_circle_least_squares(inliers)
        if refined is None:
            continue
        center_x, center_y, radius = refined
        distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
        tolerance = max(tolerance_floor, radius * 0.018)
        inliers = points[np.abs(distances - radius) <= tolerance]
        coverage = _angular_coverage_degrees(inliers, center_x, center_y)
        if coverage < 130.0 or len(inliers) < max(12, int(len(points) * 0.12)):
            continue

        residual = float(np.median(np.abs(
            np.hypot(inliers[:, 0] - center_x, inliers[:, 1] - center_y) - radius
        )))
        support = len(inliers) / max(1, len(points))
        quality = support * (coverage / 360.0) / (
            1.0 + residual / max(1.0, radius * 0.01)
        )
        if best is None or quality > best["quality"]:
            best = {
                "cx": center_x, "cy": center_y, "r": radius,
                "coverage": coverage, "support": support,
                "residual": residual, "quality": quality,
            }

    if best is None:
        return None
    if best["support"] < 0.16 or best["residual"] > max(
        tolerance_floor * 1.5, best["r"] * 0.015
    ):
        return None

    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    full_circle = np.column_stack((
        best["cx"] + best["r"] * np.cos(angles),
        best["cy"] + best["r"] * np.sin(angles),
    )).astype(np.float32)
    return {
        "polygon": full_circle,
        "cx": float(best["cx"]),
        "cy": float(best["cy"]),
        "r": float(best["r"]),
        "coverage_degrees": float(best["coverage"]),
        "support": float(best["support"]),
        "residual": float(best["residual"]),
    }


class FaceFinder:
    def __init__(
        self,
        path=MODEL_PATH,
        imgsz=1024,
    ):
        self.path = Path(path)
        self.imgsz = imgsz
        self._model = None

    @property
    def available(self):
        return self.path.exists()

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.path))

        return self._model

    def find(self, image_rgb):
        if not self.available:
            return None

        image_array = np.asarray(image_rgb)

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            return None

        image_bgr = np.ascontiguousarray(image_array[:, :, ::-1])
        image_height, image_width = image_bgr.shape[:2]

        model = self._load_model()
        result = model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=MIN_CONF,
            verbose=False,
        )[0]

        if result.masks is None or len(result.masks.xy) == 0:
            return None

        image_cx = image_width / 2.0
        image_cy = image_height / 2.0
        detections = []

        # Prefer connected regions from the model's raster masks over the
        # single returned contour. One instance mask can contain more than one
        # disconnected region; treating its combined contour as one face can
        # crop across two neighboring cutters.
        mask_data = result.masks.data
        for index, polygon_xy in enumerate(result.masks.xy):
            confidence = float(result.boxes.conf[index])
            if confidence < MIN_CONF:
                continue

            candidate_regions = []
            if mask_data is not None and index < len(mask_data):
                instance_mask = mask_data[index].detach().cpu().numpy()
                if instance_mask.ndim == 3:
                    instance_mask = np.squeeze(instance_mask)
                if instance_mask.ndim == 2:
                    if instance_mask.shape != (image_height, image_width):
                        instance_mask = cv2.resize(
                            instance_mask.astype(np.float32),
                            (image_width, image_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    binary = (instance_mask > 0.5).astype(np.uint8)
                    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        binary, connectivity=8
                    )
                    component_areas = [
                        int(stats[label, cv2.CC_STAT_AREA])
                        for label in range(1, count)
                    ]
                    largest_component = max(component_areas, default=0)
                    min_component_area = max(
                        64,
                        int(round(largest_component * 0.08)),
                    )

                    for label in range(1, count):
                        area = int(stats[label, cv2.CC_STAT_AREA])
                        if area < min_component_area:
                            continue
                        component = (labels == label).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            component,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if not contours:
                            continue
                        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        if len(contour) < 3:
                            continue
                        cx = float(centroids[label][0])
                        cy = float(centroids[label][1])
                        candidate_regions.append((contour.astype(np.float32), cx, cy, area))

            # Fallback to the model contour if its raster mask did not yield a
            # usable component, preserving behavior for unusual model outputs.
            if not candidate_regions and len(polygon_xy) >= 3:
                polygon = np.asarray(polygon_xy, dtype=np.float32)
                candidate_regions.append((
                    polygon,
                    float(np.mean(polygon[:, 0])),
                    float(np.mean(polygon[:, 1])),
                    abs(float(cv2.contourArea(polygon))),
                ))

            for polygon, center_x, center_y, area in candidate_regions:
                dx = (center_x - image_cx) / max(1.0, image_width)
                dy = (center_y - image_cy) / max(1.0, image_height)
                detections.append({
                    "polygon": polygon,
                    "conf": confidence,
                    "cx": center_x,
                    "cy": center_y,
                    "center_distance": dx * dx + dy * dy,
                    "area": float(area),
                })

        if not detections:
            return None

        max_area = max(detection["area"] for detection in detections)
        for detection in detections:
            area_ratio = (
                detection["area"] / max_area if max_area > 0 else 0.0
            )
            # Center proximity is the main target-selection signal. Keep the
            # area penalty small so a merged mask spanning multiple cutters
            # does not beat a clean component just because it is larger.
            detection["selection_score"] = (
                detection["center_distance"]
                + (1.0 - area_ratio) * 0.03
            )

        detections.sort(
            key=lambda detection: (
                detection["selection_score"],
                -detection["conf"],
            )
        )

        print("\nFACE DETECTIONS:")
        for detection in detections:
            print(
                f"  center=({detection['cx']:.1f}, {detection['cy']:.1f}) "
                f"area={detection['area']:.0f} "
                f"conf={detection['conf']:.3f} "
                f"distance={detection['center_distance']:.5f} "
                f"score={detection['selection_score']:.5f}"
            )

        best = detections[0]
        print(
            f"  SELECTED center=({best['cx']:.1f}, {best['cy']:.1f}) "
            f"from {len(detections)} candidate region(s)"
        )

        visible_polygon = np.asarray(best["polygon"], dtype=np.float32).copy()
        reconstruction = reconstruct_round_face_from_fragment(
            visible_polygon, image_bgr.shape
        )
        polygon = (
            reconstruction["polygon"]
            if reconstruction is not None
            else visible_polygon
        )

        geometry = fit_face_geometry(polygon)
        if geometry is None:
            geometry = fit_face_geometry_fallback(polygon)
        if geometry is None:
            return None

        geometry["conf"] = float(best["conf"])
        geometry["area"] = float(best["area"])
        geometry["center_distance"] = float(best["center_distance"])
        geometry["polygon"] = polygon.copy()
        geometry["visible_polygon"] = visible_polygon
        geometry["geometry_reconstructed"] = reconstruction is not None
        geometry["geometry_method"] = (
            "robust_circle_from_visible_arc"
            if reconstruction is not None
            else "segmentation_contour_fallback"
        )
        geometry["geometry_needs_review"] = (
            reconstruction is None and geometry["ratio"] <= MAX_AXIS_RATIO
        )
        if reconstruction is not None:
            geometry["reconstruction_coverage_degrees"] = reconstruction["coverage_degrees"]
            geometry["reconstruction_support"] = reconstruction["support"]
            geometry["reconstruction_residual"] = reconstruction["residual"]
            print(
                "  RECONSTRUCTED full circular reference: "
                f"center=({geometry['cx']:.1f},{geometry['cy']:.1f}) "
                f"radius={geometry['r']:.1f} "
                f"arc={reconstruction['coverage_degrees']:.1f}deg "
                f"support={reconstruction['support']:.2f}"
            )
        elif geometry["geometry_needs_review"]:
            print(
                "  WARNING: full cutter geometry could not be confidently "
                "reconstructed; verify/correct the Cutter Face polygon"
            )

        geometry["detection_count"] = len(detections)
        geometry["low_conf"] = geometry["conf"] < WARN_CONF
        geometry["not_very_circular"] = geometry["ratio"] > MAX_AXIS_RATIO
        return geometry


def crop_around_circle(
    image,
    face,
    max_size=1200,
):
    """
    Crop around the detected cutter face.
    """

    cx = float(face["cx"])
    cy = float(face["cy"])
    r = float(face["r"])

    half = int(
        round(
            r / FACE_FRACTION
        )
    )

    box = (
        int(round(cx)) - half,
        int(round(cy)) - half,
        int(round(cx)) + half + 1,
        int(round(cy)) + half + 1,
    )

    crop = image.crop(box)

    crop.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS,
    )

    return crop, box


# app.py currently uses both names.
# Keep both available for compatibility.
crop_around_face = crop_around_circle