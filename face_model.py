from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent

# Use V2 explicitly as the face-segmentation baseline. Do not silently fall
# back to V3: the V2/V3 comparison needs to be intentional and reproducible.
MODEL_PATH = ROOT / "models" / "face_combined_v2_finetune" / "weights" / "best.pt"
# Also support the earlier standalone layout used by some local copies.
LEGACY_MODEL_PATH = ROOT / "models" / "best.pt"
if not MODEL_PATH.exists() and LEGACY_MODEL_PATH.exists():
    MODEL_PATH = LEGACY_MODEL_PATH

FACE_FRACTION = 0.875

MIN_CONF = 0.1
WARN_CONF = 0.80
MAX_AXIS_RATIO = 1.15

# Geometry functions remain imported here for compatibility with existing
# modules that may import them from face_model.
from face_geometry import (
    fit_face_geometry,
    fit_face_geometry_fallback,
    reconstruct_round_face_from_fragment,
)


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