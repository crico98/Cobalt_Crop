import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from crop_pipeline import (
    centered_detection_image,
    choose_detection,
    crop_square,
    detection_region,
    map_box_to_full_image,
    normalize_manifest_records,
    resolve_angle,
    square_crop_geometry,
)


class AngleResolutionTests(unittest.TestCase):
    def test_filename_angle_mapping(self):
        self.assertEqual(resolve_angle("batch_123_angle_0_photo.jpg"), ("0", "isometric", 0))
        self.assertEqual(resolve_angle("batch_123_angle_1_photo.jpg"), ("1", "top", 1))
        self.assertEqual(resolve_angle("batch_123_angle_2_photo.jpg"), ("2", "face", 2))

    def test_missing_unsupported_and_ambiguous_angles_fail(self):
        for name in ("photo.jpg", "photo_angle_3.jpg", "angle_0_angle_2.jpg"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_angle(name)

    def test_manifest_groups_views_and_preserves_order_within_group(self):
        records = [
            {"file_path": "x_angle_2_c.jpg"},
            {"file_path": "x_angle_1_a.jpg"},
            {"file_path": "x_angle_0_a.jpg"},
            {"file_path": "x_angle_1_b.jpg"},
            {"file_path": "x_angle_0_b.jpg"},
        ]
        items = normalize_manifest_records(records)
        self.assertEqual(
            [(item.view, Path(item.file_path).name) for item in items],
            [
                ("isometric", "x_angle_0_a.jpg"),
                ("isometric", "x_angle_0_b.jpg"),
                ("top", "x_angle_1_a.jpg"),
                ("top", "x_angle_1_b.jpg"),
                ("face", "x_angle_2_c.jpg"),
            ],
        )

    def test_missing_file_path_fails(self):
        with self.assertRaises(ValueError):
            normalize_manifest_records([{}])


class GeometryTests(unittest.TestCase):
    def test_all_six_detection_region_settings(self):
        landscape = Image.new("RGB", (1600, 900))
        portrait = Image.new("RGB", (900, 1600))
        self.assertEqual(detection_region("top", landscape), (50, 68))
        self.assertEqual(detection_region("top", portrait), (63, 62))
        self.assertEqual(detection_region("isometric", landscape), (55, 80))
        self.assertEqual(detection_region("isometric", portrait), (65, 55))
        self.assertEqual(detection_region("face", landscape), (55, 80))
        self.assertEqual(detection_region("face", portrait), (60, 45))

    def test_centered_region_and_coordinate_translation(self):
        image = Image.new("RGB", (1000, 800))
        region, offset = centered_detection_image(image, 50, 50)
        self.assertEqual(region.size, (500, 400))
        self.assertEqual(offset, (250, 200))
        self.assertEqual(
            map_box_to_full_image((10, 20, 110, 120), offset),
            (260.0, 220.0, 360.0, 320.0),
        )

    def test_square_crop_pads_edges_and_is_square(self):
        image = Image.new("RGB", (100, 80), (255, 0, 0))
        box = square_crop_geometry(image.size, (0, 10, 30, 40), 0.75)
        crop = crop_square(image, box, output_size=120)
        self.assertEqual(crop.size, (120, 120))

    def test_invalid_crop_boxes_rejected(self):
        with self.assertRaises(ValueError):
            square_crop_geometry((100, 100), (5, 5, 5, 20), 0.8)
        with self.assertRaises(ValueError):
            square_crop_geometry((100, 100), (5, 5, 20, 20), 0)

    def test_highest_confidence_box_selected(self):
        boxes = SimpleNamespace(
            xyxy=SimpleNamespace(tolist=lambda: [[1, 2, 10, 20], [5, 6, 30, 40]]),
            conf=[SimpleNamespace(item=lambda: 0.4), SimpleNamespace(item=lambda: 0.9)],
        )
        result = SimpleNamespace(boxes=boxes)
        self.assertEqual(
            choose_detection(result, (100, 100)),
            {"box": (5.0, 6.0, 30.0, 40.0), "confidence": 0.9},
        )


if __name__ == "__main__":
    unittest.main()
