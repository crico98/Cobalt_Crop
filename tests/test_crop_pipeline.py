import json
import tempfile
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
    load_manifest_records,
    normalize_manifest_records,
    resolve_angle_index,
    square_crop_geometry,
)


class AngleResolutionTests(unittest.TestCase):
    def test_explicit_angle_index_mapping(self):
        self.assertEqual(resolve_angle_index(0), ("0", "isometric", 0))
        self.assertEqual(resolve_angle_index(1), ("1", "top", 1))
        self.assertEqual(resolve_angle_index(2), ("2", "face", 2))
        self.assertEqual(resolve_angle_index("1"), ("1", "top", 1))

    def test_invalid_angle_index_values_fail(self):
        for value in (None, True, 1.5, "top", -1, 3, "03"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_angle_index(value)

    def test_manifest_groups_by_angle_index_and_preserves_order(self):
        records = [
            {"file_path": "photo_c.jpg", "angle_index": 2},
            {"file_path": "photo_a.jpg", "angle_index": 1},
            {"file_path": "photo_d.jpg", "angle_index": 0},
            {"file_path": "photo_b.jpg", "angle_index": 1},
            {"file_path": "photo_e.jpg", "angle_index": 0},
        ]
        items = normalize_manifest_records(records)
        self.assertEqual(
            [(item.view, Path(item.file_path).name) for item in items],
            [
                ("isometric", "photo_d.jpg"),
                ("isometric", "photo_e.jpg"),
                ("top", "photo_a.jpg"),
                ("top", "photo_b.jpg"),
                ("face", "photo_c.jpg"),
            ],
        )

    def test_json_manifest_adapter_reads_angle_index_records(self):
        records = [
            {"file_path": "batch_face.jpg", "angle_index": 2},
            {"file_path": "batch_iso.jpg", "angle_index": 0},
        ]
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "fixture.json"
            manifest_path.write_text(json.dumps(records), encoding="utf-8")
            items = load_manifest_records(manifest_path)
        self.assertEqual([item.view for item in items], ["isometric", "face"])

    def test_manifest_requires_file_path_and_angle_index(self):
        with self.assertRaisesRegex(ValueError, "file_path"):
            normalize_manifest_records([{"angle_index": 0}])
        with self.assertRaisesRegex(ValueError, "angle_index"):
            normalize_manifest_records([{"file_path": "photo.jpg"}])

    def test_manifest_rejects_duplicate_photo_names(self):
        records = [
            {"file_path": "folder_a/photo.jpg", "angle_index": 0},
            {"file_path": "folder_b/photo.jpg", "angle_index": 1},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_manifest_records(records)


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
        self.assertEqual(crop.getpixel((0, 60)), (24, 24, 24))
        self.assertEqual(crop.getpixel((60, 60)), (255, 0, 0))

    def test_invalid_crop_boxes_rejected(self):
        with self.assertRaises(ValueError):
            square_crop_geometry((100, 100), (5, 5, 5, 20), 0.8)
        with self.assertRaises(ValueError):
            square_crop_geometry((100, 100), (5, 5, 20, 20), 0)

    def test_highest_confidence_box_selected(self):
        class FakeBoxes:
            xyxy = SimpleNamespace(tolist=lambda: [[1, 2, 10, 20], [5, 6, 30, 40]])
            conf = [SimpleNamespace(item=lambda: 0.4), SimpleNamespace(item=lambda: 0.9)]

            def __len__(self):
                return 2

        result = SimpleNamespace(boxes=FakeBoxes())
        self.assertEqual(
            choose_detection(result),
            {"box": (5.0, 6.0, 30.0, 40.0), "confidence": 0.9},
        )


if __name__ == "__main__":
    unittest.main()
