import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_crop_test_batch import prepare_batch


class PrepareCropTestBatchTests(unittest.TestCase):
    def test_copies_both_views_and_writes_authoritative_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            iso = root / "iso"
            top = root / "top"
            output = root / "output"
            iso.mkdir()
            top.mkdir()
            (iso / "iso_a.jpg").write_bytes(b"iso")
            (iso / "iso_b.JPG").write_bytes(b"iso2")
            (top / "top_a.jpg").write_bytes(b"top")

            summary = prepare_batch(iso, top, output)
            records = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["counts"], {"isometric": 2, "top": 1})
            self.assertEqual(summary["total"], 3)
            self.assertEqual(
                [(record["file_path"], record["angle_index"]) for record in records],
                [("iso_a.jpg", 0), ("iso_b.JPG", 0), ("top_a.jpg", 1)],
            )
            self.assertEqual((iso / "iso_a.jpg").read_bytes(), b"iso")
            self.assertEqual((top / "top_a.jpg").read_bytes(), b"top")

    def test_existing_output_is_not_overwritten_without_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            iso = root / "iso"
            top = root / "top"
            output = root / "output"
            iso.mkdir()
            top.mkdir()
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_batch(iso, top, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_rejects_duplicate_filenames_across_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            iso = root / "iso"
            top = root / "top"
            output = root / "output"
            iso.mkdir()
            top.mkdir()
            (iso / "same.jpg").write_bytes(b"iso")
            (top / "SAME.JPG").write_bytes(b"top")

            with self.assertRaisesRegex(ValueError, "Duplicate filename"):
                prepare_batch(iso, top, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
