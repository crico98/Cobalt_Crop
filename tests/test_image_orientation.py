import unittest

from PIL import Image

from image_orientation import normalize_uploaded_image, read_exif_orientation


class ExifNormalizationTests(unittest.TestCase):
    def make_image(self, orientation):
        image = Image.new("RGB", (3, 2))
        image.putdata([
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ])
        exif = Image.Exif()
        exif[274] = orientation
        image.info["exif"] = exif.tobytes()
        # Re-open encoded bytes so Pillow reads the EXIF block as file metadata.
        import io
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=100, subsampling=0, exif=exif)
        stream.seek(0)
        return Image.open(stream)

    def test_all_standard_exif_orientations_are_applied(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                source = self.make_image(orientation)
                self.assertEqual(read_exif_orientation(source), orientation)
                normalized = normalize_uploaded_image(source)
                expected_size = (2, 3) if orientation in {5, 6, 7, 8} else (3, 2)
                self.assertEqual(normalized.size, expected_size)
                self.assertNotIn(274, normalized.getexif())

    def test_orientation_six_rotates_pixels_clockwise(self):
        source = self.make_image(6)
        normalized = normalize_uploaded_image(source)
        self.assertEqual(normalized.size, (2, 3))
        self.assertEqual(
            list(normalized.getdata()),
            [
                (255, 255, 0), (255, 0, 0),
                (255, 0, 255), (0, 255, 0),
                (0, 255, 255), (0, 0, 255),
            ],
        )


if __name__ == "__main__":
    unittest.main()
