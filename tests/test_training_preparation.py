import unittest

from tools.prepare_top_isometric_training import expand_box, rotate_box


class DerivedTrainingBoxTests(unittest.TestCase):
    def setUp(self):
        self.box = (0.25, 0.30, 0.65, 0.70)

    def test_unrotated_expansion_adds_five_percent_total(self):
        expanded = expand_box(self.box, rotated=False)
        self.assertAlmostEqual(expanded[2] - expanded[0], 0.40 * 1.05)
        self.assertAlmostEqual(expanded[3] - expanded[1], 0.40 * 1.05)
        self.assertAlmostEqual((expanded[0] + expanded[2]) / 2, 0.45)
        self.assertAlmostEqual((expanded[1] + expanded[3]) / 2, 0.50)

    def test_rotated_expansion_keeps_width_and_adds_twenty_percent_height(self):
        expanded = expand_box(self.box, rotated=True)
        self.assertAlmostEqual(expanded[2] - expanded[0], 0.40)
        self.assertAlmostEqual(expanded[3] - expanded[1], 0.40 * 1.20)
        self.assertAlmostEqual((expanded[0] + expanded[2]) / 2, 0.45)
        self.assertAlmostEqual((expanded[1] + expanded[3]) / 2, 0.50)

    def test_clockwise_quarter_turn_transforms_box(self):
        self.assertEqual(
            rotate_box((0.2, 0.1, 0.6, 0.5), 90),
            (0.5, 0.2, 0.9, 0.6),
        )

    def test_counterclockwise_quarter_turn_transforms_box(self):
        self.assertEqual(
            rotate_box((0.2, 0.1, 0.6, 0.5), 270),
            (0.1, 0.4, 0.5, 0.8),
        )

    def test_edge_limited_expansion_preserves_center_and_bounds(self):
        box = (0.0, 0.2, 0.2, 0.8)
        expanded = expand_box(box, rotated=False)
        self.assertGreaterEqual(expanded[0], 0.0)
        self.assertLessEqual(expanded[2], 1.0)
        self.assertAlmostEqual((expanded[0] + expanded[2]) / 2, 0.1)


if __name__ == "__main__":
    unittest.main()
