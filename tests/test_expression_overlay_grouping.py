import unittest

from app.utils.expressions.expression_overlay_cv import _hold_motion
from app.utils.video.video_generator import VideoGenerator


class ExpressionOverlayGroupingTests(unittest.TestCase):
    def test_merges_adjacent_same_expression_subtitle_spans(self):
        generator = VideoGenerator()
        merged = generator._merge_expression_intervals(
            [
                (8.12, 9.97),
                (10.01, 11.55),
                (11.61, 12.85),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 8.12)
        self.assertEqual(merged[0]["end"], 12.85)
        self.assertEqual(merged[0]["source_count"], 3)

    def test_keeps_expression_spans_separate_when_gap_is_large(self):
        generator = VideoGenerator()
        merged = generator._merge_expression_intervals(
            [
                (0.0, 1.0),
                (1.3, 2.0),
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual([item["source_count"] for item in merged], [1, 1])

    def test_hold_motion_only_applies_after_entry_for_continuous_holds(self):
        self.assertEqual(_hold_motion(0.1, 0.2, True), (1.0, 0, 0, 1.0))
        self.assertEqual(_hold_motion(0.6, 0.2, False), (1.0, 0, 0, 1.0))

        scale, x_offset, y_offset, alpha = _hold_motion(0.6, 0.2, True)
        self.assertNotEqual(scale, 1.0)
        self.assertTrue(-4 <= x_offset <= 4)
        self.assertTrue(-5 <= y_offset <= 5)
        self.assertTrue(0.96 <= alpha <= 1.0)


if __name__ == "__main__":
    unittest.main()
