import unittest

from app.utils.expressions.expression_mapper import (
    enrich_expression_mapping,
    normalize_expression_mapping,
    parse_ass_file,
)
from app.utils.expressions.expression_overlay_cv import _hold_motion, _position_xy
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

    def test_sanitizes_expression_interval_that_overlaps_next_subtitle(self):
        generator = VideoGenerator()
        sanitized = generator._sanitize_expression_intervals(
            [
                (89.80, 91.43),
                (91.50, 193.86),
                (92.71, 93.35),
            ]
        )

        self.assertEqual(sanitized[1], (91.50, 92.70))

    def test_normalizes_gemini_expression_timings_to_ass_lines(self):
        sub_lines = [
            {"start": "0:02:29.80", "end": "0:02:31.43", "text": "That memory was buried."},
            {"start": "0:02:32.71", "end": "0:02:33.35", "text": "Next beat."},
        ]
        expressions = [
            {
                "start": "0:01:31.50",
                "end": "0:03:13.86",
                "text": "That memory was buried.",
                "expression": "neutral",
                "character": "narrator",
            }
        ]

        normalized = normalize_expression_mapping(expressions, sub_lines)

        self.assertEqual(normalized[0]["start"], "0:02:29.80")
        self.assertEqual(normalized[0]["end"], "0:02:31.43")

    def test_expression_parser_prefers_storyboard_comments(self):
        import tempfile
        from pathlib import Path

        ass = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:05.00,Storyboard,,0,0,0,,Blackbeard's fear of weakness becomes terrifying.
Dialogue: 0,0:00:00.10,0:00:00.80,Default,,0,0,0,,Blackbeard
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.ass"
            path.write_text(ass, encoding="utf-8")
            lines = parse_ass_file(str(path))

        self.assertEqual(len(lines), 1)
        self.assertIn("fear of weakness", lines[0]["text"])
        self.assertIn(lines[0]["expression"], {"worried", "intense"})

    def test_enrich_expression_mapping_adds_long_form_cues_when_sparse(self):
        sub_lines = [
            {
                "start": f"0:00:{i * 20:05.2f}" if i < 3 else f"0:01:{(i - 3) * 20:05.2f}",
                "end": f"0:00:{i * 20 + 5:05.2f}" if i < 3 else f"0:01:{(i - 3) * 20 + 5:05.2f}",
                "text": "Blackbeard proves ambition can consume a person's humanity.",
            }
            for i in range(7)
        ]

        enriched = enrich_expression_mapping([], sub_lines)

        self.assertGreaterEqual(len(enriched), 5)
        self.assertTrue(all(item.get("expression") != "neutral" for item in enriched))

    def test_enrich_expression_mapping_extends_sparse_shorts_to_late_beats(self):
        sub_lines = [
            {
                "start": f"0:00:{i * 5:05.2f}",
                "end": f"0:00:{i * 5 + 2:05.2f}",
                "text": "The Holy Knights reveal a terrifying truth about the final villains.",
            }
            for i in range(9)
        ]
        expressions = [
            {
                "start": "0:00:12.99",
                "end": "0:00:14.38",
                "text": "of the hierarchy. The Holy",
                "expression": "serious",
                "character": "narrator",
            },
            {
                "start": "0:00:14.42",
                "end": "0:00:15.02",
                "text": "Knights do.",
                "expression": "surprised",
                "character": "narrator",
            },
        ]

        enriched = enrich_expression_mapping(expressions, sub_lines)
        latest = max(VideoGenerator()._parse_time(item["start"]) for item in enriched)

        self.assertGreaterEqual(len(enriched), 5)
        self.assertGreaterEqual(latest, 30.0)

    def test_prepare_expression_sequence_skips_neutral_and_repeated_shorts_labels(self):
        generator = VideoGenerator()
        expressions = [
            {"start": "0:00:01.00", "end": "0:00:01.70", "text": "First claim.", "expression": "serious"},
            {"start": "0:00:03.00", "end": "0:00:03.70", "text": "Second claim.", "expression": "serious"},
            {"start": "0:00:05.00", "end": "0:00:05.70", "text": "Twist?", "expression": "surprised"},
            {"start": "0:00:07.00", "end": "0:00:07.70", "text": "Filler.", "expression": "neutral"},
        ]

        prepared = generator._prepare_expression_sequence(expressions, "", horizontal_video=False)

        self.assertEqual([item["expression"] for item in prepared], ["serious", "surprised"])

    def test_hold_motion_only_applies_after_entry_for_continuous_holds(self):
        self.assertEqual(_hold_motion(0.1, 0.2, True), (1.0, 0, 0, 1.0))
        self.assertEqual(_hold_motion(0.6, 0.2, False), (1.0, 0, 0, 1.0))

        scale, x_offset, y_offset, alpha = _hold_motion(0.6, 0.2, True)
        self.assertNotEqual(scale, 1.0)
        self.assertTrue(-4 <= x_offset <= 4)
        self.assertTrue(-5 <= y_offset <= 5)
        self.assertTrue(0.96 <= alpha <= 1.0)

    def test_expression_positions_support_horizontal_side_placements(self):
        left_x, left_y = _position_xy("bottom_left", 1920, 1080, 300, 420, "pop_in", 0.2)
        right_x, right_y = _position_xy("bottom_right", 1920, 1080, 300, 420, "pop_in", 0.2)

        self.assertLess(left_x, 1920 // 3)
        self.assertGreater(right_x, 1920 // 2)
        self.assertEqual(left_y, right_y)


if __name__ == "__main__":
    unittest.main()
