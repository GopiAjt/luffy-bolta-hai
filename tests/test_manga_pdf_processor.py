import unittest

from app.utils.manga_pdf_processor import (
    _chapter_matches_title,
    _extract_ohara_chapter_section,
    parse_chapter_number,
    score_text_quality,
)


class MangaPdfProcessorTests(unittest.TestCase):
    def test_parse_chapter_number_from_filename(self):
        self.assertEqual(parse_chapter_number("One Piece_ Chapter - 1182.pdf"), 1182)
        self.assertEqual(parse_chapter_number("one_piece_ch.1179.pdf"), 1179)

    def test_quality_rejects_symbol_heavy_ocr(self):
        text = "Ss = = & = = ua ry > q S i = © a = = S r = & a = = |."
        quality = score_text_quality(text, confidence=35)
        self.assertFalse(quality["usable"])
        self.assertEqual(quality["level"], "poor")

    def test_quality_accepts_clean_english_dialogue(self):
        text = (
            "Luffy and Usopp admire Elbaf while Sanji protects Bonney. "
            "The village is in danger, and everyone is running out of time."
        )
        quality = score_text_quality(text, confidence=88)
        self.assertTrue(quality["usable"])
        self.assertIn(quality["level"], {"fair", "good"})

    def test_ohara_title_matches_chapter_range(self):
        title = "Chapter Secrets - One Piece Chapters 1180-1182 in-depth analysis"
        self.assertTrue(_chapter_matches_title(title, 1182))
        self.assertFalse(_chapter_matches_title(title, 1183))

    def test_ohara_section_extracts_requested_chapter(self):
        article = """
        Chapter 1181
        Older analysis that should not be included.
        Chapter 1182
        Imu refers to Nidhoggr as a traitor.
        Ragnir is revealed as the hammer connected to Ratatoskr.
        Chapter 1183
        Future analysis that should not be included.
        """
        section = _extract_ohara_chapter_section(article, 1182)
        self.assertIn("Nidhoggr", section)
        self.assertIn("Ratatoskr", section)
        self.assertNotIn("Older analysis", section)
        self.assertNotIn("Future analysis", section)


if __name__ == "__main__":
    unittest.main()
