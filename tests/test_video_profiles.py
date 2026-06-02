import unittest

from app.config import get_video_profile_config, normalize_video_profile
from app.utils.audio.tts_generator import (
    _qwen_generation_kwargs,
    _split_text_for_tts,
)
from app.utils.text.subtitle_generator import SubtitleGenerator


class VideoProfileTests(unittest.TestCase):
    def test_profile_defaults_and_dimensions(self):
        self.assertEqual(normalize_video_profile(None), "short_vertical")
        self.assertEqual(normalize_video_profile("unknown"), "short_vertical")
        self.assertEqual(normalize_video_profile("long_youtube"), "long_youtube")
        self.assertEqual(
            get_video_profile_config("short_vertical")["subtitle_resolution"],
            "1080x1920",
        )
        self.assertEqual(
            get_video_profile_config("long_youtube")["video_resolution"],
            (1920, 1080),
        )

    def test_tts_split_keeps_short_text_single_chunk(self):
        self.assertEqual(_split_text_for_tts("One short line.", 100), ["One short line."])

    def test_tts_split_chunks_long_text_on_sentence_boundaries(self):
        chunks = _split_text_for_tts(
            "First sentence. Second sentence has more words. Third sentence closes it.",
            24,
        )
        self.assertGreater(len(chunks), 1)
        self.assertIn("First sentence.", chunks[0])
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_tts_split_uses_paragraphs_as_chunks(self):
        chunks = _split_text_for_tts(
            "First paragraph stays together.\n\nSecond paragraph becomes another chunk.",
            1000,
        )
        self.assertEqual(
            chunks,
            [
                "First paragraph stays together.",
                "Second paragraph becomes another chunk.",
            ],
        )

    def test_tts_generation_kwargs_have_expressive_defaults(self):
        kwargs = _qwen_generation_kwargs()
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["top_k"], 50)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["temperature"], 0.92)

    def test_long_form_subtitles_select_sparse_word_keywords(self):
        generator = object.__new__(SubtitleGenerator)
        generator.style = "clean_pro"
        phrase = {
            "start": 0.0,
            "end": 2.5,
            "text": "Zoro chooses Luffy over glory forever",
            "words": [
                {"word": "Zoro", "start": 0.0, "end": 0.25},
                {"word": "chooses", "start": 0.3, "end": 0.65},
                {"word": "Luffy", "start": 0.7, "end": 1.0},
                {"word": "over", "start": 1.05, "end": 1.2},
                {"word": "glory", "start": 1.25, "end": 1.55},
                {"word": "forever", "start": 1.6, "end": 2.0},
            ],
        }

        keywords = generator._long_form_keyword_words(phrase, 0)

        self.assertLess(len(keywords), len(phrase["words"]))
        self.assertEqual([word["word"] for word in keywords], ["Zoro", "Luffy"])


if __name__ == "__main__":
    unittest.main()
