import unittest

from app.config import get_video_profile_config, normalize_video_profile
from app.utils.tts_generator import _split_text_for_tts


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


if __name__ == "__main__":
    unittest.main()
