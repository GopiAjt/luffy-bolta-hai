import os
import unittest

from app.utils.slides.generate_slideshow import slideshow_encoder_args, x264_speed_settings
from app.utils.video.video_generator import _video_encoder_args, _x264_speed_settings


class VideoEncodingTests(unittest.TestCase):
    def test_slideshow_pro_uses_fast_visually_clean_encode(self):
        self.assertEqual(x264_speed_settings("pro"), ("fast", "22"))

    def test_slideshow_standard_uses_veryfast_encode(self):
        self.assertEqual(x264_speed_settings("standard"), ("veryfast", "23"))

    def test_final_video_pro_uses_fast_visually_clean_encode(self):
        self.assertEqual(_x264_speed_settings("pro"), ("fast", "22"))

    def test_slideshow_x264_uses_auto_threads(self):
        self.assertIn("-threads", slideshow_encoder_args("pro"))
        self.assertIn("0", slideshow_encoder_args("pro"))

    def test_final_video_x264_uses_auto_threads(self):
        self.assertIn("-threads", _video_encoder_args("pro"))
        self.assertIn("0", _video_encoder_args("pro"))

    def test_quicksync_encoder_is_configurable(self):
        original = os.environ.get("VIDEO_ENCODER")
        try:
            os.environ["VIDEO_ENCODER"] = "h264_qsv"
            self.assertIn("h264_qsv", _video_encoder_args("pro"))
        finally:
            if original is None:
                os.environ.pop("VIDEO_ENCODER", None)
            else:
                os.environ["VIDEO_ENCODER"] = original


if __name__ == "__main__":
    unittest.main()
