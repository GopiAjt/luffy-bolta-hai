import unittest

from app.config import MAX_FILE_SIZE


class AudioUploadConfigTests(unittest.TestCase):
    def test_default_audio_upload_limit_allows_long_generated_wav(self):
        self.assertGreaterEqual(MAX_FILE_SIZE, 100 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
