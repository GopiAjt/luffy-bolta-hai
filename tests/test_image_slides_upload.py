import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.utils.slides.image_slides_upload import auto_resolve_slide_assets


class ImageSlidesUploadTests(unittest.TestCase):
    def test_auto_resolve_slide_assets_fills_missing_vivre_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "zoro.png"
            Image.new("RGB", (80, 120), (20, 120, 40)).save(source)
            slides_path = root / "slides.json"
            slides_path.write_text(
                json.dumps(
                    [
                        {
                            "start_time": "0:00:00.00",
                            "end_time": "0:00:02.00",
                            "summary": "Zoro protects Luffy",
                            "image_search_query": "Roronoa Zoro",
                            "asset_confidence": 0.9,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "app.utils.slides.image_slides_upload.resolve_vivre_asset_for_query",
                return_value=str(source),
            ):
                result = auto_resolve_slide_assets("auto-resolve-test.wav", str(slides_path))

            self.assertEqual(result["resolved"], 1)
            self.assertTrue(result["complete"])
            self.assertTrue(Path(result["slides"][0]["image_path"]).exists())
            self.assertEqual(result["slides"][0]["image_source"], "auto_vivre_card")


if __name__ == "__main__":
    unittest.main()
