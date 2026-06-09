import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.utils.slides.image_slides_upload import auto_resolve_slide_assets, build_slides_response


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

    def test_build_slides_response_includes_storyboard_brief_fields(self):
        slides = [
            {
                "start_time": "0:00:00.00",
                "end_time": "0:00:04.00",
                "summary": "Blackbeard treats dreams like survival",
                "image_search_query": "Blackbeard Jaya cherry pie",
                "visual_purpose": "Introduce Blackbeard's obsession through a specific Jaya scene.",
                "viewer_focus": "Inspect Blackbeard in Jaya because his dream logic starts there.",
                "manual_upload_brief": "Pick a Jaya/Mock Town Blackbeard image, not a generic portrait.",
                "avoid_visual_reuse": "Avoid another front-facing Blackbeard headshot.",
                "asset_metadata": {"asset_type": "character", "query": "Blackbeard Jaya cherry pie"},
                "visual_intent": {"primary_intent": "context_setup"},
                "emotion_state": {"emotion": "intrigue", "intensity": 0.6},
                "visual_diversity_score": 0.84,
                "diversity_notes": "changes from nearby slides",
                "character_relationships": [{"source": "Blackbeard", "target": "Luffy", "relationship": "contrasts"}],
                "retention_score": 0.78,
                "retention_actions": ["kept concrete Jaya asset"],
                "composition_layers": [{"name": "background_asset", "type": "image"}],
                "motion_plan": {"preset": "subject_push", "camera_goal": "face_lock_push"},
                "architecture_stage": "Final Slides",
                "architecture_path": ["Subtitles", "Final Slides"],
            }
        ]

        response = build_slides_response("storyboard-test.wav", "/tmp/slides.json", slides)
        item = response["slides"][0]

        self.assertEqual(item["visual_purpose"], "Introduce Blackbeard's obsession through a specific Jaya scene.")
        self.assertEqual(item["viewer_focus"], "Inspect Blackbeard in Jaya because his dream logic starts there.")
        self.assertEqual(item["manual_upload_brief"], "Pick a Jaya/Mock Town Blackbeard image, not a generic portrait.")
        self.assertEqual(item["avoid_visual_reuse"], "Avoid another front-facing Blackbeard headshot.")
        self.assertEqual(item["asset_metadata"]["asset_type"], "character")
        self.assertEqual(item["visual_intent"]["primary_intent"], "context_setup")
        self.assertEqual(item["emotion_state"]["emotion"], "intrigue")
        self.assertEqual(item["visual_diversity_score"], 0.84)
        self.assertEqual(item["character_relationships"][0]["relationship"], "contrasts")
        self.assertEqual(item["retention_score"], 0.78)
        self.assertEqual(item["motion_plan"]["preset"], "subject_push")
        self.assertEqual(item["architecture_stage"], "Final Slides")


if __name__ == "__main__":
    unittest.main()
