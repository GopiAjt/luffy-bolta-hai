import unittest

from app.utils.slides.image_slides import (
    _apply_visual_source_plan,
    _apply_production_edit_defaults,
    _apply_visual_architecture_pass,
    _build_image_slides_full_prompt,
    _infer_query_from_subtitle_text,
    _repair_slide_timing_continuity,
    _split_long_slides_by_dialogues,
    parse_ass_dialogues,
)


class ImageSlidesPromptTests(unittest.TestCase):
    def test_full_prompt_disables_ai_image_prompts(self):
        prompt = _build_image_slides_full_prompt(
            [
                {
                    "start": "0:00:00.03",
                    "end": "0:00:03.00",
                    "text": "Loki is chained beneath Elbaf because history remembers him.",
                }
            ],
            ["Loki", "Elbaf"],
            video_profile="short_vertical",
        )

        self.assertIn('visual_source to "asset_search"', prompt)
        self.assertIn('ai_image_prompt to empty string ""', prompt)
        self.assertIn("read ALL subtitles as one complete script", prompt)
        self.assertIn("PRODUCTION EDIT STRUCTURE", prompt)
        self.assertIn("beat_type, visual_role, layout_mode", prompt)
        self.assertIn("viewer_focus", prompt)
        self.assertIn("manual_upload_brief", prompt)
        self.assertIn("avoid_visual_reuse", prompt)
        self.assertIn("visual_purpose", prompt)
        self.assertIn("narration -> visual_purpose -> visual_role -> visual", prompt)
        self.assertIn("character -> object -> crew/group/evidence -> location -> symbol -> comparison -> timeline", prompt)
        self.assertIn("SHORTS PROFILE", prompt)
        self.assertIn("HOOK RETENTION PASS", prompt)
        self.assertIn("VISUAL ARCHITECTURE", prompt)
        self.assertIn("Emotional Curve Builder -> Visual Intent Classifier -> Asset Selector", prompt)
        self.assertIn("asset_metadata, visual_intent, emotion_state", prompt)
        self.assertIn("composition_layers", prompt)
        self.assertIn("motion_plan", prompt)
        self.assertIn("split it into 2-4 smaller visual cues", prompt)
        self.assertIn("Do not hold only a logo during the hook", prompt)
        self.assertIn("curiosity gap", prompt)
        self.assertIn("Loki Elbaf chains", prompt)
        self.assertNotIn("USE ai_generate", prompt)

    def test_full_prompt_includes_long_form_profile_rules(self):
        prompt = _build_image_slides_full_prompt(
            [{"start": "0:00:00.00", "end": "0:00:05.00", "text": "Elbaf changes the entire timeline."}],
            ["Elbaf"],
            video_profile="long_youtube",
        )

        self.assertIn("LONG-FORM PROFILE", prompt)
        self.assertIn("section/chapter cards", prompt)
        self.assertIn("horizontal_feature", prompt)
        self.assertIn("what the viewer should inspect", prompt)
        self.assertIn("what should be inspected on screen", prompt)

    def test_parse_ass_prefers_hidden_storyboard_lines_for_long_form(self):
        import tempfile
        from pathlib import Path

        ass = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:04.00,Storyboard,,0,0,0,,Blackbeard treats dreams like survival.
Dialogue: 0,0:00:00.10,0:00:00.80,Default,,0,0,0,,Blackbeard
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.ass"
            path.write_text(ass, encoding="utf-8")
            lines = parse_ass_dialogues(str(path))

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "Blackbeard treats dreams like survival.")

    def test_production_defaults_differ_by_profile(self):
        short = _apply_production_edit_defaults(
            {"summary": "The hidden truth about Elbaf", "subtitle_text": "The hidden truth about Elbaf"},
            0,
            3,
            "short_vertical",
            ["Elbaf"],
        )
        long = _apply_production_edit_defaults(
            {"summary": "The hidden truth about Elbaf", "subtitle_text": "The hidden truth about Elbaf"},
            0,
            3,
            "long_youtube",
            ["Elbaf"],
        )

        self.assertEqual(short["layout_mode"], "title_card")
        self.assertEqual(long["layout_mode"], "section_card")
        self.assertIn("motion_preset", short)
        self.assertIn("asset_confidence", short)
        self.assertIn("visual_purpose", long)
        self.assertIn("viewer_focus", long)
        self.assertIn("manual_upload_brief", long)
        self.assertIn("avoid_visual_reuse", long)

    def test_editor_visual_roles_are_inferred_from_story_purpose(self):
        fruit = _apply_production_edit_defaults(
            {"summary": "Blackbeard reaches for the Yami Yami no Mi fruit", "subtitle_text": "The fruit becomes his obsession."},
            3,
            8,
            "short_vertical",
            ["Blackbeard"],
        )
        place = _apply_production_edit_defaults(
            {"summary": "Everything begins in Jaya", "subtitle_text": "Jaya is where his dream becomes visible."},
            4,
            8,
            "long_youtube",
            ["Blackbeard"],
        )
        contrast = _apply_production_edit_defaults(
            {"summary": "Unlike Luffy, Blackbeard consumes his own identity", "subtitle_text": "Unlike Luffy, Blackbeard consumes his own identity."},
            5,
            8,
            "long_youtube",
            ["Blackbeard", "Luffy"],
        )

        self.assertEqual(fruit["visual_role"], "object")
        self.assertEqual(place["visual_role"], "location")
        self.assertEqual(contrast["visual_role"], "comparison")

    def test_zoro_specific_queries_are_inferred_from_spoken_context(self):
        self.assertEqual(
            _infer_query_from_subtitle_text(
                "When Zoro learned Kuina died falling down the stairs in Shimotsuki Village",
                ["Zoro"],
            ),
            "Zoro Kuina Shimotsuki stairs",
        )
        self.assertEqual(
            _infer_query_from_subtitle_text(
                "Mihawk stops Zoro at Baratie with a tiny cross knife",
                ["Zoro"],
            ),
            "Zoro Mihawk Baratie cross knife",
        )

    def test_long_slides_split_into_subtitle_grounded_beats(self):
        slides = [
            {
                "start_time": "0:00:14.48",
                "end_time": "0:00:32.20",
                "summary": "The tragic death of Kuina on the stairs.",
                "visual_source": "asset_search",
                "image_search_query": "Kuina Shimotsuki Village",
                "ai_image_prompt": "",
                "context_entities": ["Zoro", "Kuina"],
            }
        ]
        dialogues = [
            {
                "start": "0:00:14.48",
                "end": "0:00:17.22",
                "text": "All of it traces back to a single moment in chapter five.",
            },
            {
                "start": "0:00:17.56",
                "end": "0:00:24.73",
                "text": "Inside a quiet dojo in Shimotsuki Village, Zoro learned Kuina died falling down stairs.",
            },
            {
                "start": "0:00:24.75",
                "end": "0:00:32.20",
                "text": "That moment shatters Zoro's understanding of the world and leaves him terrified of mortality.",
            },
        ]

        result = _split_long_slides_by_dialogues(slides, dialogues, ["Zoro", "Kuina"])

        self.assertGreater(len(result), 1)
        self.assertEqual(result[0]["start_time"], "0:00:14.48")
        self.assertTrue(any("Kuina" in slide["summary"] for slide in result))
        self.assertTrue(all(slide["visual_source"] == "asset_search" for slide in result))
        self.assertTrue(any("Zoro Kuina Shimotsuki stairs" in slide["image_search_query"] for slide in result))

    def test_gemini_ai_choice_is_overridden_when_canon_scene_is_searchable(self):
        slide = {
            "summary": "Mihawk exposes Zoro's weakness at Baratie",
            "visual_source": "ai_generate",
            "image_search_query": "",
            "ai_image_prompt": "Vertical 9:16 One Piece anime style illustration, no text, no watermark. Zoro faces Mihawk.",
        }

        result = _apply_visual_source_plan(
            slide,
            "Mihawk defeats Zoro at Baratie with a tiny cross knife.",
            ["Zoro", "Mihawk", "Baratie"],
            0,
        )

        self.assertEqual(result["visual_source"], "asset_search")
        self.assertEqual(result["image_search_query"], "Zoro Mihawk Baratie cross knife")
        self.assertEqual(result["ai_image_prompt"], "")

    def test_gemini_ai_choice_is_stripped_when_no_ai_prompts_are_allowed(self):
        slide = {
            "summary": "Loki is the reason Oda made us wait",
            "visual_source": "ai_generate",
            "image_search_query": "",
            "ai_image_prompt": "Vertical 9:16 One Piece anime style illustration, no text.",
        }

        result = _apply_visual_source_plan(
            slide,
            "He is the reason Oda made us wait.",
            ["Loki", "Elbaf"],
            0,
        )

        self.assertEqual(result["visual_source"], "asset_search")
        self.assertEqual(result["image_search_query"], "Loki Elbaf")
        self.assertEqual(result["ai_image_prompt"], "")

    def test_slide_timing_repair_removes_gaps_for_sequential_renderer(self):
        slides = [
            {"start_time": "0:00:00.03", "end_time": "0:00:03.24", "summary": "one"},
            {"start_time": "0:00:03.30", "end_time": "0:00:05.66", "summary": "two"},
            {"start_time": "0:00:06.60", "end_time": "0:00:08.88", "summary": "three"},
        ]
        dialogues = [
            {"start": "0:00:00.03", "end": "0:00:03.24", "text": "one"},
            {"start": "0:00:03.30", "end": "0:00:05.66", "text": "two"},
            {"start": "0:00:06.60", "end": "0:00:08.88", "text": "three"},
        ]

        repaired = _repair_slide_timing_continuity(slides, 9.5, dialogues)

        self.assertEqual(repaired[0]["start_time"], "0:00:00.00")
        self.assertEqual(repaired[0]["end_time"], repaired[1]["start_time"])
        self.assertEqual(repaired[1]["end_time"], repaired[2]["start_time"])
        self.assertEqual(repaired[-1]["end_time"], "0:00:09.50")




if __name__ == "__main__":
    unittest.main()
