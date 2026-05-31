import unittest

from app.utils.slides.image_slides import (
    _anchor_ai_image_prompt,
    _apply_visual_source_plan,
    _build_ai_image_prompt,
    _build_image_slides_full_prompt,
    _infer_query_from_subtitle_text,
    _needs_ai_visual,
    _repair_slide_timing_continuity,
    _split_long_slides_by_dialogues,
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
        )

        self.assertIn('visual_source to "asset_search"', prompt)
        self.assertIn('ai_image_prompt to empty string ""', prompt)
        self.assertIn("read ALL subtitles as one complete script", prompt)
        self.assertIn("Loki Elbaf chains", prompt)
        self.assertNotIn("USE ai_generate", prompt)

    def test_build_ai_prompt_anchors_to_spoken_line_and_context(self):
        prompt = _build_ai_image_prompt(
            "Dragon leaves the Marines due to corruption",
            "He was a Marine . He abandoned his military post after witnessing the horrific corruption",
            ["Dragon", "World Government"],
        )

        self.assertIn("Visualize this narration beat", prompt)
        self.assertIn("He was a Marine", prompt)
        self.assertIn("Main context: Dragon", prompt)
        self.assertTrue(prompt.startswith("Vertical 9:16 One Piece anime style illustration"))

    def test_anchor_ai_prompt_preserves_scene_but_adds_line_context(self):
        prompt = _anchor_ai_image_prompt(
            "Vertical 9:16 One Piece anime style illustration, no text, no watermark. "
            "A desk covered in complex tactical maps, stolen Navy documents with red wax seals, "
            "and a compass, dimly lit by a single candle.",
            "Revolutionary strategy based on stolen Navy secrets",
            "This means the Revolutionary Army 's entire global strategy is actually built on classified Navy secrets Dragon stole from the inside ,",
            ["Dragon", "Revolutionary Army"],
        )

        self.assertIn("Visualize this narration beat", prompt)
        self.assertIn("classified Navy secrets Dragon stole", prompt)
        self.assertIn("Main context: Dragon, Revolutionary Army", prompt)
        self.assertIn("A desk covered in complex tactical maps", prompt)

    def test_anchor_ai_prompt_does_not_duplicate_existing_spoken_anchor(self):
        prompt = _anchor_ai_image_prompt(
            "Vertical 9:16 One Piece anime style illustration, no text, no watermark. "
            "Visualize this narration beat, without showing words: 'Roger realized Poseidon would not be born for ten more years.' "
            "A pirate silhouette drops a useless ancient weapon blueprint beside a glowing mermaid crown.",
            "Poseidon is not born yet",
            "Roger realized Poseidon would not be born for ten more years",
            ["Poseidon", "Gol D. Roger"],
        )

        self.assertEqual(prompt.count("Visualize this narration beat"), 1)
        self.assertIn("Main context: Poseidon, Gol D. Roger", prompt)
        self.assertIn("drops a useless ancient weapon blueprint", prompt)

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

    def test_psychological_beats_stay_on_searchable_canon_scenes(self):
        self.assertFalse(
            _needs_ai_visual(
                "He is haunted by the randomness of mortality and powerlessness.",
                "Zoro's fear of weakness becomes a shield",
                "Zoro Kuina Shimotsuki stairs",
                0,
            )
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
