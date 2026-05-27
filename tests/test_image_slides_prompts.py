import unittest

from app.utils.slides.image_slides import _anchor_ai_image_prompt, _build_ai_image_prompt


class ImageSlidesPromptTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
