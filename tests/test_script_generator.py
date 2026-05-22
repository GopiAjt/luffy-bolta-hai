import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.utils import script_generator
from app.utils.script_generator import generate_script, validate_generated_script


VALID_SCRIPT = (
    "Chapter 907 puts Shanks inside Mary Geoise. "
    "The Five Elders do not treat him like a normal pirate, and that silence is the clue. "
    "Most fans watched the meeting for the warning, but the room itself matters more. "
    "Shanks stands calm while the world government listens. "
    "My payoff is simple: Shanks knows a government secret that changes how pirates reach the final island. "
    "Tell me, is he protecting Luffy or steering him? "
    "Follow for more hidden One Piece truths."
)


def gemini_text(script=VALID_SCRIPT, title="Shanks Warning Changes Mary Geoise"):
    return (
        f"TITLE: {title}\n\n"
        f"SCRIPT: {script}\n\n"
        "DESCRIPTION: - Shanks walked into power like he belonged.\n"
        "- Mary Geoise makes this warning feel different.\n"
        "- Most fans missed how calmly they listened.\n"
        "- Prove this read wrong in the comments.\n"
        "- Follow for more hidden One Piece truths.\n\n"
        "HASHTAGS: #onepiece #anime #theory #shanks #luffy #marygeoise #mindblown"
    )


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_content(self, prompt, generation_config=None):
        self.prompts.append(prompt)
        return FakeResponse(self.responses.pop(0))


class ScriptGeneratorTests(unittest.TestCase):
    def test_valid_structured_output_parses_and_returns_metadata(self):
        fake_model = FakeModel([gemini_text()])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(result["title"], "Shanks Warning Changes Mary Geoise")
        self.assertIn("Chapter 907", result["script"])
        self.assertEqual(result["resolved_topic"], "Shanks at Mary Geoise")
        self.assertEqual(result["quality_warnings"], [])
        self.assertFalse(result["retry_attempted"])

    def test_generic_topic_replacement_returns_resolved_topic(self):
        fake_model = FakeModel([gemini_text()])
        with patch.object(script_generator, "model", fake_model), patch.object(
            script_generator.random,
            "choice",
            return_value="The chapter 1 Shanks scene that defines Luffy's idea of freedom",
        ):
            result = generate_script(topic_override="Generate a One Piece narration script")

        self.assertEqual(
            result["resolved_topic"],
            "The chapter 1 Shanks scene that defines Luffy's idea of freedom",
        )

    def test_banned_hook_opener_fails_validation(self):
        result = {
            "title": "Shanks Warning Changes Mary Geoise",
            "script": VALID_SCRIPT.replace("Chapter 907 puts", "Remember Chapter 907 puts"),
            "description": "desc",
            "hashtags": "#onepiece",
        }
        validation = validate_generated_script(result)
        self.assertIn("Opening hook uses a banned opener", validation.errors)

    def test_missing_canon_anchor_fails_validation(self):
        result = {
            "title": "Shanks Warning Changes Mary Geoise",
            "script": VALID_SCRIPT.replace("Chapter 907 puts Shanks inside Mary Geoise", "Shanks walks into a quiet room"),
            "description": "desc",
            "hashtags": "#onepiece",
        }
        validation = validate_generated_script(result)
        self.assertIn("Opening lines do not include a concrete canon anchor", validation.errors)

    def test_short_form_word_count_outside_range_is_detected(self):
        result = {
            "title": "Shanks Warning Changes Mary Geoise",
            "script": "Chapter 907 puts Shanks inside Mary Geoise.",
            "description": "desc",
            "hashtags": "#onepiece",
        }
        validation = validate_generated_script(result)
        self.assertTrue(any("word count" in error for error in validation.errors))

    def test_chapter_lock_requires_requested_chapter_in_opening(self):
        result = {
            "title": "Shanks Warning Changes Mary Geoise",
            "script": VALID_SCRIPT,
            "description": "desc",
            "hashtags": "#onepiece",
        }
        validation = validate_generated_script(result, chapter_number=1182)
        self.assertIn("Chapter-locked script must open with Chapter 1182", validation.errors)

    def test_retry_succeeds_when_second_response_is_valid(self):
        invalid_script = VALID_SCRIPT.replace("Chapter 907 puts", "Remember Chapter 907 puts")
        fake_model = FakeModel([gemini_text(script=invalid_script), gemini_text()])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(result["title"], "Shanks Warning Changes Mary Geoise")
        self.assertEqual(len(fake_model.prompts), 2)
        self.assertIn("failed deterministic validation", fake_model.prompts[1])
        self.assertIn("No markdown", fake_model.prompts[1])
        self.assertTrue(result["retry_attempted"])

    def test_parse_response_strips_script_markdown_and_stage_notes(self):
        parsed = script_generator._parse_response(
            "TITLE: Test\n\n"
            "SCRIPT: Chapter 907 makes **Shanks** feel _dangerous_. [pause] Mary Geoise listens.\n\n"
            "DESCRIPTION: desc\n\n"
            "HASHTAGS: #onepiece"
        )

        self.assertEqual(
            parsed["script"],
            "Chapter 907 makes Shanks feel dangerous. Mary Geoise listens.",
        )

    def test_parse_response_accepts_markdown_wrapped_section_labels(self):
        parsed = script_generator._parse_response(
            "**TITLE:** Shanks Warning Changes Mary Geoise\n\n"
            "**SCRIPT:** Chapter 907 makes Shanks feel dangerous. Mary Geoise listens.\n\n"
            "**DESCRIPTION:** desc\n\n"
            "**HASHTAGS:** #onepiece #anime"
        )

        self.assertEqual(parsed["title"], "Shanks Warning Changes Mary Geoise")
        self.assertEqual(parsed["description"], "desc")
        self.assertEqual(parsed["hashtags"], "#onepiece #anime")

    def test_question_word_declarative_opener_is_allowed(self):
        script = (
            "Was not the real clue in Chapter 907. "
            "Mary Geoise listens while Shanks stands calmly before power itself. "
            "The Five Elders do not treat him like a normal pirate, and that silence is the clue. "
            "Most fans watched the meeting for the warning, but the room itself matters more. "
            "My payoff is simple: Shanks knows a government secret that changes how pirates reach the final island. "
            "Tell me, is he protecting Luffy or steering him? "
            "Follow for more hidden One Piece truths."
        )
        validation = validate_generated_script(
            {
                "title": "Shanks Warning Changes Mary Geoise",
                "script": script,
                "description": "desc",
                "hashtags": "#onepiece",
            }
        )

        self.assertNotIn("Opening hook must be declarative, not a question", validation.errors)

    def test_title_subject_mismatch_is_warning_not_blocking_error(self):
        script = VALID_SCRIPT.replace("Shanks", "the red-haired pirate")
        validation = validate_generated_script(
            {
                "title": "Nika Awakening Mystery",
                "script": script,
                "description": "desc",
                "hashtags": "#onepiece",
            }
        )

        self.assertTrue(any("Title and script may not share" in warning for warning in validation.warnings))
        self.assertFalse(any("Title and script" in error for error in validation.errors))

    def test_strict_repair_recovers_after_retry_still_missing_sections(self):
        incomplete_first = (
            "TITLE: Shanks Warning\n\n"
            "SCRIPT: Shanks knows the truth, and that changes everything."
        )
        incomplete_retry = (
            "TITLE: Shanks Warning\n\n"
            "SCRIPT: The secret is bigger than fans think."
        )
        fake_model = FakeModel([incomplete_first, incomplete_retry, gemini_text()])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(result["title"], "Shanks Warning Changes Mary Geoise")
        self.assertEqual(result["quality_warnings"], [])
        self.assertTrue(result["retry_attempted"])
        self.assertEqual(len(fake_model.prompts), 3)
        self.assertIn("FORMAT REPAIR REQUIRED", fake_model.prompts[2])

    def test_deterministic_fallback_prevents_500_when_all_attempts_are_incomplete(self):
        incomplete = (
            "TITLE: Shanks Warning\n\n"
            "SCRIPT: The secret is bigger than fans think."
        )
        fake_model = FakeModel([incomplete, incomplete, incomplete])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertTrue(result["title"])
        self.assertTrue(result["description"])
        self.assertTrue(result["hashtags"])
        self.assertIn("Mary Geoise", result["script"])
        self.assertTrue(result["retry_attempted"])
        self.assertTrue(any("FALLBACK_USED" in warning for warning in result["quality_warnings"]))
        self.assertEqual(len(fake_model.prompts), 3)

    def test_api_response_includes_additive_metadata(self):
        from app.api.main import app

        fake_result = {
            "title": "Shanks Warning Changes Mary Geoise",
            "script": VALID_SCRIPT,
            "description": "desc",
            "hashtags": "#onepiece",
            "resolved_topic": "Shanks at Mary Geoise",
            "quality_warnings": ["sample warning"],
        }
        with patch("app.api.main.generate_script", return_value=fake_result):
            response = app.test_client().post(
                "/api/v1/generate-script",
                json={"topic": "Shanks at Mary Geoise", "video_profile": "short_vertical"},
            )

        self.assertEqual(response.status_code, 200)
        output = response.get_json()["output"]
        self.assertEqual(output["resolved_topic"], "Shanks at Mary Geoise")
        self.assertEqual(output["quality_warnings"], ["sample warning"])
        self.assertEqual(output["script"], VALID_SCRIPT)


if __name__ == "__main__":
    unittest.main()
