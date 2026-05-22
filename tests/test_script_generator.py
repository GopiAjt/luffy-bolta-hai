import os
import unittest
from unittest.mock import patch

os.environ.setdefault("NVIDIA_API_KEY", "test-key")

from app.utils import script_generator
from app.utils.script_generator import generate_script, validate_generated_script


VALID_SCRIPT = (
    "Chapter 907 puts Shanks inside Mary Geoise. "
    "The Five Elders do not treat him like a normal pirate, and that silence is the clue. "
    "Most fans watched the meeting for the warning, but the room itself matters more. "
    "Shanks stands calm while the world government listens. "
    "That payoff changes everything: Shanks knows a government secret that changes how pirates reach the final island. "
    "Tell me, is he protecting Luffy or steering him? "
    "Follow for more hidden One Piece truths."
)


def model_text(script=VALID_SCRIPT, title="Shanks Warning Changes Mary Geoise"):
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


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return [FakeChunk("TITLE: Test\n\n"), FakeChunk("SCRIPT: Body")]


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


class ScriptGeneratorTests(unittest.TestCase):
    def test_nvidia_model_streams_openai_compatible_response(self):
        fake_client = FakeOpenAIClient()
        nvidia_model = script_generator.NvidiaScriptModel(api_key="test-key")
        nvidia_model._client = fake_client

        response = nvidia_model.generate_content(
            "hello",
            generation_config={"temperature": 1, "top_p": 0.95, "max_tokens": 16384},
        )

        self.assertEqual(response.text, "TITLE: Test\n\nSCRIPT: Body")
        kwargs = fake_client.chat.completions.kwargs
        self.assertEqual(kwargs["model"], "deepseek-ai/deepseek-v4-pro")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(kwargs["temperature"], 1)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 16384)
        self.assertEqual(kwargs["extra_body"], {"chat_template_kwargs": {"thinking": False}})
        self.assertTrue(kwargs["stream"])

    def test_valid_structured_output_parses_and_returns_metadata(self):
        fake_model = FakeModel([model_text()])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(result["title"], "Shanks Warning Changes Mary Geoise")
        self.assertIn("Chapter 907", result["script"])
        self.assertEqual(result["resolved_topic"], "Shanks at Mary Geoise")
        self.assertEqual(result["quality_warnings"], [])
        self.assertFalse(result["retry_attempted"])

    def test_generic_topic_replacement_returns_resolved_topic(self):
        fake_model = FakeModel([model_text()])
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
        fake_model = FakeModel([model_text(script=invalid_script), model_text()])
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
        fake_model = FakeModel([incomplete_first, incomplete_retry, model_text()])
        with patch.object(script_generator, "model", fake_model):
            result = generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(result["title"], "Shanks Warning Changes Mary Geoise")
        self.assertEqual(result["quality_warnings"], [])
        self.assertTrue(result["retry_attempted"])
        self.assertEqual(len(fake_model.prompts), 3)
        self.assertIn("FORMAT REPAIR REQUIRED", fake_model.prompts[2])

    def test_all_attempts_incomplete_fails_loudly_without_generic_fallback(self):
        incomplete = (
            "TITLE: Shanks Warning\n\n"
            "SCRIPT: The secret is bigger than fans think."
        )
        fake_model = FakeModel([incomplete, incomplete, incomplete])
        with patch.object(script_generator, "model", fake_model):
            with self.assertRaisesRegex(ValueError, "failed after all repair attempts"):
                generate_script(topic_override="Shanks at Mary Geoise")

        self.assertEqual(len(fake_model.prompts), 3)

    def test_hollow_template_script_fails_named_entity_density(self):
        hollow_script = (
            "This farewell feels impossible to ignore. "
            "The scene looks simple at first, but Oda hides the pressure in the quiet details. "
            "One reaction, one pause, and one choice tell us the truth is bigger than the obvious answer. "
            "That is why this clue matters: it points to a secret motive driving the story from behind the curtain. "
            "This was not random foreshadowing, it was setup. "
            "Tell me if this changes your read, and follow for more One Piece truths."
        )
        validation = validate_generated_script(
            {
                "title": "Going Merry Farewell",
                "script": hollow_script,
                "description": "desc",
                "hashtags": "#onepiece",
            }
        )

        self.assertTrue(any("named One Piece entity" in error for error in validation.errors))

    def test_low_named_entity_count_warns_without_blocking(self):
        script = (
            "Chapter 907 puts Shanks inside Mary Geoise. "
            "The room stays quiet, but that silence gives the warning more weight. "
            "Most fans focus on the meeting, while the power dynamic does the real work. "
            "The government listens instead of laughing, and that reaction changes how the scene lands. "
            "That payoff changes everything: this warning proves one pirate can enter power's highest room and still control the conversation. "
            "Tell me if this changes your read, and follow for more One Piece truths."
        )
        validation = validate_generated_script(
            {
                "title": "Shanks Warning Changes Mary Geoise",
                "script": script,
                "description": "desc",
                "hashtags": "#onepiece",
            }
        )

        self.assertFalse(validation.errors)
        self.assertTrue(any("only 3 named entities" in warning for warning in validation.warnings))

    def test_four_named_entities_avoid_density_warning(self):
        validation = validate_generated_script(
            {
                "title": "Shanks Warning Changes Mary Geoise",
                "script": VALID_SCRIPT,
                "description": "desc",
                "hashtags": "#onepiece",
            }
        )

        self.assertFalse(any("named entities" in warning for warning in validation.warnings))

    def test_topic_chapter_mismatch_warns(self):
        validation = validate_generated_script(
            {
                "title": "Going Merry Farewell",
                "script": VALID_SCRIPT,
                "description": "desc",
                "hashtags": "#onepiece",
            },
            topic="Why Chapter 430 makes Going Merry's farewell hurt",
        )

        self.assertTrue(any("Topic references chapter" in warning for warning in validation.warnings))

    def test_prompt_contains_specificity_and_voice_rules(self):
        prompt = script_generator._build_prompt(
            topic="Why Chapter 430 makes Going Merry's farewell hurt",
            language="english",
            context_text="",
            chapter_number=None,
            ohara_context="",
            context_sources=None,
            video_profile="short_vertical",
        )

        self.assertIn("CONTENT SPECIFICITY RULES", prompt)
        self.assertIn("Use punctuation as pacing control", prompt)
        self.assertIn("Shape sentence length as emotional rhythm", prompt)
        self.assertIn("word-level phonetic shaping", prompt)
        self.assertIn("WEAK (fails)", prompt)
        self.assertNotIn("My payoff is simple", prompt)

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
