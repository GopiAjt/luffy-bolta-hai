import unittest

from app.utils.slides.story_analyzer import STORY_BEAT_TYPES, StoryAnalyzer


class StoryAnalyzerTests(unittest.TestCase):
    def test_analyzes_script_into_story_beats_with_confidence_scores(self):
        script = (
            "What if Shanks warning in Mary Geoise was not about Luffy at all? "
            "Chapter 907 starts inside the safest room in the world. "
            "The mystery is that the Five Elders listen to a pirate like he belongs there. "
            "But normal pirates should never get that kind of audience. "
            "The evidence is the scene itself, because Shanks stands calm while the World Government waits. "
            "Then the stakes get bigger, because this meeting connects pirates to the final island. "
            "The reveal is that Shanks may know a government secret. "
            "This means the warning changes everything about his role. "
            "If this is true, it is bad news for anyone trusting him blindly. "
            "Follow for more hidden One Piece truths."
        )

        beats = StoryAnalyzer().analyze(script)
        beat_types = [beat["beat_type"] for beat in beats]

        self.assertGreaterEqual(len(beats), 5)
        self.assertEqual(beats[0]["beat_type"], "hook")
        self.assertIn("mystery", beat_types)
        self.assertIn("contradiction", beat_types)
        self.assertIn("evidence", beat_types)
        self.assertIn("escalation", beat_types)
        self.assertIn("reveal", beat_types)
        self.assertIn("payoff", beat_types)
        self.assertIn("warning", beat_types)
        self.assertEqual(beats[-1]["beat_type"], "cta")

        for beat in beats:
            self.assertIn("confidence", beat)
            self.assertGreaterEqual(beat["confidence"], 0.0)
            self.assertLessEqual(beat["confidence"], 1.0)
            self.assertEqual(set(beat["confidence_scores"].keys()), set(STORY_BEAT_TYPES))

    def test_empty_script_returns_no_beats(self):
        self.assertEqual(StoryAnalyzer().analyze("   "), [])

    def test_setup_and_evidence_can_be_detected_without_position_only(self):
        beats = StoryAnalyzer().analyze(
            "Chapter 1 begins with Shanks giving Luffy the straw hat. "
            "The panel proves Luffy learns freedom from sacrifice."
        )

        self.assertEqual(beats[0]["beat_type"], "hook")
        self.assertIn("evidence", [beat["beat_type"] for beat in beats])
        self.assertGreater(beats[-1]["confidence_scores"]["evidence"], 0.3)


if __name__ == "__main__":
    unittest.main()
