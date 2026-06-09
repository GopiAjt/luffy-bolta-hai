import unittest

from app.utils.slides.visual_intent_classifier import (
    VISUAL_INTENTS,
    VisualIntentClassifier,
)


class VisualIntentClassifierTests(unittest.TestCase):
    def test_examples_map_to_requested_intents(self):
        classifier = VisualIntentClassifier()

        dream = classifier.classify(
            "setup",
            "Luffy's dream of freedom is bigger than becoming Pirate King.",
            ["Luffy"],
        )
        fear = classifier.classify(
            "evidence",
            "Zoro's fear of weakness starts with Kuina's death.",
            ["Zoro", "Kuina"],
        )
        warning = classifier.classify(
            "warning",
            "Blackbeard is not just a pirate; he is becoming a monster.",
            ["Blackbeard"],
        )

        self.assertEqual(dream["intent"], "DREAM")
        self.assertEqual(fear["intent"], "FEAR")
        self.assertEqual(warning["intent"], "WARNING")
        self.assertGreaterEqual(dream["confidence"], 0.4)
        self.assertGreaterEqual(fear["confidence"], 0.4)
        self.assertGreaterEqual(warning["confidence"], 0.4)

    def test_returns_full_confidence_scores_and_entities(self):
        result = VisualIntentClassifier().classify(
            "reveal",
            "The truth is hidden inside Mary Geoise.",
            ["Mary Geoise", "Five Elders"],
        )

        self.assertEqual(set(result["confidence_scores"].keys()), set(VISUAL_INTENTS))
        self.assertIn(result["intent"], VISUAL_INTENTS)
        self.assertEqual(result["entities"], ["Mary Geoise", "Five Elders"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(result["matched_signals"])

    def test_beat_prior_can_classify_cta_without_extra_entities(self):
        result = VisualIntentClassifier().classify(
            "cta",
            "Tell me what you think and follow for more hidden One Piece truths.",
            [],
        )

        self.assertEqual(result["intent"], "CTA")
        self.assertGreater(result["confidence_scores"]["CTA"], 0.5)


if __name__ == "__main__":
    unittest.main()
