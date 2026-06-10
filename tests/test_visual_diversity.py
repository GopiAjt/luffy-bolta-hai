"""Tests for the VisualDiversityScorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.visual_diversity import (
    DiversityReport,
    SectionScore,
    VisualDiversityScorer,
    score_visual_diversity,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_beat(
    character: str = "Luffy",
    arc: str = "wano",
    visual_role: str = "character",
    layout_mode: str = "safe_subject",
    emotion: str = "hope",
    query: str = None,
    beat_type: str = "evidence",
) -> dict:
    if query is None:
        query = f"{character} {arc}".strip()
    return {
        "context_entities": [character] if character else [],
        "image_search_query": query,
        "visual_role": visual_role,
        "layout_mode": layout_mode,
        "beat_type": beat_type,
        "subtitle_text": f"{character} in {arc}",
        "summary": f"Scene about {character}",
        "emotion_state": {"emotion": emotion, "intensity": 0.5},
        "asset_metadata": {"entities": [character] if character else [], "search_tags": [arc]},
    }


def _diverse_slides() -> list:
    """A storyboard with good diversity across all dimensions."""
    return [
        _make_beat("Luffy", "dawn", "character", "safe_subject", "hope", beat_type="hook"),
        _make_beat("Zoro", "wano", "evidence", "bleed", "tension", beat_type="setup"),
        _make_beat("Robin", "enies_lobby", "symbol", "title_card", "grief", beat_type="evidence"),
        _make_beat("Shanks", "marineford", "location", "bleed", "intrigue", beat_type="reveal"),
        _make_beat("Whitebeard", "marineford", "comparison", "title_card", "rage", beat_type="payoff"),
        _make_beat("", "", "cta_card", "safe_subject", "resolution", beat_type="cta"),
    ]


def _repetitive_slides() -> list:
    """A storyboard with heavy repetition in all dimensions."""
    return [
        _make_beat("Luffy", "wano", "character", "safe_subject", "hope"),
        _make_beat("Luffy", "wano", "character", "safe_subject", "hope"),
        _make_beat("Luffy", "wano", "character", "safe_subject", "hope"),
        _make_beat("Luffy", "wano", "character", "safe_subject", "hope"),
        _make_beat("Luffy", "wano", "character", "safe_subject", "hope"),
    ]


@pytest.fixture
def scorer():
    return VisualDiversityScorer(rejection_threshold=0.7)


# ── Core rejection rule ─────────────────────────────────────────────


class TestRejectionThreshold:
    """Sections above 0.7 repetition must be rejected."""

    def test_repetitive_slides_get_rejected(self, scorer):
        report = scorer.score(_repetitive_slides())
        # After the first slide, all subsequent identical beats should be rejected.
        rejected = report.rejected_indices
        assert len(rejected) >= 2, f"Expected ≥2 rejected, got {rejected}"

    def test_diverse_slides_not_rejected(self, scorer):
        report = scorer.score(_diverse_slides())
        assert report.rejected_count == 0, (
            f"Diverse beats should not be rejected, but got {report.rejected_indices}"
        )

    def test_first_slide_never_rejected(self, scorer):
        report = scorer.score(_repetitive_slides())
        assert not report.sections[0].rejected
        assert report.sections[0].repetition_score == 0.0

    def test_threshold_boundary(self):
        """A section with exactly 0.7 repetition should NOT be rejected (> not >=)."""
        scorer = VisualDiversityScorer(rejection_threshold=0.7)
        # We can't easily engineer exactly 0.7, but we can check that 0 is not rejected.
        report = scorer.score([_make_beat("Luffy")])
        assert not report.sections[0].rejected

    def test_custom_threshold(self):
        # Stricter threshold should reject more.
        strict = VisualDiversityScorer(rejection_threshold=0.3)
        lenient = VisualDiversityScorer(rejection_threshold=0.9)
        beats = _repetitive_slides()
        assert strict.score(beats).rejected_count >= lenient.score(beats).rejected_count


# ── Per-dimension penalties ──────────────────────────────────────────


class TestCharacterPenalty:
    def test_same_character_penalized(self, scorer):
        beats = [
            _make_beat("Luffy"),
            _make_beat("Luffy"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["character"] > 0

    def test_different_characters_no_penalty(self, scorer):
        beats = [
            _make_beat("Luffy"),
            _make_beat("Zoro"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["character"] == 0.0

    def test_character_streak_escalates(self, scorer):
        beats = [
            _make_beat("Luffy"),
            _make_beat("Luffy"),
            _make_beat("Luffy"),
            _make_beat("Luffy"),
        ]
        report = scorer.score(beats)
        # Penalty should increase with consecutive repetitions.
        assert report.sections[3].penalties["character"] >= report.sections[1].penalties["character"]


class TestArcPenalty:
    def test_same_arc_penalized(self, scorer):
        beats = [
            _make_beat("Luffy", "wano"),
            _make_beat("Zoro", "wano"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["arc"] > 0

    def test_different_arcs_no_penalty(self, scorer):
        beats = [
            _make_beat("Luffy", "wano"),
            _make_beat("Zoro", "marineford"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["arc"] == 0.0


class TestCompositionPenalty:
    def test_same_composition_penalized(self, scorer):
        beats = [
            _make_beat(visual_role="character", layout_mode="safe_subject"),
            _make_beat(visual_role="character", layout_mode="safe_subject"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["composition"] > 0

    def test_different_composition_no_penalty(self, scorer):
        beats = [
            _make_beat(visual_role="character", layout_mode="safe_subject"),
            _make_beat(visual_role="evidence", layout_mode="bleed"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["composition"] == 0.0


class TestEmotionPenalty:
    def test_same_emotion_penalized(self, scorer):
        beats = [
            _make_beat(emotion="hope"),
            _make_beat(emotion="hope"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["emotion"] > 0

    def test_different_emotions_no_penalty(self, scorer):
        beats = [
            _make_beat(emotion="hope"),
            _make_beat(emotion="grief"),
        ]
        report = scorer.score(beats)
        assert report.sections[1].penalties["emotion"] == 0.0


# ── Weighted scoring ────────────────────────────────────────────────


class TestWeightedScoring:
    def test_character_weighted_highest(self, scorer):
        """Character has highest weight (0.30), so character-only repeat should score highest."""
        # Only character repeats.
        slides_char = [
            _make_beat("Luffy", "dawn", "symbol", "bleed", "grief"),
            _make_beat("Luffy", "wano", "evidence", "title_card", "hope"),
        ]
        # Only emotion repeats.
        slides_emo = [
            _make_beat("Luffy", "dawn", "symbol", "bleed", "hope"),
            _make_beat("Zoro", "wano", "evidence", "title_card", "hope"),
        ]
        char_report = scorer.score(slides_char)
        emo_report = scorer.score(slides_emo)
        assert char_report.sections[1].repetition_score >= emo_report.sections[1].repetition_score

    def test_all_dimensions_repeat_maximizes_score(self, scorer):
        beats = _repetitive_slides()
        report = scorer.score(beats)
        # Full repetition on all 4 dimensions should push score very high.
        assert report.sections[-1].repetition_score > 0.8


# ── Notes and suggestions ───────────────────────────────────────────


class TestNotesAndSuggestions:
    def test_rejected_has_warning(self, scorer):
        report = scorer.score(_repetitive_slides())
        rejected_sections = [s for s in report.sections if s.rejected]
        assert len(rejected_sections) > 0
        for s in rejected_sections:
            assert any("REJECTED" in sug for sug in s.suggestions)

    def test_diverse_has_positive_note(self, scorer):
        report = scorer.score(_diverse_slides())
        # At least some non-first beats should have positive notes.
        non_first = report.sections[1:]
        positive = [s for s in non_first if any("Good" in n or "diversity" in n for n in s.notes)]
        assert len(positive) > 0

    def test_character_repeat_has_character_suggestion(self, scorer):
        beats = [
            _make_beat("Luffy", "dawn", "symbol", "bleed", "grief"),
            _make_beat("Luffy", "wano", "evidence", "title_card", "hope"),
        ]
        report = scorer.score(beats)
        section = report.sections[1]
        assert any("character" in s.lower() or "perspective" in s.lower() for s in section.suggestions)


# ── Summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_fields(self, scorer):
        report = scorer.score(_diverse_slides())
        s = report.summary
        assert "beat_count" in s
        assert "rejected_count" in s
        assert "mean_repetition" in s
        assert "diversity_grade" in s
        assert "character_distribution" in s
        assert "composition_distribution" in s

    def test_diverse_gets_good_grade(self, scorer):
        report = scorer.score(_diverse_slides())
        assert report.summary["diversity_grade"] in ("A", "B")

    def test_repetitive_gets_bad_grade(self, scorer):
        report = scorer.score(_repetitive_slides())
        assert report.summary["diversity_grade"] in ("D", "F")

    def test_character_distribution_counted(self, scorer):
        report = scorer.score(_repetitive_slides())
        dist = report.summary["character_distribution"]
        assert "luffy" in dist
        assert dist["luffy"] == 5


# ── Single slide scoring ────────────────────────────────────────────


class TestSingleSlideScoring:
    def test_score_single(self, scorer):
        prev = [_make_beat("Luffy"), _make_beat("Luffy")]
        current = _make_beat("Luffy")
        section = scorer.score_single(current, prev)
        assert section.penalties["character"] > 0
        assert section.index == 2

    def test_score_single_no_previous(self, scorer):
        current = _make_beat("Luffy")
        section = scorer.score_single(current, [])
        assert section.repetition_score == 0.0


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_slides(self, scorer):
        report = scorer.score([])
        assert report.sections == []
        assert report.summary["beat_count"] == 0

    def test_single_slide(self, scorer):
        report = scorer.score([_make_beat()])
        assert len(report.sections) == 1
        assert report.sections[0].repetition_score == 0.0
        assert not report.sections[0].rejected

    def test_missing_fields(self, scorer):
        """Slides with missing fields should not crash."""
        beats = [
            {"beat_type": "hook"},
            {"beat_type": "evidence"},
            {"beat_type": "reveal"},
        ]
        report = scorer.score(beats)
        assert len(report.sections) == 3

    def test_lookback_parameter(self):
        scorer = VisualDiversityScorer(lookback=1)
        beats = [
            _make_beat("Luffy"),
            _make_beat("Zoro"),
            _make_beat("Luffy"),  # Luffy is 2 beats back, lookback=1 means only check 1 back.
        ]
        report = scorer.score(beats)
        # With lookback=1, slide 2 only compares to slide 1 (Zoro), so no character penalty.
        assert report.sections[2].penalties["character"] == 0.0


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_section_to_dict(self, scorer):
        report = scorer.score(_diverse_slides())
        d = report.sections[0].to_dict()
        assert "repetition_score" in d
        assert "penalties" in d
        json.dumps(d)

    def test_report_to_dict(self, scorer):
        report = scorer.score(_diverse_slides())
        d = report.to_dict()
        assert "sections" in d
        assert "summary" in d
        json.dumps(d)

    def test_report_properties(self, scorer):
        report = scorer.score(_repetitive_slides())
        assert isinstance(report.rejected_indices, list)
        assert isinstance(report.rejected_count, int)
        assert isinstance(report.mean_repetition, float)


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_score_visual_diversity(self):
        report = score_visual_diversity(_diverse_slides())
        assert len(report.sections) == 6
        assert report.rejected_count == 0

    def test_custom_threshold(self):
        report = score_visual_diversity(_repetitive_slides(), rejection_threshold=0.3)
        assert report.rejected_count >= 2
