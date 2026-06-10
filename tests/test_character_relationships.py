"""Tests for the CharacterRelationshipEngine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.assets.character_relationships import (
    CharacterRelationshipEngine,
    RankedRelationship,
    _norm,
    get_relationship_engine,
)


# ── Fixture ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine() -> CharacterRelationshipEngine:
    return CharacterRelationshipEngine()


# ── Helpers ──────────────────────────────────────────────────────────


class TestNorm:
    def test_basic(self):
        assert _norm("Monkey D. Luffy") == "monkeydluffy"

    def test_empty(self):
        assert _norm("") == ""
        assert _norm(None) == ""


# ── Name resolution ──────────────────────────────────────────────────


class TestResolveCharacter:
    def test_short_alias(self, engine):
        assert engine.resolve_character("Luffy") == "straw_hat_luffy"

    def test_full_name(self, engine):
        assert engine.resolve_character("Monkey D. Luffy") == "straw_hat_luffy"

    def test_blackbeard(self, engine):
        assert engine.resolve_character("Blackbeard") == "yonko_blackbeard"

    def test_teach(self, engine):
        assert engine.resolve_character("Teach") == "yonko_blackbeard"

    def test_shanks(self, engine):
        assert engine.resolve_character("Shanks") == "yonko_shanks"

    def test_unknown_returns_none(self, engine):
        assert engine.resolve_character("Random Person XYZ") is None

    def test_case_insensitive(self, engine):
        assert engine.resolve_character("ZORO") == "straw_hat_zoro"
        assert engine.resolve_character("shanks") == "yonko_shanks"


# ── Core relationship queries ────────────────────────────────────────


class TestBlackbeardRelationships:
    """The example from the user: Blackbeard → Ace, Whitebeard, Luffy, Shanks."""

    def test_returns_results(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        assert len(results) > 0

    def test_ace_in_top_results(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        targets = [r.target for r in results]
        target_ids = [r.target_id for r in results]
        # Ace is anchored at arc_marineford ("Portgas D. Ace" is the character).
        assert "arc_marineford" in target_ids or any("ace" in t.lower() for t in targets)

    def test_whitebeard_in_top_results(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        target_ids = [r.target_id for r in results]
        assert "yonko_whitebeard" in target_ids

    def test_luffy_in_top_results(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        target_ids = [r.target_id for r in results]
        assert "straw_hat_luffy" in target_ids

    def test_shanks_in_top_results(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        target_ids = [r.target_id for r in results]
        assert "yonko_shanks" in target_ids

    def test_correct_ordering(self, engine):
        """Ace/Whitebeard should rank above lower-signal connections."""
        results = engine.get_relationships("Blackbeard", top_k=10)
        target_ids = [r.target_id for r in results]
        # Whitebeard and Marineford (Ace) should be in top 4.
        top4_ids = set(target_ids[:4])
        assert "yonko_whitebeard" in top4_ids or "arc_marineford" in top4_ids

    def test_has_opposes_relationship(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=10)
        types = {r.relationship for r in results}
        assert "opposes" in types or "rivals" in types


class TestLuffyRelationships:
    def test_returns_results(self, engine):
        results = engine.get_relationships("Luffy", top_k=10)
        assert len(results) >= 5

    def test_shanks_mentor(self, engine):
        results = engine.get_relationships("Luffy", top_k=10)
        shanks_results = [r for r in results if r.target_id == "yonko_shanks"]
        assert len(shanks_results) > 0
        assert shanks_results[0].relationship == "mentor_link"

    def test_crew_members_present(self, engine):
        results = engine.get_relationships("Luffy", top_k=20)
        target_ids = {r.target_id for r in results}
        # At least some crew members should appear.
        crew = {"straw_hat_zoro", "straw_hat_sanji", "straw_hat_robin"}
        assert crew & target_ids, f"Expected crew members in {target_ids}"

    def test_family_present(self, engine):
        results = engine.get_relationships("Luffy", top_k=15)
        target_ids = {r.target_id for r in results}
        family = {"rev_dragon", "rev_sabo", "legend_garp"}
        assert family & target_ids, f"Expected family in {target_ids}"


class TestZoroRelationships:
    def test_mihawk_is_rival(self, engine):
        results = engine.get_relationships("Zoro", top_k=10)
        mihawk_results = [r for r in results if r.target_id == "warlord_mihawk"]
        assert len(mihawk_results) > 0
        assert mihawk_results[0].relationship == "rivals"

    def test_luffy_is_ally(self, engine):
        results = engine.get_relationships("Zoro", top_k=10)
        luffy_results = [r for r in results if r.target_id == "straw_hat_luffy"]
        assert len(luffy_results) > 0
        assert luffy_results[0].relationship == "allies_with"


# ── Narration boosting ───────────────────────────────────────────────


class TestNarrationBoosting:
    def test_narration_boosts_mentioned_characters(self, engine):
        # Without narration.
        base = engine.get_relationships("Blackbeard", top_k=20)
        base_scores = {r.target_id: r.score for r in base}

        # With narration mentioning Shanks prominently.
        narration = (
            "Blackbeard scarred Shanks years before the story began. "
            "Shanks tried to warn the Five Elders about the threat Blackbeard poses."
        )
        boosted = engine.get_relationships("Blackbeard", top_k=20, narration=narration)
        boosted_scores = {r.target_id: r.score for r in boosted}

        # Shanks should score higher with narration.
        assert boosted_scores.get("yonko_shanks", 0) > base_scores.get("yonko_shanks", 0)

    def test_narration_detects_relationship_type(self, engine):
        narration = "Luffy fought Kaido in a battle that shook Onigashima."
        results = engine.get_relationships("Luffy", top_k=10, narration=narration)
        kaido_results = [r for r in results if r.target_id == "yonko_kaido"]
        assert len(kaido_results) > 0
        assert "narration_boost" in kaido_results[0].score_breakdown
        assert kaido_results[0].score_breakdown["narration_boost"] > 0


# ── Mutual connections ───────────────────────────────────────────────


class TestMutualConnections:
    def test_luffy_blackbeard_mutuals(self, engine):
        results = engine.get_mutual_connections("Luffy", "Blackbeard", top_k=5)
        assert len(results) > 0
        # Shanks, Ace/Marineford should be mutual.
        target_ids = {r.target_id for r in results}
        assert "yonko_shanks" in target_ids or "arc_marineford" in target_ids

    def test_unknown_character_returns_empty(self, engine):
        results = engine.get_mutual_connections("Unknown Person", "Luffy", top_k=5)
        assert len(results) == 0


# ── Score structure ──────────────────────────────────────────────────


class TestScoreStructure:
    def test_scores_in_valid_range(self, engine):
        results = engine.get_relationships("Shanks", top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_score_breakdown_present(self, engine):
        results = engine.get_relationships("Luffy", top_k=3)
        assert len(results) > 0
        bd = results[0].score_breakdown
        assert "graph_weight" in bd
        assert "importance_boost" in bd

    def test_evidence_present(self, engine):
        results = engine.get_relationships("Blackbeard", top_k=3)
        assert len(results) > 0
        assert len(results[0].evidence) > 0

    def test_to_dict_serializable(self, engine):
        results = engine.get_relationships("Luffy", top_k=1)
        assert len(results) > 0
        d = results[0].to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ── Engine status & listing ──────────────────────────────────────────


class TestEngineIntrospection:
    def test_status(self, engine):
        s = engine.status()
        assert s["node_count"] > 80
        assert s["edge_count"] > 50
        assert "categories" in s

    def test_list_characters(self, engine):
        chars = engine.list_characters()
        assert len(chars) > 50
        names = [c["name"] for c in chars]
        assert "Monkey D. Luffy" in names


# ── Singleton ────────────────────────────────────────────────────────


class TestSingleton:
    def test_returns_same_instance(self):
        e1 = get_relationship_engine()
        e2 = get_relationship_engine()
        assert e1 is e2
