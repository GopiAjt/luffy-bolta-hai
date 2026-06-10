"""Tests for the AssetDatabase module.

These tests verify the core search and ranking functionality without
requiring the sentence-transformers model to be downloaded (the semantic
tests are skipped if the model is unavailable).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so app.* imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.assets.asset_database import (
    AssetDatabase,
    AssetRecord,
    ScoredAsset,
    _norm,
    _split_csv,
    get_asset_database,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def db() -> AssetDatabase:
    """Module-scoped database instance (loads taxonomy once)."""
    return AssetDatabase()


@pytest.fixture(scope="module")
def taxonomy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "app" / "data" / "one_piece_asset_taxonomy.json"


# ── Unit tests: helpers ──────────────────────────────────────────────


class TestHelpers:
    def test_norm_removes_special_chars(self):
        assert _norm("Monkey D. Luffy") == "monkeydluffy"

    def test_norm_empty(self):
        assert _norm("") == ""
        assert _norm(None) == ""

    def test_split_csv(self):
        result = _split_csv("freedom, joy, resolve, wonder")
        assert result == ["freedom", "joy", "resolve", "wonder"]

    def test_split_csv_empty(self):
        assert _split_csv("") == []
        assert _split_csv(None) == []


# ── Unit tests: AssetRecord ──────────────────────────────────────────


class TestAssetRecord:
    def test_embedding_text_includes_all_fields(self):
        rec = AssetRecord(
            id="test",
            name="Luffy",
            character="Monkey D. Luffy",
            arc="Dawn Island / Wano",
            emotion="freedom, joy",
            visual_type="hero_character",
            tags=["straw_hat", "captain"],
        )
        text = rec.embedding_text()
        assert "Luffy" in text
        assert "Monkey D. Luffy" in text
        assert "Dawn Island" in text
        assert "freedom" in text
        assert "straw_hat" in text

    def test_to_dict_excludes_internal_fields(self):
        rec = AssetRecord(id="test", name="Test")
        d = rec.to_dict()
        assert "id" in d
        assert "name" in d
        assert "_embedding_text" not in d


# ── Integration tests: index construction ────────────────────────────


class TestDatabaseInit:
    def test_loads_taxonomy_records(self, db: AssetDatabase):
        """The DB should index at least the taxonomy assets."""
        assert db.record_count > 0
        taxonomy_count = sum(1 for r in db.records if r.source == "taxonomy")
        assert taxonomy_count >= 80, f"Expected ≥80 taxonomy records, got {taxonomy_count}"

    def test_status_returns_dict(self, db: AssetDatabase):
        status = db.status()
        assert isinstance(status, dict)
        assert "total_records" in status
        assert "categories" in status
        assert status["total_records"] == db.record_count

    def test_luffy_in_records(self, db: AssetDatabase):
        ids = {r.id for r in db.records}
        assert "straw_hat_luffy" in ids

    def test_categories_populated(self, db: AssetDatabase):
        categories = {r.category for r in db.records}
        assert "straw_hats" in categories
        assert "yonko" in categories
        assert "admirals" in categories


# ── Integration tests: tag_search ────────────────────────────────────


class TestTagSearch:
    def test_single_tag(self, db: AssetDatabase):
        results = db.tag_search(["captain"], top_k=5)
        assert len(results) > 0
        assert any(r.asset.id == "straw_hat_luffy" for r in results)

    def test_multiple_tags(self, db: AssetDatabase):
        results = db.tag_search(["yonko", "marineford"], top_k=5)
        assert len(results) > 0
        # Multiple matching tags should score higher.
        for r in results:
            assert r.score > 0

    def test_nonexistent_tag_returns_empty(self, db: AssetDatabase):
        results = db.tag_search(["zzz_nonexistent_zzz"], top_k=5)
        assert len(results) == 0

    def test_empty_tags(self, db: AssetDatabase):
        results = db.tag_search([], top_k=5)
        assert len(results) == 0

    def test_score_is_ratio(self, db: AssetDatabase):
        results = db.tag_search(["captain", "straw_hat"], top_k=5)
        for r in results:
            assert 0.0 < r.score <= 1.0


# ── Integration tests: emotion_search ────────────────────────────────


class TestEmotionSearch:
    def test_grief_returns_sad_characters(self, db: AssetDatabase):
        results = db.emotion_search(["grief"], top_k=10)
        assert len(results) > 0
        names = [r.asset.name for r in results]
        # Ace's arc is about grief at Marineford.
        assert any("Marineford" in r.asset.name or "grief" in (r.asset.emotion or "").lower() for r in results)

    def test_freedom_returns_luffy(self, db: AssetDatabase):
        results = db.emotion_search(["freedom"], top_k=5)
        assert len(results) > 0
        assert any(r.asset.id == "straw_hat_luffy" for r in results)

    def test_empty_emotions(self, db: AssetDatabase):
        results = db.emotion_search([], top_k=5)
        assert len(results) == 0


# ── Integration tests: arc_search ────────────────────────────────────


class TestArcSearch:
    def test_marineford_arc(self, db: AssetDatabase):
        results = db.arc_search("Marineford", top_k=10)
        assert len(results) > 0
        # The Marineford arc entry itself should be top.
        ids = [r.asset.id for r in results]
        assert "arc_marineford" in ids

    def test_wano_arc(self, db: AssetDatabase):
        results = db.arc_search("Wano", top_k=10)
        assert len(results) > 0
        assert any("wano" in r.asset.arc.lower() for r in results)

    def test_nonexistent_arc(self, db: AssetDatabase):
        results = db.arc_search("Nonexistent Island ZZZ", top_k=5)
        assert len(results) == 0

    def test_empty_arc(self, db: AssetDatabase):
        results = db.arc_search("", top_k=5)
        assert len(results) == 0


# ── Integration tests: rank_for_beat ─────────────────────────────────


class TestRankForBeat:
    def test_basic_beat_ranking(self, db: AssetDatabase):
        """rank_for_beat should return scored results even without semantic."""
        beat = {
            "text": "Zoro sacrifices himself at Thriller Bark",
            "beat_type": "reveal",
            "entities": ["Zoro"],
            "emotion": "sacrifice",
            "tags": ["nothing_happened", "loyalty"],
        }
        # Force semantic weight to 0 so we don't need the model.
        results = db.rank_for_beat(
            beat,
            top_k=5,
            weights={"semantic": 0.0, "tag": 0.30, "emotion": 0.30, "arc": 0.20, "importance": 0.20},
        )
        assert len(results) > 0
        # Zoro should score high.
        top_ids = [r.asset.id for r in results[:3]]
        assert "straw_hat_zoro" in top_ids or any("zoro" in tid for tid in top_ids)

    def test_beat_with_emotion_dict(self, db: AssetDatabase):
        """Emotion can be a dict (as produced by the slide pipeline)."""
        beat = {
            "text": "The mystery of the Void Century",
            "beat_type": "mystery",
            "entities": [],
            "emotion": {"emotion": "intrigue", "intensity": 0.8},
            "tags": ["void_century"],
        }
        results = db.rank_for_beat(
            beat,
            top_k=5,
            weights={"semantic": 0.0, "tag": 0.30, "emotion": 0.30, "arc": 0.20, "importance": 0.20},
        )
        assert len(results) > 0

    def test_score_breakdown_present(self, db: AssetDatabase):
        beat = {
            "text": "Shanks stops the war",
            "entities": ["Shanks"],
            "tags": ["yonko"],
        }
        results = db.rank_for_beat(
            beat,
            top_k=3,
            weights={"semantic": 0.0, "tag": 0.30, "emotion": 0.20, "arc": 0.20, "importance": 0.30},
        )
        assert len(results) > 0
        breakdown = results[0].score_breakdown
        assert "tag_overlap" in breakdown
        assert "emotion_match" in breakdown
        assert "importance" in breakdown

    def test_empty_beat(self, db: AssetDatabase):
        results = db.rank_for_beat({}, top_k=5, weights={"semantic": 0.0, "tag": 0.0, "emotion": 0.0, "arc": 0.0, "importance": 1.0})
        # Should still return results ranked by importance alone.
        assert len(results) > 0

    def test_to_dict_serializable(self, db: AssetDatabase):
        beat = {"text": "Luffy", "tags": ["captain"]}
        results = db.rank_for_beat(
            beat,
            top_k=1,
            weights={"semantic": 0.0, "tag": 0.50, "emotion": 0.0, "arc": 0.0, "importance": 0.50},
        )
        assert len(results) > 0
        d = results[0].to_dict()
        # Should be JSON-serializable.
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ── Semantic search tests (skipped if model unavailable) ─────────────


def _model_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _model_available(), reason="sentence-transformers not installed")
class TestSemanticSearch:
    def test_semantic_luffy_freedom(self, db: AssetDatabase):
        results = db.semantic_search("Luffy freedom dream pirate king", top_k=5)
        assert len(results) > 0
        top_ids = [r.asset.id for r in results[:3]]
        assert "straw_hat_luffy" in top_ids

    def test_semantic_empty_query(self, db: AssetDatabase):
        results = db.semantic_search("", top_k=5)
        assert len(results) == 0

    def test_semantic_scores_positive(self, db: AssetDatabase):
        results = db.semantic_search("ancient weapons", top_k=5)
        for r in results:
            assert r.score > 0


# ── Singleton test ───────────────────────────────────────────────────


class TestSingleton:
    def test_get_asset_database_returns_same_instance(self):
        db1 = get_asset_database()
        db2 = get_asset_database()
        assert db1 is db2
