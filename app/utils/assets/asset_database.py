"""Unified asset database with semantic, tag, emotion, and arc search.

Indexes every known asset (taxonomy entries, Vivre Card PNGs) and
exposes four search modes plus a composite ranker for storyboard beats.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.utils.assets.one_piece_taxonomy import (
    TAXONOMY_PATH,
    load_one_piece_asset_taxonomy,
    one_piece_taxonomy_assets,
)

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class AssetRecord:
    """Unified record that represents a single searchable asset."""

    id: str
    name: str
    category: str = ""
    character: str = ""
    arc: str = ""
    emotion: str = ""
    visual_type: str = ""
    importance: int = 3
    tags: List[str] = field(default_factory=list)
    source: str = "taxonomy"  # "taxonomy" | "vivre_card"
    file_path: str = ""

    # Pre-computed rich text used for embedding.
    _embedding_text: str = field(default="", repr=False, compare=False)

    def embedding_text(self) -> str:
        if self._embedding_text:
            return self._embedding_text
        parts = [self.name]
        if self.character and self.character != self.name:
            parts.append(f"Character: {self.character}")
        if self.arc:
            parts.append(f"Arcs: {self.arc}")
        if self.emotion:
            parts.append(f"Emotions: {self.emotion}")
        if self.visual_type:
            parts.append(f"Type: {self.visual_type}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        text = " — ".join(parts)
        object.__setattr__(self, "_embedding_text", text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_embedding_text", None)
        return d


@dataclass
class ScoredAsset:
    """An asset with a composite relevance score and per-component breakdown."""

    asset: AssetRecord
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset.to_dict(),
            "score": round(self.score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


# ── Helpers ──────────────────────────────────────────────────────────

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORMALIZE_RE.sub("", (text or "").lower())


def _split_csv(text: str) -> List[str]:
    """Split a comma-separated string into normalized tokens."""
    return [t.strip().lower() for t in (text or "").split(",") if t.strip()]


def _taxonomy_checksum() -> str:
    """Fast MD5 of the taxonomy JSON for cache invalidation."""
    try:
        data = TAXONOMY_PATH.read_bytes()
        return hashlib.md5(data).hexdigest()
    except Exception:
        return ""


# ── AssetDatabase ────────────────────────────────────────────────────


class AssetDatabase:
    """Central database for all One Piece visual assets.

    Provides:
    - ``semantic_search(query)`` — free-text embedding similarity
    - ``tag_search(tags)`` — set-intersection on asset tags
    - ``emotion_search(emotions)`` — fuzzy match on the emotion field
    - ``arc_search(arc_name)`` — substring match on the arc field
    - ``rank_for_beat(beat)`` — composite ranker for storyboard beats

    The database lazy-loads the sentence-transformer model on first
    semantic query, keeping ``__init__`` fast.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        embedding_model: Optional[str] = None,
    ):
        from app.config import BASE_DIR

        self._cache_dir = Path(
            cache_dir
            or Path(BASE_DIR) / "output" / "cache" / "asset_db"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model

        # Build the unified record list.
        self._records: List[AssetRecord] = []
        self._build_records()

        # Embedding matrix — lazily computed on first semantic query.
        self._embeddings: Optional[np.ndarray] = None
        self._embeddings_checksum: str = ""

        # Inverted tag index for fast tag_search.
        self._tag_index: Dict[str, List[int]] = {}
        self._build_tag_index()

        logger.info(
            "AssetDatabase initialized: %s records (%s taxonomy, %s vivre)",
            len(self._records),
            sum(1 for r in self._records if r.source == "taxonomy"),
            sum(1 for r in self._records if r.source == "vivre_card"),
        )

    # ── Index construction ───────────────────────────────────────────

    def _build_records(self) -> None:
        """Load and merge all asset sources into ``self._records``."""
        seen_ids: set = set()

        # 1. Taxonomy assets (primary).
        for raw in one_piece_taxonomy_assets():
            asset_id = raw.get("id", "")
            if not asset_id or asset_id in seen_ids:
                continue
            seen_ids.add(asset_id)
            self._records.append(
                AssetRecord(
                    id=asset_id,
                    name=raw.get("name", ""),
                    category=raw.get("category", ""),
                    character=raw.get("character", ""),
                    arc=raw.get("arc", ""),
                    emotion=raw.get("emotion", ""),
                    visual_type=raw.get("visual_type", ""),
                    importance=int(raw.get("importance", 3)),
                    tags=list(raw.get("tags") or []),
                    source="taxonomy",
                )
            )

        # 2. Vivre Card indexed PNGs (optional).
        self._merge_vivre_cards(seen_ids)

    def _merge_vivre_cards(self, seen_ids: set) -> None:
        """Merge Vivre Card PNG index records as lightweight AssetRecords."""
        try:
            from app.utils.expressions.expression_assets import (
                _load_index as load_vivre_index,
            )
            vivre_records = load_vivre_index()
        except Exception:
            vivre_records = []

        for rec in vivre_records:
            # Derive a stable id from the relative path.
            rel = rec.get("relative", "") or rec.get("path", "")
            asset_id = f"vivre_{_norm(rel)}" if rel else None
            if not asset_id or asset_id in seen_ids:
                continue
            seen_ids.add(asset_id)

            primary_name = rec.get("primary_name", "")
            kind = rec.get("asset_kind", "character")
            self._records.append(
                AssetRecord(
                    id=asset_id,
                    name=primary_name or Path(rec.get("path", "")).stem,
                    category=f"vivre_{kind}",
                    character=primary_name if kind == "character" else "",
                    arc="",
                    emotion="",
                    visual_type=kind,
                    importance=2,
                    tags=[kind, rec.get("parent", "")],
                    source="vivre_card",
                    file_path=rec.get("path", ""),
                )
            )

    def _build_tag_index(self) -> None:
        """Build inverted index: tag → list of record indices."""
        self._tag_index.clear()
        for idx, rec in enumerate(self._records):
            for tag in rec.tags:
                key = tag.strip().lower()
                if key:
                    self._tag_index.setdefault(key, []).append(idx)

    # ── Embedding management ─────────────────────────────────────────

    def _ensure_embeddings(self) -> np.ndarray:
        """Build or load cached embedding matrix."""
        if self._embeddings is not None:
            return self._embeddings

        from app.utils.assets.asset_embeddings import (
            embed_texts,
            load_embeddings,
            save_embeddings,
        )

        checksum = _taxonomy_checksum()
        cache_path = self._cache_dir / "asset_embeddings.npy"
        meta_path = self._cache_dir / "asset_embeddings_meta.json"

        # Try loading from cache.
        if cache_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("checksum") == checksum
                    and meta.get("count") == len(self._records)
                ):
                    matrix = load_embeddings(cache_path)
                    if matrix is not None and matrix.shape[0] == len(self._records):
                        self._embeddings = matrix
                        self._embeddings_checksum = checksum
                        return self._embeddings
            except Exception as exc:
                logger.warning("Embedding cache invalid: %s", exc)

        # Compute fresh embeddings.
        logger.info("Computing embeddings for %s asset records ...", len(self._records))
        started = time.time()
        texts = [rec.embedding_text() for rec in self._records]
        self._embeddings = embed_texts(texts, model_name=self._embedding_model)
        self._embeddings_checksum = checksum
        elapsed = time.time() - started
        logger.info("Computed %s embeddings in %.1fs", len(self._records), elapsed)

        # Persist cache.
        try:
            save_embeddings(self._embeddings, cache_path)
            meta_path.write_text(
                json.dumps({"checksum": checksum, "count": len(self._records)}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to cache embeddings: %s", exc)

        return self._embeddings

    # ── Search: semantic ─────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[ScoredAsset]:
        """Free-text semantic search over all assets.

        Embeds the query and computes cosine similarity against the
        pre-built embedding matrix.
        """
        if not query or not query.strip():
            return []

        from app.utils.assets.asset_embeddings import cosine_similarity, embed_query

        matrix = self._ensure_embeddings()
        query_vec = embed_query(query, model_name=self._embedding_model)
        scores = cosine_similarity(query_vec, matrix)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            idx = int(idx)
            sim = float(scores[idx])
            if sim <= 0:
                continue
            results.append(
                ScoredAsset(
                    asset=self._records[idx],
                    score=sim,
                    score_breakdown={"semantic": sim},
                )
            )
        return results

    # ── Search: tag ──────────────────────────────────────────────────

    def tag_search(
        self,
        tags: Sequence[str],
        top_k: int = 10,
    ) -> List[ScoredAsset]:
        """Search by tag set intersection.

        Score = |query_tags ∩ asset_tags| / |query_tags|
        """
        if not tags:
            return []

        query_tags = {t.strip().lower() for t in tags if t.strip()}
        if not query_tags:
            return []

        # Collect candidate indices via inverted index.
        candidate_hits: Dict[int, int] = {}
        for tag in query_tags:
            for idx in self._tag_index.get(tag, []):
                candidate_hits[idx] = candidate_hits.get(idx, 0) + 1

        scored: List[Tuple[float, int]] = []
        for idx, hits in candidate_hits.items():
            score = hits / len(query_tags)
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append(
                ScoredAsset(
                    asset=self._records[idx],
                    score=score,
                    score_breakdown={"tag_overlap": score},
                )
            )
        return results

    # ── Search: emotion ──────────────────────────────────────────────

    def emotion_search(
        self,
        emotions: Sequence[str],
        top_k: int = 10,
    ) -> List[ScoredAsset]:
        """Search by emotion keywords (fuzzy match against the emotion field).

        Score = matched_emotion_tokens / total_query_emotions
        """
        if not emotions:
            return []

        query_emotions = {_norm(e) for e in emotions if e.strip()}
        if not query_emotions:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, rec in enumerate(self._records):
            if not rec.emotion:
                continue
            asset_emotions = {_norm(e) for e in _split_csv(rec.emotion)}
            # Check both exact and substring match.
            hits = 0
            for qe in query_emotions:
                if qe in asset_emotions:
                    hits += 1
                elif any(qe in ae or ae in qe for ae in asset_emotions):
                    hits += 0.6
            if hits > 0:
                score = hits / len(query_emotions)
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append(
                ScoredAsset(
                    asset=self._records[idx],
                    score=score,
                    score_breakdown={"emotion_match": score},
                )
            )
        return results

    # ── Search: arc ──────────────────────────────────────────────────

    def arc_search(
        self,
        arc_name: str,
        top_k: int = 10,
    ) -> List[ScoredAsset]:
        """Search by arc name (normalized substring match).

        Exact arc name match scores 1.0, partial substring match scores 0.6.
        """
        if not arc_name or not arc_name.strip():
            return []

        query_norm = _norm(arc_name)
        query_lower = arc_name.strip().lower()

        scored: List[Tuple[float, int]] = []
        for idx, rec in enumerate(self._records):
            if not rec.arc:
                continue
            arc_parts = [a.strip().lower() for a in rec.arc.split("/")]
            arc_norms = [_norm(a) for a in arc_parts]

            score = 0.0
            if query_norm in arc_norms:
                score = 1.0
            elif any(query_norm in an for an in arc_norms):
                score = 0.7
            elif any(query_lower in a for a in arc_parts):
                score = 0.6
            elif any(an in query_norm for an in arc_norms if len(an) >= 4):
                score = 0.4

            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append(
                ScoredAsset(
                    asset=self._records[idx],
                    score=score,
                    score_breakdown={"arc_match": score},
                )
            )
        return results

    # ── Composite ranker ─────────────────────────────────────────────

    def rank_for_beat(
        self,
        beat: Dict[str, Any],
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[ScoredAsset]:
        """Rank all assets for a storyboard beat, returning the best matches.

        The beat dict should contain (all optional):
        - ``text`` — the subtitle/narration text for the beat
        - ``beat_type`` — e.g. "hook", "reveal", "evidence"
        - ``entities`` — list of entity names
        - ``emotion`` — emotion state dict or string
        - ``tags`` — extra tags to match

        Composite score (configurable via ``weights``):
            0.35 × semantic + 0.20 × tag + 0.20 × emotion
          + 0.15 × arc + 0.10 × importance

        Returns a list of ``ScoredAsset`` sorted by descending score.
        """
        w = weights or {
            "semantic": 0.35,
            "tag": 0.20,
            "emotion": 0.20,
            "arc": 0.15,
            "importance": 0.10,
        }

        text = beat.get("text", "") or ""
        entities = beat.get("entities") or []
        beat_tags = list(beat.get("tags") or [])
        emotion_raw = beat.get("emotion", "")

        # Normalize emotion input (can be dict from emotion_state or string).
        if isinstance(emotion_raw, dict):
            emotion_str = emotion_raw.get("emotion", "")
        else:
            emotion_str = str(emotion_raw)

        # Combine entity names and tags for tag matching.
        combined_tags = beat_tags + [e.lower().replace(" ", "_") for e in entities]

        # Build query text for semantic search: concatenate everything.
        query_parts = [text]
        if entities:
            query_parts.append(" ".join(entities))
        if emotion_str:
            query_parts.append(emotion_str)
        query_text = " ".join(query_parts).strip()

        # ── Compute per-component scores ─────────────────────────────

        n = len(self._records)
        semantic_scores = np.zeros(n, dtype=np.float32)
        tag_scores = np.zeros(n, dtype=np.float32)
        emotion_scores = np.zeros(n, dtype=np.float32)
        arc_scores = np.zeros(n, dtype=np.float32)
        importance_scores = np.zeros(n, dtype=np.float32)

        # Semantic.
        if query_text and w.get("semantic", 0) > 0:
            try:
                from app.utils.assets.asset_embeddings import (
                    cosine_similarity,
                    embed_query,
                )

                matrix = self._ensure_embeddings()
                query_vec = embed_query(query_text, model_name=self._embedding_model)
                raw_sim = cosine_similarity(query_vec, matrix)
                # Clamp to [0, 1].
                semantic_scores = np.clip(raw_sim, 0.0, 1.0)
            except Exception as exc:
                logger.warning("Semantic scoring failed (degrading gracefully): %s", exc)

        # Tag.
        if combined_tags and w.get("tag", 0) > 0:
            query_tag_set = {t.strip().lower() for t in combined_tags if t.strip()}
            if query_tag_set:
                for idx, rec in enumerate(self._records):
                    asset_tag_set = {t.strip().lower() for t in rec.tags}
                    overlap = len(query_tag_set & asset_tag_set)
                    if overlap > 0:
                        tag_scores[idx] = overlap / len(query_tag_set)

        # Emotion.
        if emotion_str and w.get("emotion", 0) > 0:
            query_emotions = {_norm(e) for e in _split_csv(emotion_str) if e.strip()}
            if not query_emotions:
                query_emotions = {_norm(emotion_str)}
            for idx, rec in enumerate(self._records):
                if not rec.emotion:
                    continue
                asset_emotions = {_norm(e) for e in _split_csv(rec.emotion)}
                hits = 0.0
                for qe in query_emotions:
                    if qe in asset_emotions:
                        hits += 1.0
                    elif any(qe in ae or ae in qe for ae in asset_emotions):
                        hits += 0.6
                if hits > 0:
                    emotion_scores[idx] = hits / len(query_emotions)

        # Arc.
        if w.get("arc", 0) > 0 and text:
            # Extract potential arc names from text.
            arc_candidates = set()
            from app.utils.assets.broll_rules import ARC_LOCATION_TERMS

            text_lower = text.lower()
            for term in ARC_LOCATION_TERMS:
                if term in text_lower:
                    arc_candidates.add(term)
            # Also check entities for arc names.
            for entity in entities:
                if entity.lower() in ARC_LOCATION_TERMS or _norm(entity) in {
                    _norm(t) for t in ARC_LOCATION_TERMS
                }:
                    arc_candidates.add(entity.lower())

            if arc_candidates:
                for idx, rec in enumerate(self._records):
                    if not rec.arc:
                        continue
                    arc_lower = rec.arc.lower()
                    best = 0.0
                    for candidate in arc_candidates:
                        if candidate in arc_lower:
                            best = max(best, 1.0)
                        elif _norm(candidate) in _norm(arc_lower):
                            best = max(best, 0.7)
                    arc_scores[idx] = best

        # Importance.
        if w.get("importance", 0) > 0:
            for idx, rec in enumerate(self._records):
                importance_scores[idx] = rec.importance / 5.0

        # ── Composite score ──────────────────────────────────────────

        composite = (
            w.get("semantic", 0.35) * semantic_scores
            + w.get("tag", 0.20) * tag_scores
            + w.get("emotion", 0.20) * emotion_scores
            + w.get("arc", 0.15) * arc_scores
            + w.get("importance", 0.10) * importance_scores
        )

        top_indices = np.argsort(composite)[::-1][:top_k]
        results = []
        for idx in top_indices:
            idx = int(idx)
            total = float(composite[idx])
            if total <= 0:
                continue
            results.append(
                ScoredAsset(
                    asset=self._records[idx],
                    score=total,
                    score_breakdown={
                        "semantic": float(semantic_scores[idx]),
                        "tag_overlap": float(tag_scores[idx]),
                        "emotion_match": float(emotion_scores[idx]),
                        "arc_match": float(arc_scores[idx]),
                        "importance": float(importance_scores[idx]),
                    },
                )
            )
        return results

    # ── Introspection ────────────────────────────────────────────────

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[AssetRecord]:
        return list(self._records)

    def status(self) -> Dict[str, Any]:
        """Health-check summary."""
        taxonomy_count = sum(1 for r in self._records if r.source == "taxonomy")
        vivre_count = sum(1 for r in self._records if r.source == "vivre_card")
        categories = sorted({r.category for r in self._records if r.category})
        return {
            "total_records": len(self._records),
            "taxonomy_records": taxonomy_count,
            "vivre_card_records": vivre_count,
            "categories": categories,
            "embeddings_cached": self._embeddings is not None,
            "embedding_model": self._embedding_model or "all-MiniLM-L6-v2",
            "cache_dir": str(self._cache_dir),
            "tag_index_size": len(self._tag_index),
        }


# ── Module-level singleton ───────────────────────────────────────────

_singleton: Optional[AssetDatabase] = None


def get_asset_database() -> AssetDatabase:
    """Return the module-level AssetDatabase singleton (lazy-init)."""
    global _singleton
    if _singleton is None:
        _singleton = AssetDatabase()
    return _singleton
