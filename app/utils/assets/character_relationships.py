"""Character relationship engine for narrative-aware relationship ranking.

Builds a weighted relationship graph from the One Piece asset taxonomy
(tag cross-references, arc co-occurrence, emotion similarity, category
proximity, and explicit character mentions) and optionally enriches it
with narration context.

Example::

    >>> engine = get_relationship_engine()
    >>> for r in engine.get_relationships("Blackbeard", top_k=5):
    ...     print(f"{r.target:25s}  {r.score:.2f}  {r.relationship}")
    Portgas D. Ace             0.87  opposes
    Edward Newgate [Whitebeard] 0.82  opposes
    Monkey D. Luffy            0.71  rivals
    Shanks                     0.65  rivals
    Aokiji                     0.58  allies_with
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.utils.assets.one_piece_taxonomy import (
    load_one_piece_asset_taxonomy,
    one_piece_taxonomy_assets,
)

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class CharacterNode:
    """A character in the relationship graph."""

    id: str
    name: str
    category: str = ""
    character: str = ""       # full canonical name
    arc: str = ""
    emotion: str = ""
    visual_type: str = ""
    importance: int = 3
    tags: List[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.character or self.name

    @property
    def arc_set(self) -> Set[str]:
        return {a.strip().lower() for a in self.arc.split("/") if a.strip()}

    @property
    def emotion_set(self) -> Set[str]:
        return {e.strip().lower() for e in self.emotion.split(",") if e.strip()}

    @property
    def tag_set(self) -> Set[str]:
        return {t.strip().lower() for t in self.tags}


@dataclass
class RankedRelationship:
    """A ranked relationship between two characters."""

    source: str           # display name of the query character
    target: str           # display name of the related character
    target_id: str        # taxonomy id
    score: float          # composite relationship strength [0, 1]
    relationship: str     # type label (e.g. "rivals", "allies_with", "family")
    evidence: List[str] = field(default_factory=list)  # human-readable reasons
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "target_id": self.target_id,
            "score": round(self.score, 4),
            "relationship": self.relationship,
            "evidence": self.evidence,
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


# ── Character name resolution ───────────────────────────────────────

# Maps short names / aliases → canonical taxonomy ids.
_NAME_ALIASES: Dict[str, str] = {
    "luffy": "straw_hat_luffy",
    "monkey d luffy": "straw_hat_luffy",
    "monkey d. luffy": "straw_hat_luffy",
    "straw hat": "straw_hat_luffy",
    "zoro": "straw_hat_zoro",
    "roronoa zoro": "straw_hat_zoro",
    "sanji": "straw_hat_sanji",
    "nami": "straw_hat_nami",
    "usopp": "straw_hat_usopp",
    "chopper": "straw_hat_chopper",
    "robin": "straw_hat_robin",
    "nico robin": "straw_hat_robin",
    "franky": "straw_hat_franky",
    "brook": "straw_hat_brook",
    "jinbe": "straw_hat_jinbe",
    "jimbei": "straw_hat_jinbe",
    "shanks": "yonko_shanks",
    "blackbeard": "yonko_blackbeard",
    "teach": "yonko_blackbeard",
    "marshall d. teach": "yonko_blackbeard",
    "buggy": "yonko_buggy",
    "kaido": "yonko_kaido",
    "kaidou": "yonko_kaido",
    "big mom": "yonko_big_mom",
    "charlotte linlin": "yonko_big_mom",
    "whitebeard": "yonko_whitebeard",
    "edward newgate": "yonko_whitebeard",
    "akainu": "admiral_akainu",
    "sakazuki": "admiral_akainu",
    "kizaru": "admiral_kizaru",
    "borsalino": "admiral_kizaru",
    "aokiji": "admiral_aokiji",
    "kuzan": "admiral_aokiji",
    "fujitora": "admiral_fujitora",
    "issho": "admiral_fujitora",
    "ryokugyu": "admiral_ryokugyu",
    "aramaki": "admiral_ryokugyu",
    "imu": "imu",
    "dragon": "rev_dragon",
    "monkey d. dragon": "rev_dragon",
    "sabo": "rev_sabo",
    "roger": "legend_roger",
    "gol d. roger": "legend_roger",
    "gol d roger": "legend_roger",
    "rayleigh": "legend_rayleigh",
    "garp": "legend_garp",
    "monkey d. garp": "legend_garp",
    "ace": "arc_marineford",  # Ace has no standalone entry; Marineford arc is his anchor
    "portgas d. ace": "arc_marineford",
    "mihawk": "warlord_mihawk",
    "dracule mihawk": "warlord_mihawk",
    "crocodile": "warlord_crocodile",
    "doflamingo": "warlord_doflamingo",
    "donquixote doflamingo": "warlord_doflamingo",
    "kuma": "warlord_kuma",
    "bartholomew kuma": "warlord_kuma",
    "hancock": "warlord_hancock",
    "boa hancock": "warlord_hancock",
    "law": "supernova_law",
    "trafalgar law": "supernova_law",
    "kid": "supernova_kid",
    "eustass kid": "supernova_kid",
    "bonney": "supernova_bonney",
    "jewelry bonney": "supernova_bonney",
    "vegapunk": "lore_vegapunk",
    "joy boy": "lore_joyboy",
    "oden": "legend_oden",
    "kozuki oden": "legend_oden",
    "garling": "holy_knight_garling",
    "shamrock": "holy_knight_shamrock",
    "saturn": "five_elders_saturn",
    "sengoku": "legend_sengoku",
    "xebec": "legend_xebec",
    "rocks": "legend_xebec",
}

# ── Explicit relationship knowledge ─────────────────────────────────

# Hand-curated high-signal relationships that tag/arc mining alone would
# underweight.  Format: (source_id, target_id, type, weight, evidence).
_CANONICAL_EDGES: List[Tuple[str, str, str, float, str]] = [
    # Blackbeard's key relationships
    ("yonko_blackbeard", "arc_marineford", "opposes", 0.45, "Blackbeard captured Ace; pivotal Marineford trigger"),
    ("yonko_blackbeard", "yonko_whitebeard", "opposes", 0.50, "Blackbeard killed Whitebeard and stole the Gura Gura no Mi"),
    ("yonko_blackbeard", "straw_hat_luffy", "rivals", 0.40, "Final villain parallel; both chase One Piece"),
    ("yonko_blackbeard", "yonko_shanks", "rivals", 0.38, "Scarred Shanks; Shanks warned the Gorosei about him"),
    ("yonko_blackbeard", "admiral_aokiji", "allies_with", 0.28, "Aokiji joined Blackbeard Pirates post-timeskip"),
    # Luffy core bonds
    ("straw_hat_luffy", "yonko_shanks", "mentor_link", 0.48, "Shanks inspired Luffy and entrusted the straw hat"),
    ("straw_hat_luffy", "arc_marineford", "family_or_lineage", 0.45, "Ace is Luffy's sworn brother; died at Marineford"),
    ("straw_hat_luffy", "rev_sabo", "family_or_lineage", 0.42, "Sabo is Luffy's sworn brother"),
    ("straw_hat_luffy", "rev_dragon", "family_or_lineage", 0.35, "Dragon is Luffy's father"),
    ("straw_hat_luffy", "legend_garp", "family_or_lineage", 0.38, "Garp is Luffy's grandfather"),
    ("straw_hat_luffy", "yonko_kaido", "opposes", 0.40, "Luffy defeated Kaido at Onigashima"),
    ("straw_hat_luffy", "yonko_blackbeard", "rivals", 0.40, "Final villain; Blackbeard is Luffy's foil on dreams"),
    # Zoro key relationships
    ("straw_hat_zoro", "warlord_mihawk", "rivals", 0.48, "Zoro's ultimate goal is to surpass Mihawk"),
    ("straw_hat_zoro", "straw_hat_luffy", "allies_with", 0.45, "First mate loyalty; Nothing Happened"),
    ("straw_hat_zoro", "straw_hat_sanji", "rivals", 0.35, "Constant rivalry within the crew"),
    # Ace / Marineford connections
    ("arc_marineford", "yonko_whitebeard", "allies_with", 0.50, "Whitebeard went to war to save Ace"),
    ("arc_marineford", "admiral_akainu", "opposes", 0.48, "Akainu killed Ace at Marineford"),
    ("arc_marineford", "straw_hat_luffy", "family_or_lineage", 0.45, "Ace died protecting Luffy"),
    ("arc_marineford", "rev_sabo", "family_or_lineage", 0.38, "Sabo inherited Ace's will and fruit"),
    # Sanji
    ("straw_hat_sanji", "yonko_big_mom", "opposes", 0.35, "Whole Cake Island confrontation"),
    ("straw_hat_sanji", "straw_hat_luffy", "allies_with", 0.42, "Nothing happened at Whole Cake; 'I want to return'"),
    # Robin
    ("straw_hat_robin", "straw_hat_luffy", "allies_with", 0.42, "'I want to live!' — Enies Lobby rescue"),
    # Whitebeard
    ("yonko_whitebeard", "legend_roger", "rivals", 0.42, "Roger and Whitebeard were peers; God Valley"),
    ("yonko_whitebeard", "yonko_blackbeard", "opposes", 0.50, "Blackbeard betrayed and killed Whitebeard"),
    # Shanks
    ("yonko_shanks", "straw_hat_luffy", "mentor_link", 0.48, "Bet his arm on the new era"),
    ("yonko_shanks", "yonko_blackbeard", "rivals", 0.38, "Scarred by Blackbeard; tried to warn the world"),
    ("yonko_shanks", "holy_knight_garling", "family_or_lineage", 0.30, "Figarland bloodline connection"),
    ("yonko_shanks", "legend_roger", "mentor_link", 0.35, "Shanks was on Roger's ship as a child"),
    # Roger era
    ("legend_roger", "legend_rayleigh", "allies_with", 0.45, "First mate of the Roger Pirates"),
    ("legend_roger", "legend_oden", "allies_with", 0.40, "Oden sailed with Roger to Laugh Tale"),
    ("legend_roger", "legend_garp", "rivals", 0.38, "Rivals who respected each other; God Valley"),
    ("legend_roger", "legend_xebec", "opposes", 0.42, "Roger and Garp defeated Xebec at God Valley"),
    # Admirals
    ("admiral_akainu", "admiral_aokiji", "opposes", 0.40, "Fought for Fleet Admiral position at Punk Hazard"),
    ("admiral_kizaru", "five_elders_saturn", "allies_with", 0.30, "Kizaru served under Saturn at Egghead"),
    # Vegapunk / Egghead
    ("lore_vegapunk", "five_elders_saturn", "opposes", 0.38, "Saturn came to silence Vegapunk at Egghead"),
    ("lore_vegapunk", "warlord_kuma", "allies_with", 0.35, "Vegapunk modified Kuma; felt guilt"),
    ("supernova_bonney", "warlord_kuma", "family_or_lineage", 0.42, "Bonney is Kuma's daughter"),
    # Imu
    ("imu", "lore_lily", "opposes", 0.35, "Imu's obsession with Lily/Vivi lineage"),
    ("imu", "straw_hat_luffy", "opposes", 0.30, "Imu is the final shadow ruler opposing Luffy"),
    # Law
    ("supernova_law", "warlord_doflamingo", "opposes", 0.42, "Law's entire arc is about taking down Doflamingo"),
    ("supernova_law", "straw_hat_luffy", "allies_with", 0.38, "Pirate Alliance from Punk Hazard onward"),
]

# Tag tokens that indicate a relationship type when found in narration.
_NARRATION_RELATIONSHIP_SIGNALS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bvs\.?|versus|fight|defeat|clash|oppos|kill|attack", re.I), "opposes"),
    (re.compile(r"\bcrew|protect|save|friend|ally|loyal|trust|together|join", re.I), "allies_with"),
    (re.compile(r"\brival|compete|surpass|stronger|challenge", re.I), "rivals"),
    (re.compile(r"\btrain|teacher|student|mentor|learn|taught|inspire", re.I), "mentor_link"),
    (re.compile(r"\bfather|son|brother|sister|family|lineage|inherit|blood|child|mother", re.I), "family_or_lineage"),
    (re.compile(r"\bbetray|manipulat|use|exploit|trick|decei", re.I), "exploits"),
    (re.compile(r"\bcontrast|unlike|opposite|mirror|foil|parallel", re.I), "contrasts"),
]

# Generic tags to ignore when computing tag overlap (they're too common).
_GENERIC_TAGS = frozenset({
    "major_arc", "straw_hat", "yonko", "admiral", "marine", "marines",
    "world_government", "five_elders", "gorosei", "holy_knights",
    "revolutionary", "commander", "warlord",
})

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


# ── CharacterRelationshipEngine ─────────────────────────────────────


class CharacterRelationshipEngine:
    """Builds and queries a character relationship graph.

    Data sources for edges:
    1. **Canonical edges** — hand-curated high-signal relationships
    2. **Tag cross-references** — character A's tags mention character B
    3. **Arc co-occurrence** — characters sharing arcs
    4. **Emotion similarity** — characters with overlapping emotional profiles
    5. **Category proximity** — characters in the same faction/group
    6. **Narration context** — characters co-occurring in narration text (optional)
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, CharacterNode] = {}
        # edges[source_id][target_id] = {type: str, weight: float, evidence: [str]}
        self._edges: Dict[str, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"type": "associated_with", "weight": 0.0, "evidence": []})
        )
        self._name_index: Dict[str, str] = {}  # normalized name → node id

        self._build_nodes()
        self._build_name_index()
        self._build_edges()

        logger.info(
            "CharacterRelationshipEngine: %s nodes, %s directed edges",
            len(self._nodes),
            sum(len(targets) for targets in self._edges.values()),
        )

    # ── Graph construction ───────────────────────────────────────────

    def _build_nodes(self) -> None:
        for raw in one_piece_taxonomy_assets():
            asset_id = raw.get("id", "")
            if not asset_id:
                continue
            self._nodes[asset_id] = CharacterNode(
                id=asset_id,
                name=raw.get("name", ""),
                category=raw.get("category", ""),
                character=raw.get("character", ""),
                arc=raw.get("arc", ""),
                emotion=raw.get("emotion", ""),
                visual_type=raw.get("visual_type", ""),
                importance=int(raw.get("importance", 3)),
                tags=list(raw.get("tags") or []),
            )

    def _build_name_index(self) -> None:
        """Map every name variant to a node id for fast resolution."""
        # From the alias table.
        for alias, node_id in _NAME_ALIASES.items():
            self._name_index[_norm(alias)] = node_id

        # From taxonomy node names.
        for node_id, node in self._nodes.items():
            for name in (node.name, node.character):
                if name:
                    key = _norm(name)
                    if key and key not in self._name_index:
                        self._name_index[key] = node_id
            # Also index the id itself (e.g., "straw_hat_luffy").
            self._name_index[_norm(node_id)] = node_id

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float,
        evidence: str,
    ) -> None:
        if source_id == target_id:
            return
        if source_id not in self._nodes or target_id not in self._nodes:
            return
        edge = self._edges[source_id][target_id]
        edge["weight"] += weight
        if rel_type != "associated_with":
            # Stronger types override weaker ones.
            type_priority = {
                "opposes": 6, "rivals": 5, "family_or_lineage": 5,
                "mentor_link": 4, "allies_with": 4, "exploits": 3,
                "contrasts": 2, "associated_with": 1,
            }
            if type_priority.get(rel_type, 0) >= type_priority.get(edge["type"], 0):
                edge["type"] = rel_type
        if evidence and evidence not in edge["evidence"]:
            edge["evidence"].append(evidence)

    def _build_edges(self) -> None:
        """Build the full relationship graph from all data sources."""
        self._add_canonical_edges()
        self._add_tag_crossref_edges()
        self._add_arc_cooccurrence_edges()
        self._add_emotion_similarity_edges()
        self._add_category_proximity_edges()

    def _add_canonical_edges(self) -> None:
        for source_id, target_id, rel_type, weight, evidence in _CANONICAL_EDGES:
            self._add_edge(source_id, target_id, rel_type, weight, evidence)
            # Add reverse edge with reduced weight.
            self._add_edge(target_id, source_id, rel_type, weight * 0.8, evidence)

    def _add_tag_crossref_edges(self) -> None:
        """If character A's tags contain a token matching character B's name → edge."""
        # Build a lookup: normalized name fragment → list of node ids.
        name_tokens: Dict[str, List[str]] = defaultdict(list)
        for node_id, node in self._nodes.items():
            for name in (node.name, node.character):
                for token in (name or "").lower().split():
                    key = _norm(token)
                    if key and len(key) >= 3 and key not in _GENERIC_TAGS:
                        if node_id not in name_tokens[key]:
                            name_tokens[key].append(node_id)

        for node_id, node in self._nodes.items():
            for tag in node.tags:
                tag_norm = _norm(tag)
                if tag_norm in _GENERIC_TAGS or len(tag_norm) < 3:
                    continue
                for target_id in name_tokens.get(tag_norm, []):
                    if target_id != node_id:
                        self._add_edge(
                            node_id, target_id, "associated_with", 0.12,
                            f"Tag '{tag}' references {self._nodes[target_id].name}",
                        )

    def _add_arc_cooccurrence_edges(self) -> None:
        """Characters sharing arcs get a co-occurrence edge."""
        # Only consider character-type nodes (not arc/location entries).
        char_nodes = [
            (nid, node) for nid, node in self._nodes.items()
            if node.visual_type and "arc" not in node.visual_type.lower()
            and "weapon" not in node.visual_type.lower()
        ]

        for i, (id_a, node_a) in enumerate(char_nodes):
            arcs_a = node_a.arc_set
            if not arcs_a:
                continue
            for j in range(i + 1, len(char_nodes)):
                id_b, node_b = char_nodes[j]
                arcs_b = node_b.arc_set
                shared = arcs_a & arcs_b
                if shared:
                    weight = min(0.25, 0.08 * len(shared))
                    evidence = f"Shared arcs: {', '.join(sorted(shared)[:3])}"
                    self._add_edge(id_a, id_b, "associated_with", weight, evidence)
                    self._add_edge(id_b, id_a, "associated_with", weight, evidence)

    def _add_emotion_similarity_edges(self) -> None:
        """Characters with overlapping emotions get a weak edge."""
        char_nodes = [
            (nid, node) for nid, node in self._nodes.items()
            if node.emotion and "arc" not in (node.visual_type or "").lower()
        ]

        for i, (id_a, node_a) in enumerate(char_nodes):
            emo_a = node_a.emotion_set
            if not emo_a:
                continue
            for j in range(i + 1, len(char_nodes)):
                id_b, node_b = char_nodes[j]
                emo_b = node_b.emotion_set
                shared = emo_a & emo_b
                if len(shared) >= 2:
                    weight = min(0.15, 0.05 * len(shared))
                    evidence = f"Shared emotions: {', '.join(sorted(shared)[:3])}"
                    self._add_edge(id_a, id_b, "contrasts", weight, evidence)
                    self._add_edge(id_b, id_a, "contrasts", weight, evidence)

    def _add_category_proximity_edges(self) -> None:
        """Characters in the same category (e.g., straw_hats) get crew edges."""
        by_category: Dict[str, List[str]] = defaultdict(list)
        for nid, node in self._nodes.items():
            if node.category and "arc" not in node.category:
                by_category[node.category].append(nid)

        for category, ids in by_category.items():
            if len(ids) < 2 or len(ids) > 20:
                continue
            for i, id_a in enumerate(ids):
                for id_b in ids[i + 1:]:
                    self._add_edge(
                        id_a, id_b, "allies_with", 0.10,
                        f"Same group: {category}",
                    )
                    self._add_edge(
                        id_b, id_a, "allies_with", 0.10,
                        f"Same group: {category}",
                    )

    # ── Name resolution ──────────────────────────────────────────────

    def resolve_character(self, name: str) -> Optional[str]:
        """Resolve a character name/alias to a taxonomy node id."""
        key = _norm(name)
        if not key:
            return None

        # Direct match.
        if key in self._name_index:
            return self._name_index[key]

        # Substring match (e.g., "teach" inside "marshall d teach").
        for indexed_key, node_id in self._name_index.items():
            if key in indexed_key or indexed_key in key:
                return node_id

        return None

    # ── Query API ────────────────────────────────────────────────────

    def get_relationships(
        self,
        character: str,
        top_k: int = 10,
        narration: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RankedRelationship]:
        """Return ranked relationships for a character.

        Parameters
        ----------
        character : str
            Character name or alias (e.g., "Blackbeard", "Luffy", "Zoro").
        top_k : int
            Maximum number of results.
        narration : str, optional
            Narration/script text to boost characters mentioned alongside
            the query character.
        min_score : float
            Minimum score threshold.

        Returns
        -------
        list of RankedRelationship
            Sorted by descending score.
        """
        node_id = self.resolve_character(character)
        if not node_id:
            logger.warning("Character '%s' not found in taxonomy", character)
            return []

        source_node = self._nodes[node_id]
        targets = self._edges.get(node_id, {})

        if not targets:
            logger.info("No relationships found for '%s' (%s)", character, node_id)
            return []

        # Compute narration boost if provided.
        narration_scores: Dict[str, float] = {}
        narration_types: Dict[str, str] = {}
        if narration:
            narration_scores, narration_types = self._score_narration(
                node_id, narration
            )

        # Build ranked list.
        results: List[RankedRelationship] = []
        for target_id, edge_data in targets.items():
            target_node = self._nodes.get(target_id)
            if not target_node:
                continue

            base_weight = edge_data["weight"]
            narration_boost = narration_scores.get(target_id, 0.0)
            importance_boost = target_node.importance / 5.0 * 0.10

            raw_score = base_weight + narration_boost + importance_boost
            # Normalize to [0, 1].
            score = min(1.0, max(0.0, raw_score))

            # Determine relationship type.
            rel_type = edge_data["type"]
            if narration_types.get(target_id):
                rel_type = narration_types[target_id]

            evidence = list(edge_data["evidence"])
            if narration_boost > 0:
                evidence.append(f"Narration co-occurrence (+{narration_boost:.2f})")

            if score >= min_score:
                results.append(
                    RankedRelationship(
                        source=source_node.display_name,
                        target=target_node.display_name,
                        target_id=target_id,
                        score=score,
                        relationship=rel_type,
                        evidence=evidence[:5],
                        score_breakdown={
                            "graph_weight": round(base_weight, 4),
                            "narration_boost": round(narration_boost, 4),
                            "importance_boost": round(importance_boost, 4),
                        },
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _score_narration(
        self,
        source_id: str,
        narration: str,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Score how strongly each character co-occurs with the source in narration."""
        scores: Dict[str, float] = {}
        types: Dict[str, str] = {}
        text_lower = narration.lower()

        for node_id, node in self._nodes.items():
            if node_id == source_id:
                continue

            # Check if the character is mentioned in narration.
            mentioned = False
            for name in (node.name, node.character):
                if not name:
                    continue
                # Use word boundary matching for names ≥3 chars.
                pattern = r"\b" + re.escape(name.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    mentioned = True
                    break

            if not mentioned:
                # Also check short aliases.
                for alias, alias_id in _NAME_ALIASES.items():
                    if alias_id == node_id and len(alias) >= 3:
                        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                            mentioned = True
                            break

            if mentioned:
                # Base co-occurrence score.
                scores[node_id] = 0.20

                # Bonus: check relationship signals in narration.
                for pattern, rel_type in _NARRATION_RELATIONSHIP_SIGNALS:
                    if pattern.search(narration):
                        scores[node_id] += 0.08
                        types[node_id] = rel_type
                        break

        return scores, types

    # ── Graph queries ────────────────────────────────────────────────

    def get_mutual_connections(
        self,
        char_a: str,
        char_b: str,
        top_k: int = 5,
    ) -> List[RankedRelationship]:
        """Find characters connected to both A and B (narrative bridges)."""
        id_a = self.resolve_character(char_a)
        id_b = self.resolve_character(char_b)
        if not id_a or not id_b:
            return []

        targets_a = set(self._edges.get(id_a, {}).keys())
        targets_b = set(self._edges.get(id_b, {}).keys())
        mutual = targets_a & targets_b - {id_a, id_b}

        results = []
        for mid in mutual:
            node = self._nodes.get(mid)
            if not node:
                continue
            edge_a = self._edges[id_a][mid]
            edge_b = self._edges[id_b][mid]
            score = (edge_a["weight"] + edge_b["weight"]) / 2
            results.append(
                RankedRelationship(
                    source=f"{self._nodes[id_a].display_name} ↔ {self._nodes[id_b].display_name}",
                    target=node.display_name,
                    target_id=mid,
                    score=min(1.0, score),
                    relationship="mutual_connection",
                    evidence=edge_a["evidence"][:2] + edge_b["evidence"][:2],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def list_characters(self) -> List[Dict[str, Any]]:
        """List all indexed characters with their node ids."""
        return [
            {
                "id": node.id,
                "name": node.display_name,
                "category": node.category,
                "importance": node.importance,
                "relationship_count": len(self._edges.get(node.id, {})),
            }
            for node in sorted(self._nodes.values(), key=lambda n: -n.importance)
        ]

    # ── Introspection ────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        total_edges = sum(len(t) for t in self._edges.values())
        return {
            "node_count": len(self._nodes),
            "edge_count": total_edges,
            "categories": sorted({n.category for n in self._nodes.values() if n.category}),
            "resolvable_aliases": len(self._name_index),
        }


# ── Module-level singleton ───────────────────────────────────────────

_singleton: Optional[CharacterRelationshipEngine] = None


def get_relationship_engine() -> CharacterRelationshipEngine:
    """Return the module-level singleton (lazy-init)."""
    global _singleton
    if _singleton is None:
        _singleton = CharacterRelationshipEngine()
    return _singleton
