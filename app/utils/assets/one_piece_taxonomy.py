"""Load the curated One Piece visual asset taxonomy."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "one_piece_asset_taxonomy.json"
REQUIRED_ASSET_FIELDS = ("character", "arc", "emotion", "visual_type", "importance", "tags")


@lru_cache(maxsize=1)
def load_one_piece_asset_taxonomy() -> Dict:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def one_piece_taxonomy_assets(category: Optional[str] = None) -> List[Dict]:
    taxonomy = load_one_piece_asset_taxonomy()
    assets = list(taxonomy.get("assets") or [])
    if category:
        return [asset for asset in assets if asset.get("category") == category]
    return assets


def find_one_piece_taxonomy_asset(asset_id: str) -> Optional[Dict]:
    for asset in one_piece_taxonomy_assets():
        if asset.get("id") == asset_id:
            return asset
    return None
