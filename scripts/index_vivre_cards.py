#!/usr/bin/env python3
"""Index Vivre Card PNGs from app/data/vivre-card (or VIVRE_CARD_ASSETS_DIR)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.expression_assets import build_vivre_card_index


def main():
    parser = argparse.ArgumentParser(
        description="Build .vivre_index.json for expression overlay lookup."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Override VIVRE_CARD_ASSETS_DIR (default: app/data/vivre-card)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild index even if cached")
    args = parser.parse_args()

    if args.dir:
        import os
        os.environ["VIVRE_CARD_ASSETS_DIR"] = args.dir
    else:
        from app.config import VIVRE_CARD_ASSETS_DIR
        import os
        os.environ.setdefault("VIVRE_CARD_ASSETS_DIR", str(VIVRE_CARD_ASSETS_DIR))

    records = build_vivre_card_index(force=args.force)
    print(f"Indexed {len(records)} PNG files")
    if records[:3]:
        for sample in records[:3]:
            print(f"  - {sample['relative']}")


if __name__ == "__main__":
    main()
