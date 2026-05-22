#!/usr/bin/env python3
"""Index Vivre Card PNGs from a local Google Drive sync folder."""

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
        help="Override VIVRE_CARD_ASSETS_DIR (folder synced from Google Drive)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild index even if cached")
    args = parser.parse_args()

    if args.dir:
        import os
        os.environ["VIVRE_CARD_ASSETS_DIR"] = args.dir

    records = build_vivre_card_index(force=args.force)
    print(f"Indexed {len(records)} PNG files")
    if records[:3]:
        for sample in records[:3]:
            print(f"  - {sample['relative']}")


if __name__ == "__main__":
    main()
