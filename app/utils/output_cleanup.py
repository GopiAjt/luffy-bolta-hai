import logging
import os
import time
from pathlib import Path
from typing import Iterable

from app.config import COMPILED_VIDEO_DIR, IMAGE_SLIDES_DIR, UPLOADS_DIR

logger = logging.getLogger(__name__)

OUTPUT_DIRS = (UPLOADS_DIR, IMAGE_SLIDES_DIR, COMPILED_VIDEO_DIR)
DEFAULT_MAX_AGE_HOURS = 24


def _iter_output_files(output_dirs: Iterable[Path] = OUTPUT_DIRS):
    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            continue
        for path in output_dir.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                yield path


def get_output_usage(output_dirs: Iterable[Path] = OUTPUT_DIRS) -> dict:
    files = list(_iter_output_files(output_dirs))
    total_bytes = sum(path.stat().st_size for path in files if path.exists())
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def cleanup_output(max_age_hours: float = DEFAULT_MAX_AGE_HOURS, force: bool = False) -> dict:
    """
    Delete generated output files.

    By default this only removes files older than max_age_hours. Set force=True
    to remove all generated files in the configured output directories.
    """
    max_age_seconds = max(0, float(max_age_hours)) * 3600
    now = time.time()
    deleted = []
    skipped = []
    errors = []
    bytes_deleted = 0

    for path in _iter_output_files():
        try:
            stat = path.stat()
            age_seconds = now - stat.st_mtime
            if not force and age_seconds < max_age_seconds:
                skipped.append(str(path))
                continue

            size = stat.st_size
            path.unlink()
            deleted.append(str(path))
            bytes_deleted += size
            logger.info("Deleted output file: %s", path)
        except OSError as exc:
            logger.warning("Failed to delete output file %s: %s", path, exc)
            errors.append({"path": str(path), "error": str(exc)})

    _remove_empty_dirs()

    return {
        "deleted_count": len(deleted),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "bytes_deleted": bytes_deleted,
        "mb_deleted": round(bytes_deleted / (1024 * 1024), 2),
        "deleted": deleted,
        "errors": errors,
        "remaining": get_output_usage(),
    }


def _remove_empty_dirs():
    for output_dir in OUTPUT_DIRS:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            continue
        for root, dirs, _files in os.walk(output_dir, topdown=False):
            for dirname in dirs:
                path = Path(root) / dirname
                try:
                    path.rmdir()
                    logger.info("Removed empty output directory: %s", path)
                except OSError:
                    pass
