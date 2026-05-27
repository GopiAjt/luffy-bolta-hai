"""Tests for static expression overlay assets."""

from pathlib import Path

import pytest

from app.config import EXPRESSIONS_DIR, USE_STATIC_EXPRESSIONS_ONLY
from app.utils.expressions.expression_assets import (
    list_static_expression_files,
    normalize_expression_label,
    resolve_expression_image,
    resolve_static_expression_image,
)


@pytest.fixture(scope="module")
def expressions_dir():
    if not EXPRESSIONS_DIR.is_dir():
        pytest.skip(f"Expressions folder missing: {EXPRESSIONS_DIR}")
    return EXPRESSIONS_DIR


def test_static_only_mode_enabled():
    assert USE_STATIC_EXPRESSIONS_ONLY is True


def test_expression_files_exist(expressions_dir):
    labels = list_static_expression_files(expressions_dir)
    assert "neutral" in labels
    assert "happy" in labels
    assert len(labels) >= 8


def test_normalize_aliases():
    assert normalize_expression_label("emotional") == "sad"
    assert normalize_expression_label("EXCITED") == "excited"
    assert normalize_expression_label("unknown_xyz") == "neutral"


def test_resolve_happy(expressions_dir):
    path = resolve_static_expression_image("happy", expressions_dir)
    assert path
    assert path.endswith("happy.png")


def test_resolve_via_public_api(expressions_dir):
    path = resolve_expression_image("luffy", "intense", fallback_dir=expressions_dir)
    assert path
    assert "intense.png" in path.lower()
