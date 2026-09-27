"""Shared test fixtures: make the project root importable and provide the default config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import load_config  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    """A fresh copy of the default configuration."""
    return load_config()
