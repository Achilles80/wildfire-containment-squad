"""Configuration loading: read ``config.yaml`` once and deep-merge overrides onto it."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively onto ``base``.

    Nested dicts are merged key by key; any other value in ``override`` replaces the base value.
    Neither input is modified.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(overrides: dict[str, Any] | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Load the default configuration and apply ``overrides`` on top of it."""
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    return deep_merge(base, overrides)
