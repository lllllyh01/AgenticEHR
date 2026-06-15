"""Configuration loading.

A thin, dependency-light wrapper around a YAML file. We intentionally keep the
config as nested dicts (with dotted-path access) rather than a rigid schema so
benchmark-specific fields can be added without code changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class Config:
    """Dotted-path accessor over a nested config dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", path)
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


_MISSING = object()
