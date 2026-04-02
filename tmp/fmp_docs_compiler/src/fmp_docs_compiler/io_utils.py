"""Filesystem helpers for :mod:`~fmp_docs_compiler`.

Purpose:
    Centralize small file I/O utilities used by the CLI and generators.

Design:
    The functions are intentionally tiny and side-effect explicit.

Attributes:
    None.

Examples:
    ::
        >>> callable(write_text)
        True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a path.

    Args:
        path:
            Target path.

    Returns:
        None.

    Raises:
        OSError:
            Raised when directory creation fails.

    Examples:
        ::
            >>> from pathlib import Path
            >>> ensure_parent(Path('tmp/example.txt'))
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to disk.

    Args:
        path:
            Target path.
        content:
            File content.

    Returns:
        None.

    Raises:
        OSError:
            Raised when writing fails.

    Examples:
        ::
            >>> callable(write_text)
            True
    """
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk.

    Args:
        path:
            Target path.
        payload:
            JSON-serializable content.

    Returns:
        None.

    Raises:
        OSError:
            Raised when writing fails.
        TypeError:
            Raised when serialization fails.

    Examples:
        ::
            >>> callable(write_json)
            True
    """
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
