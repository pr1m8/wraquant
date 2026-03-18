"""Tests for wraquant.core.config."""

from __future__ import annotations

from pathlib import Path

from wraquant._compat import Backend
from wraquant.core.config import WQConfig, get_config, reset_config


class TestWQConfig:
    def test_defaults(self) -> None:
        cfg = WQConfig()
        assert cfg.backend == Backend.PANDAS
        assert cfg.cache_enabled is True
        assert cfg.float_precision == 64
        assert cfg.trading_days_per_year == 252
        assert cfg.base_currency == "USD"

    def test_cache_dir_is_path(self) -> None:
        cfg = WQConfig()
        assert isinstance(cfg.cache_dir, Path)

    def test_backend_assignment(self) -> None:
        cfg = WQConfig()
        cfg.backend = Backend.POLARS
        assert cfg.backend == Backend.POLARS


class TestGetConfig:
    def test_singleton(self) -> None:
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset(self) -> None:
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2
