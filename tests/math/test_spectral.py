"""Tests for wraquant.math.spectral."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.spectral import (
    bandpass_filter,
    dominant_frequencies,
    fft_decompose,
    spectral_density,
    spectral_entropy,
)


class TestFFTDecompose:
    """Tests for fft_decompose."""

    def test_recovers_single_sine_frequency(self) -> None:
        """A pure sine wave should produce a single dominant frequency."""
        n = 256
        freq = 0.1  # cycles per sample
        t = np.arange(n)
        data = np.sin(2 * np.pi * freq * t)

        result = fft_decompose(data)
        # The dominant positive frequency should be close to 0.1
        idx = np.argmax(result["amplitudes"][1:]) + 1  # skip DC
        recovered_freq = result["frequencies"][idx]
        assert abs(recovered_freq - freq) < 1.0 / n

    def test_n_components_limits_output(self) -> None:
        """Passing n_components should limit the number of returned entries."""
        data = np.random.default_rng(42).standard_normal(128)
        result = fft_decompose(data, n_components=5)
        assert len(result["frequencies"]) == 5
        assert len(result["amplitudes"]) == 5
        assert len(result["phases"]) == 5
        assert len(result["power"]) == 5

    def test_power_equals_amplitude_squared(self) -> None:
        data = np.random.default_rng(0).standard_normal(64)
        result = fft_decompose(data)
        np.testing.assert_allclose(result["power"], result["amplitudes"] ** 2)


class TestDominantFrequencies:
    """Tests for dominant_frequencies."""

    def test_finds_known_periodicity(self) -> None:
        """Two sine components should both appear in the top frequencies."""
        n = 512
        t = np.arange(n)
        f1, f2 = 0.05, 0.15
        data = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

        result = dominant_frequencies(data, n_top=5)
        found_freqs = result["frequency"]
        # Both frequencies should be present (within resolution)
        assert any(abs(f - f1) < 2.0 / n for f in found_freqs)
        assert any(abs(f - f2) < 2.0 / n for f in found_freqs)

    def test_period_is_inverse_of_frequency(self) -> None:
        n = 256
        t = np.arange(n)
        data = np.sin(2 * np.pi * 0.1 * t)
        result = dominant_frequencies(data, n_top=3)
        np.testing.assert_allclose(
            result["period"],
            np.where(result["frequency"] > 0, 1.0 / result["frequency"], np.inf),
        )


class TestSpectralEntropy:
    """Tests for spectral_entropy."""

    def test_random_higher_than_periodic(self) -> None:
        """White noise should have higher spectral entropy than a sine wave."""
        rng = np.random.default_rng(123)
        n = 1024
        t = np.arange(n)

        periodic = np.sin(2 * np.pi * 0.1 * t)
        random_data = rng.standard_normal(n)

        se_periodic = spectral_entropy(periodic)
        se_random = spectral_entropy(random_data)

        assert se_random > se_periodic

    def test_normalised_between_0_and_1(self) -> None:
        data = np.random.default_rng(7).standard_normal(256)
        se = spectral_entropy(data)
        assert 0.0 <= se <= 1.0


class TestBandpassFilter:
    """Tests for bandpass_filter."""

    def test_isolates_correct_frequency_band(self) -> None:
        """A bandpass around f1 should suppress f2."""
        n = 1024
        t = np.arange(n)
        f1, f2 = 0.05, 0.25
        data = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

        filtered = bandpass_filter(data, low_freq=0.03, high_freq=0.08)

        # FFT of filtered signal should have most energy near f1
        fft_vals = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n)
        # Energy in f1 band should dominate f2 band
        f1_mask = (freqs >= 0.03) & (freqs <= 0.08)
        f2_mask = (freqs >= 0.20) & (freqs <= 0.30)
        energy_f1 = np.sum(fft_vals[f1_mask] ** 2)
        energy_f2 = np.sum(fft_vals[f2_mask] ** 2)
        assert energy_f1 > 10 * energy_f2


class TestSpectralDensity:
    """Tests for spectral_density."""

    def test_periodogram_returns_correct_keys(self) -> None:
        data = np.random.default_rng(0).standard_normal(128)
        result = spectral_density(data, method="periodogram")
        assert "frequencies" in result
        assert "psd" in result

    def test_welch_returns_correct_keys(self) -> None:
        data = np.random.default_rng(0).standard_normal(256)
        result = spectral_density(data, method="welch")
        assert "frequencies" in result
        assert "psd" in result

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            spectral_density(np.ones(64), method="unknown")
