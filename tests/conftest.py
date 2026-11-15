# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Shared test fixtures for MetaBCI test suite
#
# This file provides reusable synthetic EEG data fixtures for all test modules.
# No real data is downloaded — all data is generated deterministically with fixed seeds.

import numpy as np
import pytest

# ============================================================================
# Constants
# ============================================================================
RANDOM_SEED = 42
N_CHANNELS = 8
N_SAMPLES = 250  # 1 second at 250 Hz
SRATE = 250
N_CLASSES_2 = 2  # binary classification
N_CLASSES_4 = 4  # multi-class
N_TRIALS_PER_CLASS = 20


# ============================================================================
# Helper functions for synthetic EEG generation
# ============================================================================


def _make_motor_imagery_data(
    n_trials_per_class, n_channels, n_samples, n_classes, seed
):
    """Generate synthetic Motor Imagery (MI) EEG data.

    Creates two-class data where class 0 has more power in the first few channels
    and class 1 has more power in the last few channels (mimicking CSP-separable data).
    """
    rng = np.random.RandomState(seed)
    trials_list = []
    labels_list = []

    for c in range(n_classes):
        for _ in range(n_trials_per_class):
            # Base signal: pink-ish noise
            signal = rng.randn(n_channels, n_samples)
            # Add class-specific spatial pattern
            if c == 0:
                signal[: n_channels // 2, :] += 2.0 * rng.randn(
                    n_channels // 2, n_samples
                )
            else:
                signal[n_channels // 2 :, :] += 2.0 * rng.randn(
                    n_channels - n_channels // 2, n_samples
                )
            trials_list.append(signal)
            labels_list.append(c)

    X = np.array(trials_list)
    y = np.array(labels_list)
    return X, y


def _make_ssvep_data(n_trials_per_class, n_channels, n_samples, srate, freqs, seed):
    """Generate synthetic SSVEP EEG data.

    Creates data with embedded sinusoidal components at target frequencies
    plus Gaussian noise, mimicking SSVEP responses.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, n_samples / srate, n_samples, endpoint=False)
    trials_list = []
    labels_list = []

    for c, freq in enumerate(freqs):
        for _ in range(n_trials_per_class):
            # Base noise
            noise = 0.5 * rng.randn(n_channels, n_samples)
            # SSVEP signal (shared across channels with random mixing)
            ssvep_signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(
                2 * np.pi * 2 * freq * t
            )
            mixing = rng.randn(n_channels, 1) * 0.3 + 1.0
            signal = noise + mixing * ssvep_signal[np.newaxis, :]
            trials_list.append(signal)
            labels_list.append(c)

    X = np.array(trials_list)
    y = np.array(labels_list)
    return X, y


def _make_cca_references(freqs, srate, n_samples, n_harmonics=3):
    """Generate sine-cosine reference signals for CCA-based methods.

    Returns shape (n_classes, 2*n_harmonics, n_samples).
    """
    T = n_samples / srate
    t = np.linspace(0, T, n_samples, endpoint=False)
    Yf = []
    for freq in freqs:
        refs = []
        for h in range(1, n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        Yf.append(np.array(refs))
    return np.array(Yf)


def _make_p300_data(n_trials_per_class, n_channels, n_samples, seed):
    """Generate synthetic P300 ERP data.

    Class 1 (target) has a positive bump around 300ms, class 0 (non-target) is noise.
    """
    rng = np.random.RandomState(seed)
    trials_list = []
    labels_list = []

    # P300 template: Gaussian bump around sample 75 (=300ms at 250Hz)
    t = np.arange(n_samples)
    p300_template = 3.0 * np.exp(-0.5 * ((t - 75) / 15) ** 2)

    for c in range(2):
        for _ in range(n_trials_per_class):
            signal = 0.8 * rng.randn(n_channels, n_samples)
            if c == 1:
                # Add P300 component to central channels
                signal[n_channels // 4 : 3 * n_channels // 4, :] += p300_template[
                    np.newaxis, :
                ]
            trials_list.append(signal)
            labels_list.append(c)

    X = np.array(trials_list)
    y = np.array(labels_list)
    return X, y


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def rng():
    """Deterministic random number generator."""
    return np.random.RandomState(RANDOM_SEED)


@pytest.fixture(scope="session")
def mi_data_2class():
    """Synthetic 2-class Motor Imagery data.

    Returns (X, y) where X has shape (40, 8, 250) and y has shape (40,).
    """
    return _make_motor_imagery_data(
        N_TRIALS_PER_CLASS, N_CHANNELS, N_SAMPLES, N_CLASSES_2, RANDOM_SEED
    )


@pytest.fixture(scope="session")
def mi_data_4class():
    """Synthetic 4-class Motor Imagery data.

    Returns (X, y) where X has shape (80, 8, 250) and y has shape (80,).
    """
    return _make_motor_imagery_data(
        N_TRIALS_PER_CLASS, N_CHANNELS, N_SAMPLES, N_CLASSES_4, RANDOM_SEED
    )


@pytest.fixture(scope="session")
def ssvep_freqs():
    """SSVEP target frequencies (Hz)."""
    return [8.0, 10.0, 12.0, 15.0]


@pytest.fixture(scope="session")
def ssvep_data(ssvep_freqs):
    """Synthetic SSVEP data with 4 frequency classes.

    Returns (X, y) where X has shape (80, 8, 250) and y has shape (80,).
    """
    return _make_ssvep_data(
        N_TRIALS_PER_CLASS, N_CHANNELS, N_SAMPLES, SRATE, ssvep_freqs, RANDOM_SEED
    )


@pytest.fixture(scope="session")
def cca_references(ssvep_freqs):
    """Sine-cosine reference signals for CCA.

    Returns Yf with shape (4, 6, 250) for 4 frequencies and 3 harmonics.
    """
    return _make_cca_references(ssvep_freqs, SRATE, N_SAMPLES, n_harmonics=3)


@pytest.fixture(scope="session")
def p300_data():
    """Synthetic P300 ERP data.

    Returns (X, y) where X has shape (40, 8, 250) and y has shape (40,).
    """
    return _make_p300_data(N_TRIALS_PER_CLASS, N_CHANNELS, N_SAMPLES, RANDOM_SEED)


@pytest.fixture(scope="session")
def spd_matrices():
    """Generate a set of random Symmetric Positive Definite (SPD) matrices.

    Returns array of shape (20, 8, 8).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n_matrices = 20
    n_dim = N_CHANNELS
    matrices = []
    for _ in range(n_matrices):
        A = rng.randn(n_dim, n_dim)
        matrices.append(A @ A.T + np.eye(n_dim))
    return np.array(matrices)
