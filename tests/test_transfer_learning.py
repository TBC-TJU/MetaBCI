# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for transfer learning algorithms
#
# Tests for: metabci.brainda.algorithms.transfer_learning

import numpy as np

from metabci.brainda.algorithms.transfer_learning.lst import (
    LST,
    lst_kernel,
)


class TestLstKernel:
    """Tests for the Least Squares Transformation kernel.

    lst_kernel(S, T) computes P = T @ S.T @ pinv(S @ S.T)
    where S has shape (n_source, n_features) and T has shape (n_target, n_features).
    Output P has shape (n_target, n_source).
    """

    def test_output_shape(self):
        """Projection matrix should have correct shape."""
        n_source, n_target, n_features = 5, 3, 10
        S = np.random.RandomState(42).randn(n_source, n_features)
        T = np.random.RandomState(43).randn(n_target, n_features)
        P = lst_kernel(S, T)
        # P = T @ S.T @ pinv(S @ S.T) -> shape (n_target, n_source)
        assert P.shape == (n_target, n_source)

    def test_identity_transform(self):
        """If source equals target, P @ S should approximate S."""
        rng = np.random.RandomState(42)
        S = rng.randn(20, 10)
        P = lst_kernel(S, S)
        reconstructed = P @ S
        assert np.allclose(reconstructed, S, atol=0.1)


class TestLST:
    """Tests for the LST transformer.

    LST usage pattern:
    - lst.fit(X_target, y_target)       : learns class templates from TARGET data
    - lst.transform(X_source, y_source) : transforms SOURCE data using target templates

    Note: transform() requires y (labels) to match source trials to class templates.
    """

    def test_fit_returns_self(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_target = rng.randn(10, n_ch, n_samples)
        y_target = np.array([0] * 5 + [1] * 5)

        lst = LST()
        result = lst.fit(X_target, y_target)
        assert result is lst

    def test_transform_shape(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_target = rng.randn(10, n_ch, n_samples)
        X_source = rng.randn(10, n_ch, n_samples)
        y_target = np.array([0] * 5 + [1] * 5)
        y_source = np.array([0] * 5 + [1] * 5)

        lst = LST()
        lst.fit(X_target, y_target)
        X_transformed = lst.transform(X_source, y_source)
        assert X_transformed.shape == X_source.shape

    def test_transform_finite(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_target = rng.randn(10, n_ch, n_samples)
        X_source = rng.randn(10, n_ch, n_samples)
        y_target = np.array([0] * 5 + [1] * 5)
        y_source = np.array([0] * 5 + [1] * 5)

        lst = LST()
        lst.fit(X_target, y_target)
        X_transformed = lst.transform(X_source, y_source)
        assert np.all(np.isfinite(X_transformed))

    def test_classes_attribute(self):
        """After fit, LST should store class labels."""
        rng = np.random.RandomState(42)
        X_target = rng.randn(10, 8, 250)
        y_target = np.array([0] * 5 + [1] * 5)

        lst = LST()
        lst.fit(X_target, y_target)
        assert hasattr(lst, "classes_")
        np.testing.assert_array_equal(np.sort(lst.classes_), [0, 1])
