# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for SKLDA and STDA classifiers
#
# Tests for: metabci.brainda.algorithms.decomposition.SKLDA
#             metabci.brainda.algorithms.decomposition.STDA

import numpy as np

from metabci.brainda.algorithms.decomposition.SKLDA import SKLDA
from metabci.brainda.algorithms.decomposition.STDA import STDA


class TestSKLDA:
    """Tests for Shrinkage LDA classifier."""

    def test_fit_returns_self(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(20, 10) - 1, rng.randn(20, 10) + 1])
        y = np.array([0] * 20 + [1] * 20)
        clf = SKLDA()
        result = clf.fit(X, y)
        assert result is clf

    def test_transform_shape(self):
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(20, 10) - 1, rng.randn(20, 10) + 1])
        y_train = np.array([0] * 20 + [1] * 20)
        X_test = rng.randn(5, 10)
        clf = SKLDA()
        clf.fit(X_train, y_train)
        scores = clf.transform(X_test)
        assert scores.shape == (5,)

    def test_decision_values_finite(self):
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(20, 10) - 1, rng.randn(20, 10) + 1])
        y_train = np.array([0] * 20 + [1] * 20)
        X_test = rng.randn(10, 10)
        clf = SKLDA()
        clf.fit(X_train, y_train)
        scores = clf.transform(X_test)
        assert np.all(np.isfinite(scores))

    def test_classes_attribute(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(10, 5), rng.randn(10, 5)])
        y = np.array([0] * 10 + [1] * 10)
        clf = SKLDA()
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(np.sort(clf.classes_), [0, 1])


class TestSTDA:
    """Tests for Spatial-Temporal Discriminant Analysis.

    Note: STDA inherits from TransformerMixin (and ClassifierMixin),
    but does NOT implement predict(). STDA.transform() returns
    decision values (1D array), not class labels.
    """

    def test_fit_returns_self(self):
        rng = np.random.RandomState(42)
        n_trials, n_ch, n_samples = 40, 8, 50
        X = rng.randn(n_trials, n_ch, n_samples)
        X[:20, :4, 20:40] += 2.0
        y = np.array([0] * 20 + [1] * 20)
        stda = STDA()
        result = stda.fit(X, y)
        assert result is stda

    def test_transform_reduces_dimensionality(self):
        rng = np.random.RandomState(42)
        n_trials, n_ch, n_samples = 40, 8, 50
        X = rng.randn(n_trials, n_ch, n_samples)
        X[:20, :4, 20:40] += 2.0
        y = np.array([0] * 20 + [1] * 20)
        stda = STDA()
        stda.fit(X, y)
        features = stda.transform(X)
        assert features.shape[0] == n_trials
        assert features.ndim <= 2

    def test_predict_shape(self):
        """STDA.transform() returns decision values.

        We verify shape and finiteness of decision values.
        Classification can be done by thresholding (e.g., np.sign).
        """
        rng = np.random.RandomState(42)
        n_trials, n_ch, n_samples = 40, 8, 50
        X = rng.randn(n_trials, n_ch, n_samples)
        X[:20, :4, 20:40] += 2.0
        y = np.array([0] * 20 + [1] * 20)
        stda = STDA()
        stda.fit(X, y)
        decision_values = stda.transform(X)
        assert decision_values.shape[0] == n_trials
        assert np.all(np.isfinite(decision_values))
