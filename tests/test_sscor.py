# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for SSCOR algorithms
#
# Tests for: metabci.brainda.algorithms.decomposition.sscor

import numpy as np

from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.decomposition.sscor import (
    FBSSCOR,
    SSCOR,
    sscor_kernel,
)


class TestSscorKernel:
    """Tests for the SSCOR kernel function."""

    def test_output_shapes(self, ssvep_data):
        X, y = ssvep_data
        X_class0 = X[y == 0]
        W, D, A = sscor_kernel(X_class0)
        n_channels = X_class0.shape[1]
        assert W.shape == (n_channels, n_channels)
        assert D.shape == (n_channels,)
        assert A.shape == (n_channels, n_channels)

    def test_eigenvalues_sorted(self, ssvep_data):
        X, y = ssvep_data
        X_class0 = X[y == 0]
        _, D, _ = sscor_kernel(X_class0)
        assert np.all(D[:-1] >= D[1:] - 1e-10)


class TestSSCOR:
    """Tests for the SSCOR estimator.

    Note: SSCOR is a TransformerMixin, NOT a classifier.
    It has fit() and transform() but NOT predict().

    With default transform_method=None, transform() returns
    concatenated spatial features with shape (n_trials, n_classes*n_comp*n_samples).

    With transform_method="corr", it returns per-class correlations.
    """

    def test_fit_predict(self, ssvep_data):
        """Test that SSCOR can fit and transform (no predict method)."""
        X, y = ssvep_data
        sscor = SSCOR(n_components=1)
        result = sscor.fit(X, y)
        assert result is sscor
        # SSCOR has no predict() — test transform instead
        features = sscor.transform(X)
        assert features.shape[0] == X.shape[0]
        assert features.ndim == 2

    def test_transform_shape(self, ssvep_data):
        """With transform_method='corr', returns per-class correlations."""
        X, y = ssvep_data
        n_classes = len(np.unique(y))
        sscor = SSCOR(n_components=1, transform_method="corr")
        sscor.fit(X, y)
        rhos = sscor.transform(X)
        assert rhos.shape[0] == X.shape[0]
        # With corr method, one correlation per class
        assert rhos.shape[1] == n_classes

    def test_classes_stored(self, ssvep_data):
        X, y = ssvep_data
        sscor = SSCOR(n_components=1)
        sscor.fit(X, y)
        assert hasattr(sscor, "classes_")
        np.testing.assert_array_equal(np.sort(sscor.classes_), np.sort(np.unique(y)))


class TestFBSSCOR:
    """Tests for filter bank SSCOR.

    Note: FBSSCOR is also a TransformerMixin with no predict() method.
    """

    def test_fit_predict(self, ssvep_data):
        """Test that FBSSCOR can fit and transform."""
        X, y = ssvep_data
        wp = [(5, 90), (14, 90), (22, 90)]
        ws = [(3, 92), (12, 92), (20, 92)]
        fb = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        fw = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(len(fb))])
        fbsscor = FBSSCOR(filterbank=fb, n_components=1, filterweights=fw)
        fbsscor.fit(X, y)
        features = fbsscor.transform(X)
        assert features.shape[0] == X.shape[0]
        assert features.ndim == 2
        assert np.all(np.isfinite(features))
