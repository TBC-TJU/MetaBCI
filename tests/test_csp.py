# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for CSP algorithms
#
# Tests for: metabci.brainda.algorithms.decomposition.csp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from metabci.brainda.algorithms.decomposition.csp import (
    CSP,
    MultiCSP,
    csp_feature,
    csp_kernel,
)


class TestCspKernel:
    """Tests for the low-level csp_kernel function."""

    def test_output_shapes(self, mi_data_2class):
        X, y = mi_data_2class
        W, D, A = csp_kernel(X, y)
        n_channels = X.shape[1]
        assert W.shape == (n_channels, n_channels)
        assert D.shape == (n_channels,)
        assert A.shape == (n_channels, n_channels)

    def test_eigenvalues_between_0_and_1(self, mi_data_2class):
        X, y = mi_data_2class
        _, D, _ = csp_kernel(X, y)
        assert np.all(D >= -1e-10), "Eigenvalues should be non-negative"
        assert np.all(D <= 1.0 + 1e-10), "Eigenvalues should be <= 1"

    def test_raises_for_multiclass(self, mi_data_4class):
        """CSP kernel only supports 2-class problems."""
        X, y = mi_data_4class
        with pytest.raises(ValueError, match="2-class"):
            csp_kernel(X, y)

    def test_deterministic_output(self, mi_data_2class):
        """Same input should produce same output."""
        X, y = mi_data_2class
        W1, D1, A1 = csp_kernel(X, y)
        W2, D2, A2 = csp_kernel(X, y)
        assert_array_almost_equal(W1, W2)
        assert_array_almost_equal(D1, D2)
        assert_array_almost_equal(A1, A2)


class TestCspFeature:
    """Tests for the csp_feature function."""

    def test_output_shape(self, mi_data_2class):
        X, y = mi_data_2class
        W, _, _ = csp_kernel(X, y)
        n_components = 4
        features = csp_feature(W, X, n_components=n_components)
        assert features.shape == (X.shape[0], n_components)

    def test_log_transform_finite(self, mi_data_2class):
        X, y = mi_data_2class
        W, _, _ = csp_kernel(X, y)
        features = csp_feature(W, X, n_components=2)
        assert np.all(np.isfinite(features))

    def test_n_components_exceeds_channels_raises(self, mi_data_2class):
        X, y = mi_data_2class
        W, _, _ = csp_kernel(X, y)
        with pytest.raises(ValueError, match="n_components"):
            csp_feature(W, X, n_components=X.shape[1] + 1)


class TestCSP:
    """Tests for the CSP sklearn-compatible estimator."""

    def test_fit_returns_self(self, mi_data_2class):
        X, y = mi_data_2class
        csp = CSP(n_components=4)
        result = csp.fit(X, y)
        assert result is csp

    def test_transform_output_shape(self, mi_data_2class):
        X, y = mi_data_2class
        csp = CSP(n_components=4)
        csp.fit(X, y)
        features = csp.transform(X)
        assert features.shape == (X.shape[0], 4)

    def test_fit_transform_consistency(self, mi_data_2class):
        X, y = mi_data_2class
        csp = CSP(n_components=4)
        f1 = csp.fit(X, y).transform(X)
        csp2 = CSP(n_components=4)
        f2 = csp2.fit_transform(X, y)
        assert_array_almost_equal(f1, f2)

    def test_classes_attribute(self, mi_data_2class):
        X, y = mi_data_2class
        csp = CSP(n_components=4)
        csp.fit(X, y)
        assert hasattr(csp, "classes_")
        np.testing.assert_array_equal(np.sort(csp.classes_), np.array([0, 1]))

    def test_spatial_filter_shape(self, mi_data_2class):
        X, y = mi_data_2class
        csp = CSP(n_components=4)
        csp.fit(X, y)
        assert hasattr(csp, "W_")
        assert csp.W_.shape[0] == X.shape[1]

    @pytest.mark.parametrize("n_components", [1, 2, 4, 6])
    def test_various_n_components(self, mi_data_2class, n_components):
        X, y = mi_data_2class
        csp = CSP(n_components=n_components)
        csp.fit(X, y)
        features = csp.transform(X)
        assert features.shape[1] == n_components


class TestMultiCSP:
    """Tests for the MultiCSP multi-class estimator."""

    @pytest.mark.parametrize("strategy", ["ovr"])
    def test_multiclass_strategies(self, mi_data_4class, strategy):
        X, y = mi_data_4class
        mcsp = MultiCSP(n_components=2, multiclass=strategy)
        mcsp.fit(X, y)
        features = mcsp.transform(X)
        assert features.shape[0] == X.shape[0]
        assert features.ndim == 2

    def test_multiclass_strategies_ovo(self, mi_data_4class):
        """OVO produces two finite features for each of six class pairs."""
        X, y = mi_data_4class
        mcsp = MultiCSP(n_components=2, multiclass="ovo")
        mcsp.fit(X, y)
        features = mcsp.transform(X)
        assert features.shape == (80, 12)
        assert np.isfinite(features).all()

    def test_invalid_multiclass_raises(self, mi_data_4class):
        X, y = mi_data_4class
        mcsp = MultiCSP(n_components=2, multiclass="invalid")
        with pytest.raises(ValueError):
            mcsp.fit(X, y)

    def test_grosse_wentrup_method(self, mi_data_4class):
        X, y = mi_data_4class
        mcsp = MultiCSP(n_components=2, multiclass="grosse-wentrup")
        mcsp.fit(X, y)
        features = mcsp.transform(X)
        assert features.shape[0] == X.shape[0]
        assert np.all(np.isfinite(features))
