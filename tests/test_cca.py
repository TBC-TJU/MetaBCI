# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for CCA-based SSVEP algorithms
#
# Tests for: metabci.brainda.algorithms.decomposition.cca

import numpy as np
import pytest

from metabci.brainda.algorithms.decomposition.base import (
    generate_cca_references,
    generate_filterbank,
)
from metabci.brainda.algorithms.decomposition.cca import (
    ECCA,
    FBECCA,
    FBSCCA,
    SCCA,
    FBItCCA,
    FBMsCCA,
    ItCCA,
    MsCCA,
)

# ============================================================================
# Helper to generate filter bank for tests
# ============================================================================


def _get_test_filterbank(srate=250):
    """Create a small filter bank for testing."""
    wp = [(5, 90), (14, 90), (22, 90)]
    ws = [(3, 92), (12, 92), (20, 92)]
    return generate_filterbank(wp, ws, srate=srate, order=4, rp=0.5)


def _get_filterweights(n_filters):
    """Create filter weights."""
    return np.array([(i + 1) ** (-1.25) + 0.25 for i in range(n_filters)])


# ============================================================================
# Tests for generate_cca_references
# ============================================================================


class TestGenerateCcaReferences:
    """Tests for the reference signal generator."""

    def test_output_shape_single_freq(self):
        Yf = generate_cca_references(10.0, srate=250, T=1.0, n_harmonics=3)
        assert Yf.shape == (1, 6, 250)

    def test_output_shape_multi_freq(self):
        freqs = [8.0, 10.0, 12.0]
        Yf = generate_cca_references(freqs, srate=250, T=1.0, n_harmonics=2)
        assert Yf.shape == (3, 4, 250)

    def test_sine_cosine_orthogonality(self):
        """Sin and cos of same frequency should be nearly orthogonal."""
        Yf = generate_cca_references(10.0, srate=250, T=1.0, n_harmonics=1)
        sin_ref = Yf[0, 0, :]
        cos_ref = Yf[0, 1, :]
        dot_product = np.abs(np.dot(sin_ref, cos_ref))
        # Should be near zero for integer number of periods
        assert dot_product < 5.0  # relaxed threshold for non-integer periods

    def test_values_bounded(self):
        Yf = generate_cca_references([8.0, 10.0], srate=250, T=1.0, n_harmonics=3)
        assert np.all(np.abs(Yf) <= 1.0 + 1e-10)


# ============================================================================
# Tests for generate_filterbank
# ============================================================================


class TestGenerateFilterbank:
    """Tests for the filter bank generator."""

    def test_output_length(self):
        wp = [(5, 90), (14, 90)]
        ws = [(3, 92), (12, 92)]
        fb = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        assert len(fb) == 2

    def test_filter_coefficients_not_empty(self):
        wp = [(5, 90)]
        ws = [(3, 92)]
        fb = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        assert fb[0].shape[0] > 0  # has filter sections
        assert fb[0].shape[1] == 6  # SOS format: 6 columns


# ============================================================================
# Tests for SCCA (Standard CCA)
# ============================================================================


class TestSCCA:
    """Tests for Standard CCA classifier."""

    def test_fit_returns_self(self, ssvep_data, cca_references):
        X, y = ssvep_data
        scca = SCCA(n_components=1)
        result = scca.fit(X, y, Yf=cca_references)
        assert result is scca

    def test_fit_requires_yf(self, ssvep_data):
        X, y = ssvep_data
        scca = SCCA(n_components=1)
        with pytest.raises(ValueError, match="Yf"):
            scca.fit(X, y, Yf=None)

    def test_transform_output_shape(self, ssvep_data, cca_references):
        X, y = ssvep_data
        scca = SCCA(n_components=1)
        scca.fit(X, y, Yf=cca_references)
        rhos = scca.transform(X)
        assert rhos.shape == (X.shape[0], cca_references.shape[0])

    def test_predict_output_shape(self, ssvep_data, cca_references):
        X, y = ssvep_data
        scca = SCCA(n_components=1)
        scca.fit(X, y, Yf=cca_references)
        labels = scca.predict(X)
        assert labels.shape == (X.shape[0],)

    def test_predict_labels_valid(self, ssvep_data, cca_references):
        X, y = ssvep_data
        scca = SCCA(n_components=1)
        scca.fit(X, y, Yf=cca_references)
        labels = scca.predict(X)
        # Labels should be valid class indices
        assert np.all(labels >= 0)
        assert np.all(labels < cca_references.shape[0])

    def test_above_chance_accuracy(self, ssvep_data, cca_references):
        """SCCA should perform above chance on clean SSVEP data."""
        X, y = ssvep_data
        n_classes = len(np.unique(y))
        scca = SCCA(n_components=1)
        scca.fit(X, y, Yf=cca_references)
        labels = scca.predict(X)
        accuracy = np.mean(labels == y)
        chance = 1.0 / n_classes
        assert accuracy > chance, (
            f"Accuracy {accuracy:.3f} should be above chance {chance:.3f}"
        )


# ============================================================================
# Tests for ItCCA (Individual Template CCA)
# ============================================================================


class TestItCCA:
    """Tests for Individual Template CCA."""

    def test_fit_returns_self(self, ssvep_data, cca_references):
        X, y = ssvep_data
        itcca = ItCCA(n_components=1, method="itcca2")
        result = itcca.fit(X, y, Yf=cca_references)
        assert result is itcca

    def test_templates_computed(self, ssvep_data, cca_references):
        X, y = ssvep_data
        itcca = ItCCA(n_components=1, method="itcca2")
        itcca.fit(X, y, Yf=cca_references)
        assert hasattr(itcca, "templates_")
        n_classes = len(np.unique(y))
        assert itcca.templates_.shape[0] == n_classes

    def test_predict_shape(self, ssvep_data, cca_references):
        X, y = ssvep_data
        itcca = ItCCA(n_components=1, method="itcca2")
        itcca.fit(X, y, Yf=cca_references)
        labels = itcca.predict(X)
        assert labels.shape == y.shape

    @pytest.mark.parametrize("method", ["itcca1", "itcca2"])
    def test_both_methods(self, ssvep_data, cca_references, method):
        X, y = ssvep_data
        itcca = ItCCA(n_components=1, method=method)
        if method == "itcca2":
            itcca.fit(X, y, Yf=cca_references)
        else:
            itcca.fit(X, y)
        labels = itcca.predict(X)
        assert labels.shape == y.shape

    def test_itcca2_requires_yf(self, ssvep_data):
        X, y = ssvep_data
        itcca = ItCCA(n_components=1, method="itcca2")
        with pytest.raises(ValueError, match="Yf"):
            itcca.fit(X, y, Yf=None)


# ============================================================================
# Tests for MsCCA (Multiset CCA)
# ============================================================================


class TestMsCCA:
    """Tests for Multiset CCA."""

    def test_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        mscca = MsCCA(n_components=1)
        mscca.fit(X, y, Yf=cca_references)
        labels = mscca.predict(X)
        assert labels.shape == y.shape

    def test_classes_stored(self, ssvep_data, cca_references):
        X, y = ssvep_data
        mscca = MsCCA(n_components=1)
        mscca.fit(X, y, Yf=cca_references)
        assert hasattr(mscca, "classes_")


# ============================================================================
# Tests for ECCA (Extended CCA)
# ============================================================================


class TestECCA:
    """Tests for Extended CCA."""

    def test_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        ecca = ECCA(n_components=1)
        ecca.fit(X, y, Yf=cca_references)
        labels = ecca.predict(X)
        assert labels.shape == y.shape

    def test_above_chance(self, ssvep_data, cca_references):
        X, y = ssvep_data
        n_classes = len(np.unique(y))
        ecca = ECCA(n_components=1)
        ecca.fit(X, y, Yf=cca_references)
        labels = ecca.predict(X)
        accuracy = np.mean(labels == y)
        chance = 1.0 / n_classes
        assert accuracy > chance, (
            f"ECCA accuracy {accuracy:.3f} should be above chance {chance:.3f}"
        )


# ============================================================================
# Tests for Filter Bank variants
# ============================================================================


class TestFilterBankCCA:
    """Tests for filter bank CCA estimators."""

    def test_fbscca_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        fb = _get_test_filterbank()
        fw = _get_filterweights(len(fb))
        fbscca = FBSCCA(filterbank=fb, n_components=1, filterweights=fw)
        fbscca.fit(X, y, Yf=cca_references)
        labels = fbscca.predict(X)
        assert labels.shape == y.shape

    def test_fbitcca_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        fb = _get_test_filterbank()
        fw = _get_filterweights(len(fb))
        fbitcca = FBItCCA(
            filterbank=fb, n_components=1, filterweights=fw, method="itcca2"
        )
        fbitcca.fit(X, y, Yf=cca_references)
        labels = fbitcca.predict(X)
        assert labels.shape == y.shape

    def test_fbmscca_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        fb = _get_test_filterbank()
        fw = _get_filterweights(len(fb))
        fbmscca = FBMsCCA(filterbank=fb, n_components=1, filterweights=fw)
        fbmscca.fit(X, y, Yf=cca_references)
        labels = fbmscca.predict(X)
        assert labels.shape == y.shape

    def test_fbecca_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        fb = _get_test_filterbank()
        fw = _get_filterweights(len(fb))
        fbecca = FBECCA(filterbank=fb, n_components=1, filterweights=fw)
        fbecca.fit(X, y, Yf=cca_references)
        labels = fbecca.predict(X)
        assert labels.shape == y.shape

    def test_fbscca_without_weights(self, ssvep_data, cca_references):
        """Test FBSCCA without explicit filter weights (should use mean)."""
        X, y = ssvep_data
        fb = _get_test_filterbank()
        fbscca = FBSCCA(filterbank=fb, n_components=1, filterweights=None)
        fbscca.fit(X, y, Yf=cca_references)
        labels = fbscca.predict(X)
        assert labels.shape == y.shape
