# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for DSP algorithms
#
# Tests for: metabci.brainda.algorithms.decomposition.dsp

import numpy as np

from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.decomposition.dsp import (
    DSP,
    FBDSP,
    xiang_dsp_kernel,
)


class TestXiangDspKernel:
    """Tests for the DSP kernel function."""

    def test_output_shapes(self, mi_data_2class):
        X, y = mi_data_2class
        W, D, M, A = xiang_dsp_kernel(X, y)
        n_channels = X.shape[1]
        assert W.shape == (n_channels, n_channels)
        assert D.shape == (n_channels,)
        assert M.shape == (n_channels, X.shape[2])
        assert A.shape == (n_channels, n_channels)

    def test_eigenvalues_sorted_descending(self, mi_data_2class):
        X, y = mi_data_2class
        _, D, _, _ = xiang_dsp_kernel(X, y)
        # Eigenvalues should be sorted in descending order
        assert np.all(D[:-1] >= D[1:] - 1e-10)


class TestDSP:
    """Tests for the DSP estimator."""

    def test_fit_returns_self(self, ssvep_data, cca_references):
        X, y = ssvep_data
        dsp = DSP(n_components=1)
        result = dsp.fit(X, y)
        assert result is dsp

    def test_transform_shape(self, ssvep_data):
        X, y = ssvep_data
        dsp = DSP(n_components=1)
        dsp.fit(X, y)
        rhos = dsp.transform(X)
        n_classes = len(np.unique(y))
        assert rhos.shape[0] == X.shape[0]
        assert rhos.shape[1] == n_classes

    def test_predict_shape(self, ssvep_data):
        X, y = ssvep_data
        dsp = DSP(n_components=1)
        dsp.fit(X, y)
        labels = dsp.predict(X)
        assert labels.shape == y.shape

    def test_classes_stored(self, ssvep_data):
        X, y = ssvep_data
        dsp = DSP(n_components=1)
        dsp.fit(X, y)
        assert hasattr(dsp, "classes_")
        np.testing.assert_array_equal(np.sort(dsp.classes_), np.sort(np.unique(y)))


class TestFBDSP:
    """Tests for filter bank DSP."""

    def test_fit_predict(self, ssvep_data, cca_references):
        X, y = ssvep_data
        wp = [(5, 90), (14, 90), (22, 90)]
        ws = [(3, 92), (12, 92), (20, 92)]
        fb = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        fw = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(len(fb))])
        fbdsp = FBDSP(filterbank=fb, n_components=1, filterweights=fw)
        fbdsp.fit(X, y)
        labels = fbdsp.predict(X)
        assert labels.shape == y.shape
