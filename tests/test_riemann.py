# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for Riemannian geometry
#
# Tests for: metabci.brainda.algorithms.manifold.riemann

import numpy as np
from numpy.testing import assert_array_almost_equal

from metabci.brainda.algorithms.manifold.riemann import (
    FGDA,
    MDRM,
    distance_riemann,
    expmap,
    geodesic,
    logmap,
    mean_riemann,
    tangent_space,
    unvectorize,
    vectorize,
)
from metabci.brainda.algorithms.utils.covariance import covariances


class TestLogExpMap:
    """Tests for logarithm and exponential maps."""

    def test_logmap_expmap_roundtrip(self, spd_matrices):
        """Applying logmap then expmap should return original."""
        P = spd_matrices[0]
        Pi = spd_matrices[1]
        Si = logmap(Pi, P)
        Pi_reconstructed = expmap(Si, P)
        assert_array_almost_equal(Pi_reconstructed.reshape(Pi.shape), Pi, decimal=4)

    def test_logmap_at_identity(self, spd_matrices):
        """Log map at identity should give log of input."""
        identity = np.eye(spd_matrices.shape[1])
        Pi = spd_matrices[0]
        Si = logmap(Pi, identity)
        assert Si.shape[-2:] == Pi.shape
        assert np.all(np.isfinite(Si))


class TestGeodesic:
    """Tests for geodesic curves.

    Note: geodesic() internally reshapes 2D input (n, n) to (1, n, n),
    so the output has shape (1, n, n). We squeeze for comparison.
    """

    def test_geodesic_endpoints(self, spd_matrices):
        """Geodesic at t=0 returns P1, at t=1 returns P2."""
        P1 = spd_matrices[0]
        P2 = spd_matrices[1]
        G0 = geodesic(P1, P2, 0)
        G1 = geodesic(P1, P2, 1)
        # Squeeze from (1, n, n) to (n, n) for comparison
        assert_array_almost_equal(G0.squeeze(), P1, decimal=4)
        assert_array_almost_equal(G1.squeeze(), P2, decimal=4)

    def test_geodesic_midpoint_is_spd(self, spd_matrices):
        """Midpoint of geodesic should be SPD."""
        P1 = spd_matrices[0]
        P2 = spd_matrices[1]
        G_mid = geodesic(P1, P2, 0.5)
        G_mid_2d = G_mid.squeeze()
        eigenvalues = np.linalg.eigvalsh(G_mid_2d)
        assert np.all(eigenvalues > -1e-10)


class TestDistance:
    """Tests for Riemannian distance."""

    def test_distance_self_is_zero(self, spd_matrices):
        C = spd_matrices[0:1]
        d = distance_riemann(C, C[0])
        assert_array_almost_equal(d, [0.0], decimal=5)

    def test_distance_positive(self, spd_matrices):
        d = distance_riemann(spd_matrices[:5], spd_matrices[0])
        assert np.all(d >= -1e-10)

    def test_distance_symmetry(self, spd_matrices):
        """d(A, B) should equal d(B, A)."""
        A = spd_matrices[0:1]
        B = spd_matrices[1:2]
        d_ab = distance_riemann(A, B[0])
        d_ba = distance_riemann(B, A[0])
        assert_array_almost_equal(d_ab, d_ba, decimal=5)


class TestMeanRiemann:
    """Tests for Riemannian mean."""

    def test_mean_shape(self, spd_matrices):
        M = mean_riemann(spd_matrices[:10])
        assert M.shape == (spd_matrices.shape[1], spd_matrices.shape[2])

    def test_mean_is_spd(self, spd_matrices):
        M = mean_riemann(spd_matrices[:10])
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > -1e-10)

    def test_mean_of_identical_is_same(self, spd_matrices):
        """Mean of identical matrices should be the same matrix."""
        C = spd_matrices[0]
        C_batch = np.stack([C] * 5)
        M = mean_riemann(C_batch)
        assert_array_almost_equal(M, C, decimal=4)


class TestTangentSpace:
    """Tests for tangent space projection.

    Note: vectorize() reshapes 2D input (n, n) to (1, n, n) internally.
    For roundtrip tests, use batch input (m, n, n) to avoid shape mismatch.
    """

    def test_vectorize_unvectorize_roundtrip(self, spd_matrices):
        """Vectorize then unvectorize should return original."""
        # Use batch input to match expected shapes
        S = spd_matrices[:3]
        v = vectorize(S)
        S_back = unvectorize(v)
        assert_array_almost_equal(S_back, S, decimal=5)

    def test_tangent_space_shape(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        ts = tangent_space(C, np.eye(X.shape[1]))
        assert ts.shape[0] == X.shape[0]
        n_ch = X.shape[1]
        expected_features = n_ch * (n_ch + 1) // 2
        assert ts.shape[1] == expected_features


class TestMDRM:
    """Tests for Minimum Distance to Riemannian Mean classifier."""

    def test_fit_returns_self(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        mdrm = MDRM()
        result = mdrm.fit(C, y)
        assert result is mdrm

    def test_predict_shape(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        mdrm = MDRM()
        mdrm.fit(C, y)
        labels = mdrm.predict(C)
        assert labels.shape == y.shape

    def test_predict_valid_labels(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        mdrm = MDRM()
        mdrm.fit(C, y)
        labels = mdrm.predict(C)
        assert set(labels).issubset(set(y))

    def test_above_chance_accuracy(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        mdrm = MDRM()
        mdrm.fit(C, y)
        labels = mdrm.predict(C)
        accuracy = np.mean(labels == y)
        assert accuracy > 0.5, f"Accuracy {accuracy:.3f} should be above chance 0.5"


class TestFGDA:
    """Tests for Fisher Geodesic Discriminant Analysis."""

    def test_fit_transform(self, mi_data_2class):
        X, y = mi_data_2class
        C = covariances(X, estimator="cov")
        fgda = FGDA()
        fgda.fit(C, y)
        features = fgda.transform(C)
        assert features.shape[0] == X.shape[0]
        assert np.all(np.isfinite(features))
