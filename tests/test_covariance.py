# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for covariance utilities
#
# Tests for: metabci.brainda.algorithms.utils.covariance

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from metabci.brainda.algorithms.utils.covariance import (
    Covariance,
    covariances,
    expm,
    invsqrtm,
    isPD,
    logm,
    nearestPD,
    powm,
    sqrtm,
)


class TestIsPD:
    """Tests for isPD (positive-definite check)."""

    def test_identity_is_pd(self):
        assert isPD(np.eye(5)) is True

    def test_zero_matrix_is_not_pd(self):
        assert isPD(np.zeros((3, 3))) is False

    def test_negative_eigenvalue_not_pd(self):
        A = np.diag([1, 1, -1])
        assert isPD(A) is False

    def test_random_spd_is_pd(self, rng):
        A = rng.randn(5, 5)
        C = A @ A.T + np.eye(5)
        assert isPD(C) is True


class TestNearestPD:
    """Tests for nearestPD (nearest positive-definite matrix)."""

    def test_pd_matrix_unchanged(self):
        C = np.eye(4) * 2.0
        C_pd = nearestPD(C)
        assert isPD(C_pd)
        assert_array_almost_equal(C, C_pd)

    def test_non_pd_becomes_pd(self):
        A = np.array([[1, 2], [2, 1]])  # eigenvalues: 3, -1
        C_pd = nearestPD(A)
        assert isPD(C_pd)

    def test_symmetry_preserved(self, rng):
        A = rng.randn(5, 5)
        C_pd = nearestPD(A)
        assert_array_almost_equal(C_pd, C_pd.T)


class TestCovariances:
    """Tests for covariance estimation functions."""

    def test_empirical_covariance_shape(self, mi_data_2class):
        X, _ = mi_data_2class
        C = covariances(X, estimator="cov")
        assert C.shape == (X.shape[0], X.shape[1], X.shape[1])

    def test_empirical_covariance_symmetry(self, mi_data_2class):
        X, _ = mi_data_2class
        C = covariances(X, estimator="cov")
        for i in range(C.shape[0]):
            assert_array_almost_equal(C[i], C[i].T)

    def test_empirical_covariance_positive_definite(self, mi_data_2class):
        X, _ = mi_data_2class
        C = covariances(X, estimator="cov")
        for i in range(C.shape[0]):
            eigenvalues = np.linalg.eigvalsh(C[i])
            assert np.all(eigenvalues >= -1e-10), (
                f"Trial {i}: negative eigenvalue found"
            )

    @pytest.mark.parametrize("estimator", ["cov", "lwf", "oas"])
    def test_estimator_types(self, mi_data_2class, estimator):
        X, _ = mi_data_2class
        C = covariances(X, estimator=estimator)
        assert C.shape == (X.shape[0], X.shape[1], X.shape[1])

    def test_invalid_estimator_raises(self, mi_data_2class):
        X, _ = mi_data_2class
        with pytest.raises(ValueError):
            covariances(X, estimator="invalid_method")

    def test_covariance_transformer(self, mi_data_2class):
        """Test that Covariance class follows sklearn API."""
        X, _ = mi_data_2class
        cov = Covariance(estimator="cov")
        # fit should return self
        result = cov.fit(X)
        assert result is cov
        # transform should return covariance matrices
        C = cov.transform(X)
        assert C.shape == (X.shape[0], X.shape[1], X.shape[1])

    def test_covariance_fit_transform(self, mi_data_2class):
        """Test fit_transform consistency."""
        X, _ = mi_data_2class
        cov = Covariance(estimator="cov")
        C1 = cov.fit(X).transform(X)
        C2 = cov.fit_transform(X)
        assert_array_almost_equal(C1, C2)


class TestMatrixOperations:
    """Tests for matrix square root, log, exp, inverse square root, power."""

    def test_sqrtm_identity(self):
        identity = np.eye(4)
        result = sqrtm(identity)
        assert_array_almost_equal(result, identity)

    def test_sqrtm_squared_equals_original(self, spd_matrices):
        C = spd_matrices[0]
        C_sqrt = sqrtm(C)
        reconstructed = C_sqrt @ C_sqrt
        assert_array_almost_equal(reconstructed, C, decimal=5)

    def test_invsqrtm_times_sqrtm_is_identity(self, spd_matrices):
        C = spd_matrices[0]
        C_sqrt = sqrtm(C)
        C_isqrt = invsqrtm(C)
        product = C_sqrt @ C_isqrt
        assert_array_almost_equal(product, np.eye(C.shape[0]), decimal=5)

    def test_logm_expm_roundtrip(self, spd_matrices):
        C = spd_matrices[0]
        C_log = logm(C)
        C_reconstructed = expm(C_log)
        assert_array_almost_equal(C_reconstructed, C, decimal=5)

    def test_powm_identity(self, spd_matrices):
        C = spd_matrices[0]
        C_pow1 = powm(C, 1.0)
        assert_array_almost_equal(C_pow1, C, decimal=5)

    def test_powm_half_equals_sqrtm(self, spd_matrices):
        C = spd_matrices[0]
        C_pow_half = powm(C, 0.5)
        C_sqrt = sqrtm(C)
        assert_array_almost_equal(C_pow_half, C_sqrt, decimal=5)

    def test_batch_sqrtm(self, spd_matrices):
        """Test sqrtm on a batch of matrices."""
        result = sqrtm(spd_matrices)
        assert result.shape == spd_matrices.shape
        for i in range(spd_matrices.shape[0]):
            reconstructed = result[i] @ result[i]
            assert_array_almost_equal(reconstructed, spd_matrices[i], decimal=4)
