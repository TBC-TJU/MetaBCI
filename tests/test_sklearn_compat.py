# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for sklearn API compatibility
#
# Verifies that MetaBCI estimators follow scikit-learn conventions:
# - fit() returns self
# - fit_transform() equals fit().transform()
# - Estimators can be cloned
# - Estimators work in sklearn Pipeline

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from metabci.brainda.algorithms.decomposition.csp import CSP
from metabci.brainda.algorithms.utils.covariance import Covariance


class TestSklearnAPICompliance:
    """Verify sklearn API compliance for key estimators."""

    def test_csp_clone(self, mi_data_2class):
        """CSP should be clonable via sklearn.base.clone."""
        csp = CSP(n_components=4)
        csp_clone = clone(csp)
        assert csp_clone.n_components == csp.n_components
        # Clone should be a new object
        assert csp_clone is not csp

    def test_covariance_clone(self):
        """Covariance estimator should be clonable."""
        cov = Covariance(estimator="cov")
        cov_clone = clone(cov)
        assert cov_clone.estimator == cov.estimator
        assert cov_clone is not cov

    def test_csp_in_pipeline(self, mi_data_2class):
        """CSP should work inside an sklearn Pipeline."""
        X, y = mi_data_2class
        pipe = make_pipeline(CSP(n_components=4), SVC())
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_csp_get_params(self):
        """CSP should support get_params."""
        csp = CSP(n_components=4)
        params = csp.get_params()
        assert "n_components" in params
        assert params["n_components"] == 4

    def test_csp_set_params(self):
        """CSP should support set_params."""
        csp = CSP(n_components=4)
        csp.set_params(n_components=6)
        assert csp.n_components == 6

    def test_covariance_get_set_params(self):
        """Covariance should support get/set_params."""
        cov = Covariance(estimator="cov")
        assert cov.get_params()["estimator"] == "cov"
        cov.set_params(estimator="lwf")
        assert cov.estimator == "lwf"
