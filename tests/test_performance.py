# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for BCI performance metrics
#
# Tests for: metabci.brainda.utils.performance

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from metabci.brainda.utils.performance import (
    Performance,
    _accuracy,
    _balance_accuracy,
    _confusion_matrix,
    _fnr_count,
    _fpr_count,
    _indicators,
    _practical_itr,
    _theoretical_itr,
    _tnr_count,
    _tpr_count,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    return y_true, y_pred


@pytest.fixture
def binary_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    return y_true, y_pred


# ============================================================================
# Tests for accuracy metrics
# ============================================================================


class TestAccuracy:
    """Tests for accuracy function."""

    def test_perfect_accuracy(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert_almost_equal(_accuracy(y_true, y_pred), 1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert_almost_equal(_accuracy(y_true, y_pred), 0.0)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            _accuracy(np.array([0, 1]), np.array([0]))


class TestBalanceAccuracy:
    """Tests for balanced accuracy function."""

    def test_perfect_balanced_accuracy(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert_almost_equal(_balance_accuracy(y_true, y_pred), 1.0)

    def test_imbalanced_data(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])  # all predicted as 0
        # Balanced accuracy: (4/4 + 0/1) / 2 = 0.5
        assert_almost_equal(_balance_accuracy(y_true, y_pred), 0.5)


# ============================================================================
# Tests for ITR metrics
# ============================================================================


class TestITR:
    """Tests for Information Transfer Rate."""

    def test_theoretical_itr_perfect(self):
        y_true = np.array([0, 1, 2, 3] * 10)
        y_pred = y_true.copy()
        itr = _theoretical_itr(y_true, y_pred, Tw=1.0)
        # For 4 classes with P≈1: ITR ≈ 60 * log2(4) = 120 bits/min
        assert itr > 100  # should be close to 120

    def test_practical_itr_less_than_theoretical(self):
        y_true = np.array([0, 1, 2] * 20)
        y_pred = y_true.copy()
        titr = _theoretical_itr(y_true, y_pred, Tw=1.0)
        pitr = _practical_itr(y_true, y_pred, Tw=1.0, Ts=0.5)
        assert pitr < titr

    def test_itr_positive(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        itr = _theoretical_itr(y_true, y_pred, Tw=1.0)
        # ITR should be non-negative for above-chance accuracy
        assert itr >= 0

    def test_itr_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            _theoretical_itr(np.array([0, 1]), np.array([0]), Tw=1.0)


# ============================================================================
# Tests for confusion matrix and derived metrics
# ============================================================================


class TestConfusionMatrix:
    """Tests for confusion matrix and indicators."""

    def test_confusion_matrix_shape(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        matrix = _confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))
        assert matrix.shape == (n_classes, n_classes)

    def test_confusion_matrix_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        matrix = _confusion_matrix(y_true, y_pred)
        # Should be diagonal
        off_diagonal = matrix.sum() - np.diag(matrix).sum()
        assert off_diagonal == 0

    def test_indicators_sum(self, binary_predictions):
        y_true, y_pred = binary_predictions
        TP, FP, FN, TN = _indicators(y_true, y_pred)
        total = (TP + FP + FN + TN).sum()
        # Sum should equal n_classes * n_samples
        n_classes = len(np.unique(y_true))
        assert total == n_classes * len(y_true)

    def test_tpr_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        tpr = _tpr_count(y_true, y_pred)
        assert_almost_equal(tpr, 1.0)

    def test_fnr_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        fnr = _fnr_count(y_true, y_pred)
        assert_almost_equal(fnr, 0.0)

    def test_tpr_plus_fnr_equals_one(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        tpr = _tpr_count(y_true, y_pred)
        fnr = _fnr_count(y_true, y_pred)
        assert_almost_equal(tpr + fnr, 1.0)

    def test_fpr_plus_tnr_equals_one(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        fpr = _fpr_count(y_true, y_pred)
        tnr = _tnr_count(y_true, y_pred)
        assert_almost_equal(fpr + tnr, 1.0)


# ============================================================================
# Tests for Performance class
# ============================================================================


class TestPerformanceClass:
    """Tests for the Performance evaluator class."""

    def test_basic_evaluation(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        perf = Performance(estimators_list=["Acc", "bAcc", "TPR"], Tw=1.0)
        results = perf.evaluate(y_true, y_pred)
        assert "Acc" in results
        assert "bAcc" in results
        assert "TPR" in results

    def test_itr_evaluation(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        perf = Performance(estimators_list=["tITR", "pITR"], Tw=1.0, Ts=0.5)
        results = perf.evaluate(y_true, y_pred)
        assert "tITR" in results
        assert "pITR" in results
        assert results["tITR"] >= results["pITR"]

    def test_missing_tw_raises(self):
        with pytest.raises(ValueError, match="Tw"):
            Performance(estimators_list=["tITR"])

    def test_missing_ts_raises(self):
        with pytest.raises(ValueError):
            Performance(estimators_list=["pITR"], Tw=1.0)

    def test_auc_requires_yscore(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        perf = Performance(estimators_list=["AUC"], Tw=1.0)
        with pytest.raises(ValueError, match="y_score"):
            perf.evaluate(y_true, y_pred, y_score=None)
