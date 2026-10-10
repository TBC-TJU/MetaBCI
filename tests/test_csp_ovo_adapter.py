"""Behavioral regressions for MultiCSP's OVO input adapter.

All references use the package's original three-dimensional CSP directly.
No EEG data downloads or external source snapshots are required.
"""

import pickle
from itertools import combinations

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import butter, sosfiltfilt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from metabci.brainda.algorithms.decomposition import csp as csp_module
from metabci.brainda.algorithms.decomposition.csp import (
    CSP,
    FBMultiCSP,
    MultiCSP,
    _CSPInputAdapter,
)


@pytest.fixture(autouse=True)
def bounded_threading():
    """Keep the small reference fits from oversubscribing native threads."""
    with threadpool_limits(limits=1), joblib.parallel_backend("threading"):
        yield


@pytest.fixture
def eeg():
    rng = np.random.default_rng(42)
    y = np.repeat(np.array(["A", "B", "C", "D"]), 6)
    X = rng.normal(size=(len(y), 4, 64))
    for class_index, label in enumerate(np.unique(y)):
        X[y == label, class_index, :] *= 3
    return X, y


def pairwise_reference(X, y, X_query, n_components, max_components=None):
    """Fit original CSPs on the same class pairs and binary target coding."""
    features = []
    components = []
    for left, right in combinations(np.unique(y), 2):
        selected = (y == left) | (y == right)
        binary_y = (y[selected] == right).astype(int)
        csp = CSP(n_components=n_components, max_components=max_components)
        csp.fit(X[selected], binary_y)
        features.append(csp.transform(X_query))
        components.append(getattr(csp, "best_n_components_", n_components))
    return np.concatenate(features, axis=-1), components


def test_coding_commutes_with_trial_selection(eeg):
    X, _ = eeg
    selected = np.array([0, 3, 9, 17])
    adapter = _CSPInputAdapter(n_channels=X.shape[1], n_components=2)
    flat = X.reshape(X.shape[0], -1, order="C")
    assert_array_equal(adapter._decode(flat[selected]), X[selected])


def test_fixed_components_match_original_pairwise_csp(eeg):
    X, y = eeg
    model = MultiCSP(n_components=2, multiclass="ovo")
    result = model.fit_transform(X, y)
    expected, _ = pairwise_reference(X, y, X, n_components=2)
    assert result.shape == (len(y), 12)
    assert len(model.estimator_.estimators_) == 6
    assert np.isfinite(result).all()
    assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    assert all(
        list(est.named_steps) == ["csp", "svc"] for est in model.estimator_.estimators_
    )


def test_auto_components_match_per_pair_grid_search(eeg, monkeypatch):
    # Fix each search's folds, rather than rely on consumption order of the
    # global RNG by parallel fits. This affects the test harness only.
    def deterministic_folds(*args, **kwargs):
        if kwargs.get("shuffle"):
            kwargs["random_state"] = 42
        return StratifiedKFold(*args, **kwargs)

    monkeypatch.setattr(csp_module, "StratifiedKFold", deterministic_folds)
    X, y = eeg
    model = MultiCSP(n_components=None, max_components=2, multiclass="ovo")
    result = model.fit_transform(X, y)
    expected, components = pairwise_reference(X, y, X, None, max_components=2)
    assert result.shape[1] == sum(components)
    assert [est[0].best_n_components_ for est in model.estimator_.estimators_] == (
        components
    )
    assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_transform_allows_different_trial_count_and_time_length(eeg):
    X, y = eeg
    model = MultiCSP(n_components=2, multiclass="ovo").fit(X, y)
    query = np.random.default_rng(123).normal(size=(5, X.shape[1], 97))
    result = model.transform(query)
    expected, _ = pairwise_reference(X, y, query, n_components=2)
    assert result.shape == (5, 12)
    assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    step = model.estimator_.estimators_[0][0]
    assert_allclose(step.transform(query.reshape(5, -1)), step.transform(query))


def test_clone_and_pickle_preserve_model_behavior(eeg):
    X, y = eeg
    model = MultiCSP(n_components=2, multiclass="ovo")
    fresh = clone(model)
    assert fresh.get_params() == model.get_params()
    assert not hasattr(fresh, "estimator_")
    fresh.fit(X, y)
    restored = pickle.loads(pickle.dumps(fresh))
    assert_allclose(restored.transform(X), fresh.transform(X))


def test_ovo_fit_works_with_process_parallelism(eeg, monkeypatch):
    X, y = eeg
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        model = MultiCSP(n_components=2, multiclass="ovo").fit(X, y)
    expected, _ = pairwise_reference(X, y, X, n_components=2)
    assert_allclose(model.transform(X), expected, rtol=1e-10, atol=1e-10)


def test_outer_pipeline_cross_validation(eeg):
    X, y = eeg
    pipeline = Pipeline(
        [
            ("multicsp", MultiCSP(n_components=2, multiclass="ovo")),
            ("svc", SVC()),
        ]
    )
    scores = cross_val_score(pipeline, X, y, cv=3, error_score="raise", n_jobs=1)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_ovr_branch_matches_original_csp_one_vs_rest_reference(eeg):
    X, y = eeg
    expected = []
    for label in np.unique(y):
        binary_y = (y == label).astype(int)
        csp = CSP(n_components=2).fit(X, binary_y)
        expected.append(csp.transform(X))
    result = MultiCSP(n_components=2, multiclass="ovr").fit_transform(X, y)
    assert result.shape == (len(y), 8)
    assert_allclose(result, np.concatenate(expected, axis=-1), rtol=1e-10, atol=1e-10)


def test_invalid_input_structure_is_rejected(eeg):
    X, y = eeg
    model = MultiCSP(n_components=2, multiclass="ovo")
    flat = X.reshape(len(y), -1)
    with pytest.raises(ValueError, match="3D"):
        model.fit(flat, y)
    model.fit(X, y)
    with pytest.raises(ValueError, match="3D"):
        model.transform(flat)
    with pytest.raises(ValueError, match="channel count"):
        model.transform(X[:, :3, :])
    with pytest.raises(ValueError, match="divisible"):
        model.estimator_.estimators_[0][0].transform(np.zeros((5, 17)))


def test_encoding_handles_noncontiguous_input(eeg):
    X, y = eeg
    X = X[:, :, ::2]
    assert not X.flags.c_contiguous
    result = MultiCSP(n_components=2, multiclass="ovo").fit_transform(X, y)
    expected, _ = pairwise_reference(X, y, X, n_components=2)
    assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_filterbank_ovo_matches_filtered_pairwise_csp_reference(eeg):
    X, y = eeg
    filterbank = [
        butter(2, [8, 30], btype="bandpass", fs=128, output="sos"),
        butter(2, [30, 50], btype="bandpass", fs=128, output="sos"),
    ]
    model = FBMultiCSP(
        n_components=2,
        multiclass="ovo",
        n_mutualinfo_components=4,
        filterbank=filterbank,
    )
    assert model.fit(X, y) is model
    query = np.random.default_rng(123).normal(size=(5, X.shape[1], 97))
    expected = []
    for sos in filterbank:
        filtered_train = sosfiltfilt(sos, X, axis=-1)
        filtered_query = sosfiltfilt(sos, query, axis=-1)
        features, _ = pairwise_reference(
            filtered_train, y, filtered_query, n_components=2
        )
        expected.append(features)
    concatenated = np.concatenate(expected, axis=-1)
    result = model.transform(query)
    assert len(model.estimators_) == 2
    assert all(len(est.estimator_.estimators_) == 6 for est in model.estimators_)
    assert concatenated.shape == (5, 24)
    assert result.shape == (5, 4)
    assert np.isfinite(result).all()
    assert_allclose(
        result, model.selector_.transform(concatenated), rtol=1e-10, atol=1e-10
    )
