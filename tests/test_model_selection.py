# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for model selection utilities
#
# Tests for: metabci.brainda.algorithms.utils.model_selection

import numpy as np
import pandas as pd
import pytest

from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedStratifiedKFold,
    EnhancedStratifiedShuffleSplit,
    generate_kfold_indices,
    match_kfold_indices,
    set_random_seeds,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_meta():
    """Create a sample meta DataFrame for testing."""
    n_trials_per_event = 10
    events = ["left_hand", "right_hand"]
    data = {
        "subject": [1] * (n_trials_per_event * len(events)),
        "event": events * n_trials_per_event,
    }
    # Sort by event to group them
    meta = pd.DataFrame(data)
    meta = meta.sort_values("event").reset_index(drop=True)
    return meta


@pytest.fixture
def multi_subject_meta():
    """Create a multi-subject meta DataFrame."""
    rows = []
    for subj in [1, 2]:
        for event in ["left_hand", "right_hand"]:
            for _ in range(10):
                rows.append({"subject": subj, "event": event})
    return pd.DataFrame(rows)


# ============================================================================
# Tests for set_random_seeds
# ============================================================================


class TestSetRandomSeeds:
    """Tests for random seed setting."""

    def test_numpy_reproducibility(self):
        set_random_seeds(42)
        a = np.random.rand(5)
        set_random_seeds(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_random_seeds(42)
        a = np.random.rand(5)
        set_random_seeds(123)
        b = np.random.rand(5)
        assert not np.allclose(a, b)


# ============================================================================
# Tests for EnhancedStratifiedKFold
# ============================================================================


class TestEnhancedStratifiedKFold:
    """Tests for Enhanced Stratified KFold."""

    def test_with_validation(self):
        X = np.ones((40, 2))
        y = np.array([0] * 20 + [1] * 20)
        splitter = EnhancedStratifiedKFold(
            n_splits=5, shuffle=True, return_validate=True, random_state=42
        )
        splits = list(splitter.split(X, y))
        assert len(splits) == 5
        for train, val, test in splits:
            # All indices should be unique
            all_idx = np.concatenate([train, val, test])
            assert len(all_idx) == len(np.unique(all_idx))
            # Union should cover all samples
            assert len(all_idx) == len(y)

    def test_without_validation(self):
        X = np.ones((40, 2))
        y = np.array([0] * 20 + [1] * 20)
        splitter = EnhancedStratifiedKFold(
            n_splits=5, shuffle=True, return_validate=False, random_state=42
        )
        splits = list(splitter.split(X, y))
        assert len(splits) == 5
        for train, test in splits:
            all_idx = np.concatenate([train, test])
            assert len(all_idx) == len(y)

    def test_stratification(self):
        """Each fold should maintain class balance."""
        X = np.ones((100, 2))
        y = np.array([0] * 50 + [1] * 50)
        splitter = EnhancedStratifiedKFold(
            n_splits=5, shuffle=True, return_validate=False, random_state=42
        )
        for train, test in splitter.split(X, y):
            # Test set should have roughly equal class proportions
            test_balance = np.mean(y[test])
            assert 0.3 < test_balance < 0.7


# ============================================================================
# Tests for generate_kfold_indices / match_kfold_indices
# ============================================================================


class TestKfoldIndices:
    """Tests for kfold index generation and matching."""

    def test_generate_kfold_structure(self, sample_meta):
        kfold = 3
        indices = generate_kfold_indices(sample_meta, kfold=kfold, random_state=42)
        subjects = sample_meta["subject"].unique()
        events = sample_meta["event"].unique()
        for sub in subjects:
            assert sub in indices
            for event in events:
                assert event in indices[sub]
                assert len(indices[sub][event]) == kfold

    def test_match_kfold_indices(self, sample_meta):
        kfold = 3
        indices = generate_kfold_indices(sample_meta, kfold=kfold, random_state=42)
        for k in range(kfold):
            train_ix, val_ix, test_ix = match_kfold_indices(k, sample_meta, indices)
            # All indices should be valid
            assert np.all(train_ix < len(sample_meta))
            assert np.all(val_ix < len(sample_meta))
            assert np.all(test_ix < len(sample_meta))
            # No overlap between train, val, test
            all_ix = np.concatenate([train_ix, val_ix, test_ix])
            assert len(all_ix) == len(np.unique(all_ix))

    def test_multi_subject_kfold(self, multi_subject_meta):
        kfold = 3
        indices = generate_kfold_indices(
            multi_subject_meta, kfold=kfold, random_state=42
        )
        for k in range(kfold):
            train_ix, val_ix, test_ix = match_kfold_indices(
                k, multi_subject_meta, indices
            )
            total = len(train_ix) + len(val_ix) + len(test_ix)
            assert total == len(multi_subject_meta)


# ============================================================================
# Tests for EnhancedStratifiedShuffleSplit
# ============================================================================


class TestEnhancedStratifiedShuffleSplit:
    """Tests for Enhanced Stratified Shuffle Split."""

    def test_with_validation(self):
        X = np.ones((100, 2))
        y = np.array([0] * 50 + [1] * 50)
        splitter = EnhancedStratifiedShuffleSplit(
            test_size=0.2,
            train_size=0.6,
            validate_size=0.2,
            n_splits=3,
            return_validate=True,
            random_state=42,
        )
        splits = list(splitter.split(X, y))
        assert len(splits) == 3
        for train, val, test in splits:
            all_idx = np.concatenate([train, val, test])
            # All should be unique
            assert len(all_idx) == len(np.unique(all_idx))

    def test_without_validation(self):
        X = np.ones((100, 2))
        y = np.array([0] * 50 + [1] * 50)
        splitter = EnhancedStratifiedShuffleSplit(
            test_size=0.2,
            train_size=0.8,
            n_splits=3,
            return_validate=False,
            random_state=42,
        )
        splits = list(splitter.split(X, y))
        assert len(splits) == 3
        for train, test in splits:
            assert len(train) + len(test) == len(y)
