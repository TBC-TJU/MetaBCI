"""Verify feature pairing and archive loading without training a model."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from metabci.brainda.algorithms.deep_learning.transformer_data import (
    EEGFeatureDataset,
    load_feature_dataset,
)


def test_pairing_reorders_and_masks_missing_records():
    dataset = EEGFeatureDataset(
        np.zeros((3, 2, 100)),
        np.array([0, 1, 0]),
        pair_ids=["run/0", "run/1", "run/2"],
        auxiliary={"context": np.array([[20.0, 21.0], [10.0, 11.0]])},
        auxiliary_pair_ids={"context": ["run/2", "run/0"]},
    )
    torch.testing.assert_close(
        dataset.auxiliary["context"],
        torch.tensor([[[10.0, 11.0]], [[0.0, 0.0]], [[20.0, 21.0]]]),
    )
    assert dataset.auxiliary_mask["context"].tolist() == [[True], [False], [True]]
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    assert batch["X"].shape == (3, 2, 100)
    assert batch["auxiliary"]["context"].shape == (3, 1, 2)
    assert dataset.auxiliary_dims == {"context": 2}


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "unknown"]])
def test_invalid_feature_identifiers_fail(ids):
    with pytest.raises(ValueError):
        EEGFeatureDataset(
            np.zeros((2, 2, 100)),
            np.array([0, 1]),
            pair_ids=["a", "b"],
            auxiliary={"context": np.zeros((2, 3))},
            auxiliary_pair_ids={"context": ids},
        )


def test_auxiliary_requires_explicit_identifiers():
    with pytest.raises(ValueError, match="pair_ids"):
        EEGFeatureDataset(
            np.zeros((2, 2, 100)),
            np.array([0, 1]),
            auxiliary={"context": np.zeros((2, 3))},
        )


def test_masked_nan_is_sanitized_without_mutating_source():
    features = np.array([[[1.0, 2.0], [np.nan, np.nan]]])
    dataset = EEGFeatureDataset(
        np.zeros((1, 2, 100)),
        np.array([0]),
        pair_ids=["a"],
        auxiliary={"context": features},
        auxiliary_pair_ids={"context": ["a"]},
        auxiliary_mask={"context": np.array([[True, False]])},
    )
    assert torch.isfinite(dataset[0]["auxiliary"]["context"]).all()
    assert np.isnan(features[0, 1]).all()
    with pytest.raises(ValueError, match="finite"):
        EEGFeatureDataset(
            np.zeros((1, 2, 100)),
            np.array([0]),
            pair_ids=["a"],
            auxiliary={"context": features},
            auxiliary_pair_ids={"context": ["a"]},
        )


def test_archive_roundtrip_and_unknown_fields(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        X=np.zeros((2, 2, 100)),
        y=np.array([0, 1]),
        pair_ids=np.array(["a", "b"]),
        aux__context=np.ones((1, 2, 3)),
        ids__context=np.array(["b"]),
        mask__context=np.array([[True, False]]),
    )
    dataset = load_feature_dataset(path, dtype=torch.float64)
    assert dataset.X.dtype == torch.float64
    assert dataset.auxiliary_mask["context"].tolist() == [[False, False], [True, False]]
    np.savez(path, X=np.zeros((2, 2, 100)), y=np.array([0, 1]), typo=np.ones(2))
    with pytest.raises(ValueError, match="unrecognized"):
        load_feature_dataset(path)


@pytest.mark.parametrize("labels", [[-1, 0], [0.5, 1.0], [[0], [1]]])
def test_invalid_targets(labels):
    with pytest.raises(ValueError):
        EEGFeatureDataset(np.zeros((2, 2, 100)), np.asarray(labels))
