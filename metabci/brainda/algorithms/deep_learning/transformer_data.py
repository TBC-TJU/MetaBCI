"""Trial-aligned EEG and optional external feature datasets.

External encoders remain the caller's responsibility. These helpers align whole
trial feature records by identifier; they do not synchronize raw sensors or
construct semantic alignment between modalities.
"""

from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def _ids(values, expected, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != expected:
        raise ValueError(f"{name} must contain one identifier per record")
    identifiers = values.tolist()
    if not all(isinstance(value, str) and value for value in identifiers):
        raise ValueError(f"{name} must contain nonempty string identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique identifiers")
    return identifiers


class EEGFeatureDataset(Dataset):
    """EEG trials, integer targets and independently ordered feature records.

    Parameters
    ----------
    X : array-like, shape (n_trials, n_channels, n_samples)
        Fixed-length EEG windows, with a consistent channel order and units.
    y : array-like, shape (n_trials,)
        Zero-based class indices. Targets are never passed as model features.
    auxiliary : mapping, optional
        Named external features shaped (n_records, n_tokens, n_features), or
        (n_records, n_features). The caller supplies features, not raw images or
        text. Different records may be absent from a modality.
    auxiliary_mask : mapping, optional
        Boolean masks shaped (n_records, n_tokens), with True meaning valid.
    pair_ids : array-like, optional
        Unique string trial identifiers in EEG order. Required with auxiliary
        features. Use identifiers including dataset, subject, session and run.
    auxiliary_pair_ids : mapping, optional
        Unique string identifiers in each modality's own record order. Every
        supplied auxiliary modality requires its own identifiers.
    dtype : torch.dtype
        Floating dtype for EEG and features, default float32.

    Notes
    -----
    A missing feature record becomes zeros with an all-False mask. Extra records
    referring to unknown EEG trials are rejected. Identifiers must refer to the
    same trial/window; matching identifiers do not establish sensor time sync.
    """

    def __init__(
        self,
        X,
        y,
        auxiliary: Optional[Mapping] = None,
        auxiliary_mask: Optional[Mapping] = None,
        pair_ids=None,
        auxiliary_pair_ids: Optional[Mapping] = None,
        dtype=torch.float32,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.X = torch.as_tensor(X, dtype=dtype)
        labels = np.asarray(y)
        if self.X.device.type != "cpu":
            raise ValueError("dataset inputs must be on CPU; the trainer moves batches")
        if self.X.ndim != 3 or any(size == 0 for size in self.X.shape):
            raise ValueError("X must be a nonempty (trials, channels, samples) array")
        if not torch.isfinite(self.X).all():
            raise ValueError("X must contain finite values")
        if labels.ndim != 1 or len(labels) != len(self.X):
            raise ValueError("y must contain one class index per EEG trial")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("y must contain nonnegative integer class indices")
        self.y = torch.as_tensor(labels, dtype=torch.long)
        self.pair_ids = (
            _ids(pair_ids, len(self.X), "pair_ids") if pair_ids is not None else None
        )
        auxiliary = {} if auxiliary is None else dict(auxiliary)
        masks = {} if auxiliary_mask is None else dict(auxiliary_mask)
        aux_ids = {} if auxiliary_pair_ids is None else dict(auxiliary_pair_ids)
        if set(masks) - set(auxiliary) or set(aux_ids) - set(auxiliary):
            raise ValueError(
                "auxiliary masks and identifiers must refer to supplied features"
            )
        if auxiliary and self.pair_ids is None:
            raise ValueError("pair_ids are required to align auxiliary features")
        self.auxiliary = {}
        self.auxiliary_mask = {}
        self.auxiliary_dims = {}
        for name, value in auxiliary.items():
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("feature names must be nonempty strings without '__'")
            features = torch.as_tensor(value, dtype=dtype)
            if features.device.type != "cpu":
                raise ValueError("auxiliary dataset inputs must be on CPU")
            if features.ndim == 2:
                features = features.unsqueeze(1)
            if features.ndim != 3 or features.shape[1] == 0 or features.shape[2] == 0:
                raise ValueError(f"{name}: expected (records, tokens, features)")
            if name not in aux_ids:
                raise ValueError(f"{name}: auxiliary_pair_ids are required")
            identifiers = _ids(aux_ids[name], len(features), f"{name} identifiers")
            if set(identifiers) - set(self.pair_ids):
                raise ValueError(f"{name}: feature records refer to unknown EEG trials")
            if name in masks:
                mask = torch.as_tensor(masks[name])
                if (
                    mask.device.type != "cpu"
                    or mask.dtype != torch.bool
                    or mask.shape != features.shape[:2]
                ):
                    raise ValueError(f"{name}: mask must be boolean (records, tokens)")
            else:
                mask = torch.ones(features.shape[:2], dtype=torch.bool)
            if not torch.isfinite(features[mask]).all():
                raise ValueError(f"{name}: valid features must be finite")
            # Masked placeholders may contain NaN, but must never enter projection.
            features = torch.where(
                mask.unsqueeze(-1), features, torch.zeros_like(features)
            )
            aligned = torch.zeros(
                (len(self.X), features.shape[1], features.shape[2]), dtype=dtype
            )
            aligned_mask = torch.zeros(aligned.shape[:2], dtype=torch.bool)
            source_indices = {identifier: i for i, identifier in enumerate(identifiers)}
            for destination, identifier in enumerate(self.pair_ids):
                if identifier in source_indices:
                    source = source_indices[identifier]
                    aligned[destination] = features[source]
                    aligned_mask[destination] = mask[source]
            self.auxiliary[name] = aligned
            self.auxiliary_mask[name] = aligned_mask
            self.auxiliary_dims[name] = features.shape[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        batch = {"X": self.X[index], "y": self.y[index]}
        if self.auxiliary:
            batch["auxiliary"] = {
                name: values[index] for name, values in self.auxiliary.items()
            }
            batch["auxiliary_mask"] = {
                name: values[index] for name, values in self.auxiliary_mask.items()
            }
        return batch


def load_feature_dataset(path: Union[str, Path], dtype=torch.float32):
    """Read a non-pickled NPZ archive using explicit named array fields.

    Required keys are X and y. Optional keys are pair_ids, aux__NAME,
    ids__NAME and mask__NAME. A modality's ids are required when aux is present.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = {"X", "y"} - set(archive.files)
        if missing:
            raise ValueError(f"archive is missing required keys: {sorted(missing)}")
        auxiliary, masks, identifiers = {}, {}, {}
        for key in archive.files:
            if key.startswith("aux__"):
                auxiliary[key[5:]] = archive[key]
            elif key.startswith("mask__"):
                masks[key[6:]] = archive[key]
            elif key.startswith("ids__"):
                identifiers[key[5:]] = archive[key]
            elif key not in ("X", "y", "pair_ids"):
                raise ValueError(f"unrecognized archive field: {key}")
        return EEGFeatureDataset(
            archive["X"],
            archive["y"],
            auxiliary=auxiliary,
            auxiliary_mask=masks,
            pair_ids=archive["pair_ids"] if "pair_ids" in archive else None,
            auxiliary_pair_ids=identifiers,
            dtype=dtype,
        )
