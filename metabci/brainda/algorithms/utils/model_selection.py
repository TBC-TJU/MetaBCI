# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/08
# License: MIT License
import random
import warnings
from typing import Optional, Union, Dict
from collections import defaultdict

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    LeaveOneGroupOut,
)
import torch


def set_random_seeds(seed: int):
    """Set seeds for python random module numpy.random and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = False
        # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


class EnhancedStratifiedKFold(StratifiedKFold):
    """Enhanced Stratified KFold cross-validator.

    if return_validate is True, split return (train, validate, test) indexs,
    else (train, test) as the sklearn StratifiedKFold.fit

    the validate size should be the same as the test size.

    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        return_validate: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ):

        self.return_validate = return_validate
        if self.return_validate:
            # test_size = 1/(n_splits - 1) if n_splits > 2 else 0.5
            test_size = 1 / n_splits
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y, groups=None):
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedStratifiedShuffleSplit(StratifiedShuffleSplit):
    def __init__(
        self,
        test_size: float,
        train_size: float,
        n_splits: int = 5,
        validate_size: Optional[float] = None,
        return_validate: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ):

        self.return_validate = return_validate
        if self.return_validate:
            if validate_size is None:
                validate_size = 1 - test_size - train_size
        else:
            validate_size = 0

        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size + validate_size,
            random_state=random_state,
        )

        if self.return_validate:
            total_size = validate_size + train_size
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validate_size / total_size,
                train_size=train_size / total_size,
                random_state=random_state,
            )

    def split(self, X, y, groups=None):
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedLeaveOneGroupOut(LeaveOneGroupOut):
    def __init__(self, return_validate: bool = True):
        super().__init__()
        self.return_validate = return_validate
        if self.return_validate:
            self.validate_spliter = LeaveOneGroupOut()

    def split(self, X, y=None, groups=None):
        if groups is None and y is not None:
            groups = self._generate_sequential_groups(y)
        n_splits = super().get_n_splits(groups=groups)
        for train, test in super().split(X, y, groups):
            if self.return_validate:
                n_repeat = np.random.randint(1, n_splits)
                validate_iter = self.validate_spliter.split(
                    X[train], y[train], groups[train]
                )
                for i in range(n_repeat):
                    train_ind, validate_ind = next(validate_iter)
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test

    def _generate_sequential_groups(self, y):
        labels = np.unique(y)
        groups = np.zeros((len(y)))
        inds = [y == label for label in labels]
        n_labels = [np.sum(ind) for ind in inds]
        if len(np.unique(n_labels)) > 1:
            warnings.warn(
                "y is not balanced, the generated groups is not balanced as well.",
                RuntimeWarning,
            )
        for ind, n_label in zip(inds, n_labels):
            groups[ind] = np.arange(n_label)
        return groups


def generate_kfold_indices(
    meta: DataFrame,
    kfold: int = 5,
    random_state: Optional[Union[int, RandomState]] = None,
):
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedStratifiedKFold(
                n_splits=kfold, shuffle=True, random_state=random_state
            )
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_kfold_indices(k: int, meta: DataFrame, indices):
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def generate_loo_indices(meta: DataFrame):
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedLeaveOneGroupOut()
            groups = np.arange(np.sum(ix))
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix))), groups=groups
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_loo_indices(k: int, meta: DataFrame, indices):
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def match_loo_indices_dict(
        X: Dict,
        y: Dict,
        meta: DataFrame,
        indices,
        k: int
):
    train_X, dev_X, test_X = defaultdict(list), defaultdict(list), defaultdict(list)
    train_y, dev_y, test_y = defaultdict(list), defaultdict(list), defaultdict(list)
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_index, sub_id in enumerate(subjects):
        for e_name in event_names:
            train_idx = list(indices[sub_id][e_name][k][0])
            dev_idx = list(indices[sub_id][e_name][k][1])
            test_idx = list(indices[sub_id][e_name][k][2])
            train_X[e_name].extend([X[e_name][sub_index][i] for i in train_idx])
            dev_X[e_name].extend([X[e_name][sub_index][i] for i in dev_idx])
            test_X[e_name].extend([X[e_name][sub_index][i] for i in test_idx])
            train_y[e_name].extend([y[e_name][sub_index][i] for i in train_idx])
            dev_y[e_name].extend([y[e_name][sub_index][i] for i in dev_idx])
            test_y[e_name].extend([y[e_name][sub_index][i] for i in test_idx])

    return dict(train_X), dict(train_y), dict(dev_X), \
        dict(dev_y), dict(test_X), dict(test_y)


def generate_shuffle_indices(
    meta: DataFrame,
    n_splits: int = 5,
    test_size: float = 0.1,
    validate_size: float = 0.1,
    train_size: float = 0.8,
    random_state: Optional[Union[int, RandomState]] = None,
):
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedStratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=train_size,
                test_size=test_size,
                validate_size=validate_size,
                return_validate=True,
                random_state=random_state,
            )
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_shuffle_indices(k: int, meta: DataFrame, indices):
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def generate_char_indices(
    meta: DataFrame,
    kfold: int = 6,
    random_state: Optional[Union[int, RandomState]] = None,
):
    """ Generate the trail index of train set, validation set and test set.
        This method directly manipulate characters
        -author: Jieyu Wu
        -Created on: 2023-03-17
        -update log:

        Parameters
        ----------
            meta: DataFrame,
                meta of all trials.
            kfold: int,
                Number of folds for cross validation.
            random_state: Optional[Union[int, RandomState]],
                State of random, default: None.
        Returns:
        ----------
            indices: list,
                Trial index for train set, validation set and test set.
                Ensemble in a tuple.
        """
    subjects = meta["subject"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        # classes_indices = {}
        # char_total = meta.event.__len__()
        k_indices = []
        ix = sub_ix
        spliter = EnhancedStratifiedKFold(
            n_splits=kfold, shuffle=True, random_state=random_state
        )
        for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
        ):
            k_indices.append((ix_train, ix_val, ix_test))
        classes_indices = k_indices

        indices[sub_id] = classes_indices
    return indices


def match_char_kfold_indices(k: int, meta: DataFrame, indices):
    """ Divide train set, validation set and test set.
        This method directly manipulate characters
        -author: Jieyu Wu
        -Created on: 2023-03-17
        -update log:

        Parameters
        ----------
            k: int,
                Number of folds for cross validation.
            meta: DataFrame,
                meta of all trials.
            indices: list,
                indices of trial index.
        Returns:
        ----------
            train_ix, val_ix, test_ix: list
                trial index for train set, validation set and test set.
        """
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    for sub_id in subjects:
        sub_meta = meta[(meta["subject"] == sub_id)]
        train_ix.append(
            sub_meta.iloc[indices[sub_id][k][0]].index.to_numpy()
        )
        val_ix.append(sub_meta.iloc[indices[sub_id][k][1]].index.to_numpy())
        test_ix.append(
            sub_meta.iloc[indices[sub_id][k][2]].index.to_numpy()
        )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix
