# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/08
# License: MIT License
# update log:2023-12-10 by sunxiwang 18822197631@163.com


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

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

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
    else (train, test) as the sklearn StratifiedKFold.fit the validate size should be the same as the test size.

    Hierarchical K-fold cross-validation.
    When the samples are unbalanced,
    the data set is divided according to the proportion of each type of sample to the total sample.

    Performs hierarchical k-fold cross-validation that can contain validation sets.
    The sample size of the validation set will be the same as that of the test set.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    n_splits : int
        Cross validation fold, default is 5.
    shuffle: bool
        Whether to scramble the sample order. The default is False.
    return_validate: bool
        Whether a validation set is required, which defaults to True.
    random_state: int or numpy.random.RandomState()
        Random initial state. When shuffle is True,
        random_state determines the initial ordering of the samples,
        hrough which the randomness of the selection of various data samples in each compromise can be controlled.
        See sklearn. Model_selection. StratifiedKFold () for details. The default is None.

    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.


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
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).

        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
           2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.
            groups: None
                Ignorable parameter, used only for version matching.


            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.
            """
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedStratifiedShuffleSplit(StratifiedShuffleSplit):
    """Hierarchical random cross validation.
    When the samples are unbalanced,
    the data set is divided according to the proportion of each type of sample to the total sample.
    Perform hierarchical random cross validation that can contain validation sets.
    The sample size of the validation set will be the same as that of the test set.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    test_size: float
        Test set ratio (0-1).
    train_size: float
        Train set ratio (0-1).
    n_splits: int
        Cross validation fold, default is 5.
    validate_size: float or None
        The proportion of the validation set (when return_validate is True) (0-1), defaults to None.
    return_validate: bool
        Whether a validation set is required, which defaults to True.
    random_state: int or numpy.random.RandomState()
        Random initial state. See sklearn. Model_selection. StratifiedShuffleSplit () for details,
        the default value is None.


    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.



    """
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
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).


        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
           2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.
            groups: None
                Ignorable parameter, used only for version matching.


            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.
        """
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedLeaveOneGroupOut(LeaveOneGroupOut):
    """
    Leave one method for cross-validation.
    Performs leave-one method cross validation that can contain validation sets.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    return_validate: bool
        Whether a validation set is required, which defaults to True.


    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.
    """
    def __init__(self, return_validate: bool = True):
        super().__init__()
        self.return_validate = return_validate
        if self.return_validate:
            self.validate_spliter = LeaveOneGroupOut()

    def split(self, X, y=None, groups=None):
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).

        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
            2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.Further adjustment is required by _generate_sequential_groups(y).
            groups: None
                The grouping label of the sample used when the data set is split into training,
                validation (return_validate is True), and test sets.
                The number of groups (the number of validation breaks) is calculated by this parameter.
                The number of groups here actually determines the sample size of the "one" part of the leave-one method.
                For example, a set composed of 6 samples with the group number
                [1,1,2,3,3] means that the set is divided into three parts,
                with the number of samples being 2, 1 and 3 respectively.
                In the reserve-one method, the set composed of 2 samples,1 samples and 3 samples is regarded as a test set,
                and the remaining part is regarded as a training set.
                groups can be entered externally or computed by an internal function based on the category label.

            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.

            See Also:
            -------
            get_n_splits：Returns the number of packet iterators, that is, the number of packets.
            _generate_sequential_groups：The sample group tag “groups” is generated.
        """

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
    """The EnhancedStratifiedKFold class is invoked at the meta data structure level
    to generate cross-validation grouping subscripts.
    The subscript of K-fold cross-validation is generated based on meta class data structure.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.
    kfold: int
        Cross validation fold, default is 5.
    random_state: int 或 numpy.random.RandomState
        Random initial state, defaults to None.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format,
        and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name
        and the value is the attempt index subscript k_indices for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test)
        composed of the indexes of the corresponding data sets.


    """
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
    """At the level of meta data structure,
    hierarchical K-fold cross-validation packet subscripts are matched to generate specific indexes.
    Based on meta class data structure and combined with the output results of generate_kfold_indices(),
    the specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        Subscript dictionary generated by generate_kfold_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trials required for k-fold verification
        of the full class data of all subjects (i.e., meta-class data).
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.
    """
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
    """
    The EnhancedLeaveOneGroupOut class is invoked at the meta data structure level
    to generate cross-validation grouping subscripts.
    The subscript of leave-one method cross-validation is generated based on meta class data structure.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format,
        and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name
        and the value is the attempt index subscript k_indices for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test)
        composed of the indexes of the corresponding data sets.
    """
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
    """
    At the meta data structure level, a method is matched
    to cross-validate the grouping subscript and generate the specific index.
    Based on the meta class data structure and combined with the output of generate_loo_indices(),
    the specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        Subscript dictionary generated by generate_loo_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trial required by the k-fold verification of meta class data.
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.

    """
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
    """
    Level in the meta data structure called EnhancedStratifiedShuffleSplit class,
    generating cross validation grouping subscript.
    Generate hierarchical random cross-validation subscripts based on meta-class data structures.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.
    n_splits: int
        Random verification fold, default is 5.
    test_size: float
        The default value is 0.1.
    validate_size: int
        The default value is 0.1, which is the same as that of the test set.
    train_size: int
        The proportion of the number of training sets is 0.8 by default
        (the sum of the proportion of test sets and verification sets is 1).
    random_state: int 或 numpy.random.RandomState
        Random initial state, defaults to None.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format, and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name and the value is the attempt index subscript k_indices
        for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test) composed of the indexes of the corresponding
        data sets.

    """
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
    """
    Random cross-validation grouping subscripts are matched at the meta data structure level
    to generate specific indexes.
    Based on the meta class data structure and combined with the output of generate_shuffle_indices(),
    a specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        A subscript dictionary generated by generate_shuffle_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trial required by the k-fold verification of meta class data.
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.

    """
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

        author: WuJieYu

        Created on: 2023-03-17

        update log:2023-12-26 by sunchang<18822197631@163.com>

        Parameters
        ----------
            meta: DataFrame
                meta of all trials.
            kfold: int
                Number of folds for cross validation.
            random_state: Optional[Union[int, RandomState]]
                State of random, default: None.
        Returns
        ----------
            indices: list
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

        author: WuJieYu

        Created on: 2023-03-17

        update log:2023-12-26 by sunchang<18822197631@163.com>

        Parameters
        ----------
            k: int
                Number of folds for cross validation.
            meta: DataFrame
                meta of all trials.
            indices: list
                indices of trial index.
        Returns
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
