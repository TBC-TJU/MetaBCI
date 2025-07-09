# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/9/18
# License: MIT License
"""
SSCOR.
"""
from typing import Optional, List, Tuple
from functools import partial

import numpy as np
from scipy.linalg import eigh, cholesky, inv

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

from .base import robust_pattern, FilterBank


def sscor_kernel(
    X: ndarray, y: Optional[ndarray] = None, n_jobs: Optional[int] = None
) -> Tuple[ndarray, ndarray, ndarray]:
    """The kernel part in SSCOR algorithm based on paper[1]_., [2]_.

    Modified from https://github.com/mnakanishi/TRCA-SSVEP/blob/master/src/train_sscor.m

    Parameters
    ----------
    X : ndarray
        EEG data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials, ), not used here
    n_jobs: int, optional
        the number of jobs to use, default None

    Returns
    -------
    W: ndarray
        filters, shape (n_channels, n_filters)
    D: ndarray
        eigenvalues in descending order
    A: ndarray
        spatial patterns, shape (n_channels, n_filters)

    References
    ----------
    .. [1] Kumar G R K, Reddy M R. Designing a sum of squared correlations framework for enhancing SSVEP-based BCIs[J].
           IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 27(10): 2044-2050.
    .. [2] Kumar G R K, Reddy M R. Correction to “Designing a Sum of Squared Correlations Framework for Enhancing SSVEP
           Based BCIs”[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28(4): 1044-1045.
    """
    X = np.copy(X)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    mean_X = np.mean(X, axis=0)
    K1 = cholesky(mean_X @ mean_X.T)  # upper-triangular X=K.T@K
    iK1 = inv(K1)
    xC = mean_X @ np.transpose(X, axes=(0, 2, 1))
    C = X @ np.transpose(X, axes=(0, 2, 1))

    def target(iK1, xCi, Ci):
        Ki = cholesky(Ci)
        Gi = iK1.T @ xCi @ inv(Ki)
        return Gi.T @ Gi

    target = partial(target, iK1)
    G_T_G = np.sum(
        Parallel(n_jobs=n_jobs)(delayed(target)(xCi, Ci) for xCi, Ci in zip(xC, C)),
        axis=0,
    )
    D, W = eigh(G_T_G)
    ind = np.argsort(D)[::-1]
    D, W = D[ind], W[:, ind]
    W = iK1 @ W
    A = robust_pattern(W, G_T_G, W.T @ G_T_G @ W)
    return W, D, A


def sscor_feature(W: ndarray, X: ndarray, n_components: int = 1) -> ndarray:
    """Return sscor features.

    Modified from https://github.com/mnakanishi/TRCA-SSVEP/blob/master/src/test_sscor.m

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    n_components : int, optional
        the first k components to use, usually even number, by default 1

    Returns
    -------
    ndarray
        features of shape (n_trials, n_components, n_samples)

    Raises
    ------
    ValueError
        n_components should less than half of the number of channels
    """
    W, X = np.copy(W), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_components].T, X)
    return features


class SSCOR(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 1,
        transform_method: Optional[str] = None,
        ensemble: bool = False,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.transform_method = transform_method
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray):
        self.classes_ = np.unique(y)
        Ws, Ds, As = zip(
            *[
                sscor_kernel(X[y == label], n_jobs=self.n_jobs)
                for label in self.classes_
            ]
        )
        self.Ws_, self.Ds_, self.As_ = np.stack(Ws), np.stack(Ds), np.stack(As)

        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )
        return self

    def transform(self, X: ndarray):
        n_components = self.n_components

        if self.transform_method is None:
            features = np.concatenate(
                [sscor_feature(W, X, n_components=n_components) for W in self.Ws_],
                axis=-2,
            )
            features = np.reshape(features, (features.shape[0], -1))
            return features
        elif self.transform_method == "corr":
            if self.ensemble:
                W = np.transpose(self.Ws_[..., :n_components], (1, 0, 2))
                W = np.reshape(W, (W.shape[0], -1))
                X = sscor_feature(W, X, n_components=W.shape[1])
                features = [
                    self._pearson_features(
                        X, sscor_feature(W, template, n_components=W.shape[1])
                    )
                    for template in self.templates_
                ]
            else:
                features = [
                    self._pearson_features(
                        sscor_feature(W, X, n_components=n_components),
                        sscor_feature(W, template, n_components=n_components),
                    )
                    for W, template in zip(self.Ws_, self.templates_)
                ]
            features = np.concatenate(features, axis=-1)
            return features
        else:
            raise ValueError("non-supported transform method")

    def _pearson_features(self, X: ndarray, templates: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        templates = np.reshape(templates, (-1, *templates.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = templates - np.mean(templates, axis=-1, keepdims=True)
        X = np.reshape(X, (X.shape[0], -1))
        templates = np.reshape(templates, (templates.shape[0], -1))
        istd_X = 1 / np.std(X, axis=-1, keepdims=True)
        istd_templates = 1 / np.std(templates, axis=-1, keepdims=True)
        corr = (X @ templates.T) / (templates.shape[1] - 1)
        corr = istd_X * corr * istd_templates.T
        return corr


class FBSSCOR(FilterBank):
    """Filter Bank SSCOR method in paper [1]_., [2]_.

    filterbank and weights suggested in the paper.

    wp = [
        [6, 90], [14, 90], [22, 90], [30, 90], [38, 90], [46, 90], [54, 90], [62, 90], [70, 90], [78, 90]
    ]
    ws = [
        [4, 100], [10, 100], [16, 100], [24, 100], [32, 100], [40, 100], [48, 100], [56, 100], [64, 100], [72, 100]
    ]

    filterweights:
        np.arange(1, 11)**(-1.25) + 0.25

    References
    ----------
    .. [1] Kumar G R K, Reddy M R. Designing a sum of squared correlations framework for enhancing SSVEP-based BCIs[J].
           IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 27(10): 2044-2050.
    .. [2] Kumar G R K, Reddy M R. Correction to “Designing a Sum of Squared Correlations Framework for Enhancing SSVEP
           Based BCIs”[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28(4): 1044-1045.
    """

    def __init__(
        self,
        n_components: int = 1,
        ensemble: bool = False,
        n_jobs: Optional[int] = None,
        filterbank: List[ndarray] = [],
        filterweights: Optional[ndarray] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs
        self.filterbank = filterbank
        self.filterweights = filterweights
        if filterweights is not None:
            if filterbank is None:
                self.filterweights = None
            else:
                if len(filterweights) != len(filterbank):
                    raise ValueError(
                        "the len of filterweights must be the same as that of filterbank"
                    )

        super().__init__(
            SSCOR(
                n_components=n_components,
                transform_method="corr",
                ensemble=ensemble,
                n_jobs=n_jobs,
            ),
            filterbank=filterbank,
        )

    def transform(self, X: ndarray):  # type: ignore[override]
        features = super().transform(X)
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            return np.sum(
                features * self.filterweights[np.newaxis, :, np.newaxis], axis=1
            )
