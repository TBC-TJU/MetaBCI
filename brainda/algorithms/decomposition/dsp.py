# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
"""
Discriminal Spatial Patterns.
"""
from typing import Optional, List, Tuple, Dict

import numpy as np
from scipy.linalg import eigh
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .base import robust_pattern
from .cca import FilterBankSSVEP

def xiang_dsp_kernel(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """dsp algorihtm based on paper[1].

    Parameters
    ----------
    X : ndarray
        EEG data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials, )
    
    Returns
    -------
    W: ndarray
        filters, shape (n_channels, n_filters)
    D: ndarray
        eigenvalues in descending order
    M: ndarray
        template for all classes, shape (n_channel, n_samples)
    A: ndarray
        spatial patterns, shape (n_channels, n_filters)

    Notes
    -----
    1. the implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.    
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(y==label) for label in labels])
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms, Ss =zip(*[
        (np.mean(X[y==label], axis=0), np.sum(np.matmul(X[y==label], np.swapaxes(X[y==label], -1, -2)), axis=0)) for label in labels
        ])
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(Ss - n_labels[:, np.newaxis, np.newaxis]*np.matmul(Ms, np.swapaxes(Ms, -1, -2)), axis=0) 
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(n_labels[:, np.newaxis, np.newaxis]*np.matmul(Ms, np.swapaxes(Ms, -1, -2)), axis=0)

    D, W = eigh(Sb, Sw)
    ix = np.argsort(D)[::-1] # in descending order
    D, W = D[ix], W[:, ix]
    A = robust_pattern(W, Sb, W.T@Sb@W)

    return W, D, M, A

def xiang_dsp_feature(W: ndarray, M: ndarray, X: ndarray,
        n_components: int = 1) -> ndarray:
    """Return DSP features in paper [1]_.

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    M: ndarray
        common template for all classes, shape (n_channel, n_samples)
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

    Notes
    -----
    1. instead of meaning of filtered signals in paper [1]_., we directly return filtered signals.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    W, M, X = np.copy(W), np.copy(M), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_components].T, X - M)
    return features

class DSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self,
            n_components: int = 1,
            transform_method: str = 'corr'):
        self.n_components = n_components
        self.transform_method = transform_method

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        X -= np.mean(X, axis=-1, keepdims=True)
        self.classes_ = np.unique(y)
        self.W_, self.D_, self.M_, self.A_ = xiang_dsp_kernel(X, y)

        self.templates_ = np.stack([
            np.mean(xiang_dsp_feature(self.W_, self.M_, X[y==label], n_components=self.W_.shape[1]), axis=0) for label in self.classes_
            ])
        return self
        
    def transform(self, X: ndarray):
        n_components = self.n_components
        X -= np.mean(X, axis=-1, keepdims=True)
        features = xiang_dsp_feature(self.W_, self.M_, X, n_components=n_components)
        if self.transform_method is None:
            return features.reshape((features.shape[0], -1))
        elif self.transform_method == 'mean':
            return np.mean(features, axis=-1)
        elif self.transform_method == 'corr':
            return self._pearson_features(features, self.templates_[:, :n_components, :])
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
        corr = (X@templates.T) / (templates.shape[1]-1)
        corr = istd_X * corr * istd_templates.T
        return corr

    def predict(self, X: ndarray):
        feat = self.transform(X)
        if self.transform_method == 'corr':
            labels = self.classes_[np.argmax(feat, axis=-1)]
        else:
            raise NotImplementedError()
        return labels

class FBDSP(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        transform_method: str = 'corr',
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.transform_method = transform_method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            DSP(n_components=n_components, transform_method=transform_method),
            filterweights=filterweights,
            n_jobs=n_jobs
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels
