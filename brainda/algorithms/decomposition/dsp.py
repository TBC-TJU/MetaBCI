# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
"""
Discriminal Spatial Patterns.
"""
from typing import Optional, Union, List, Tuple, Dict
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from joblib import Parallel, delayed

from .base import robust_pattern, FilterBank

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

class DSP(BaseEstimator, TransformerMixin):
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None,
            transform_method: Optional[str] = None):
        self.n_components = n_components
        self.max_components = max_components
        self.transform_method = transform_method

    def fit(self, X: ndarray, y: ndarray):
        X -= np.mean(X, axis=-1, keepdims=True)
        X /= np.std(X, axis=(-2, -1), keepdims=True)
        self.classes_ = np.unique(y)
        self.W_, self.D_, self.M_, self.A_ = xiang_dsp_kernel(X, y)

        # auto-tuning
        if self.n_components is None:
            estimator = make_pipeline(*[DSP(n_components=self.n_components, transform_method=self.transform_method), SVC()])
            if self.max_components is None:
                params = {'dsp__n_components': np.arange(1, self.W_.shape[1]+1)}
            else:
                params = {'dsp__n_components': np.arange(1, self.max_components+1)}
            
            n_splits = np.min(np.unique(y, return_counts=True)[1])
            n_splits = 5 if n_splits > 5 else n_splits

            gs = GridSearchCV(estimator,
                param_grid=params, scoring='accuracy', 
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True), refit=False, n_jobs=-1, verbose=False)
            gs.fit(X, y)
            self.best_n_components_ = gs.best_params_['dsp__n_components']

        self.templates_ = np.stack([
            np.mean(xiang_dsp_feature(self.W_, self.M_, X[y==label], n_components=self.W_.shape[1]), axis=0) for label in self.classes_
            ])
        return self
        
    def transform(self, X: ndarray):
        n_components = self.best_n_components_ if self.n_components is None else self.n_components
        X -= np.mean(X, axis=-1, keepdims=True)
        X /= np.std(X, axis=(-2, -1), keepdims=True)
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

class FBDSP(FilterBank):
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None,
            filterbank: Optional[List[ndarray]] = None,
            filterweights: Optional[ndarray] = None):
        self.n_components = n_components
        self.max_components = max_components
        self.filterbank = filterbank
        self.filterweights = filterweights
        if filterweights is not None:
            if filterbank is None:
                self.filterweights = None
            else:
                if len(filterweights) != len(filterbank):
                    raise ValueError("the len of filterweights must be the same as that of filterbank")
        super().__init__(
            DSP(
                n_components=n_components,
                max_components=max_components,transform_method='corr'),
            filterbank=filterbank)

    def transform(self, X: ndarray):
        features = super().transform(X)
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            return np.sum(features*self.filterweights[np.newaxis, :, np.newaxis], axis=1)


