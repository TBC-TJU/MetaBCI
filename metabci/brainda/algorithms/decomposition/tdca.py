# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/10/10
# License: MIT License
"""
Task Decomposition Component Analysis.
"""
import enum
from typing import Optional, List, Tuple, Dict
from functools import partial

import numpy as np
from scipy.linalg import eigh, qr
from scipy.stats import pearsonr
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import Parallel, delayed

from .base import robust_pattern, generate_cca_references
from .cca import FilterBankSSVEP
from .dsp import xiang_dsp_kernel, xiang_dsp_feature

def proj_ref(Yf: ndarray):
    Q, R = qr(Yf.T, mode='economic')
    P = Q@Q.T
    return P

def aug_2(X: ndarray, n_samples: int, l: int, P: ndarray, training: bool = True):
    X = X.reshape((-1, *X.shape[-2:]))
    n_trials, n_channels, n_points = X.shape
    if n_points < l+n_samples:
        raise ValueError("the length of X should be larger than l+n_samples.")
    aug_X = np.zeros((n_trials, (l+1)*n_channels, n_samples))
    if training:
        for i in range(l+1):
            aug_X[:, i*n_channels:(i+1)*n_channels, :] = X[..., i:i+n_samples]
    else:
        for i in range(l+1):
            aug_X[:, i*n_channels:(i+1)*n_channels, :n_samples-i] = X[..., i:n_samples]
    aug_Xp = aug_X@P
    aug_X = np.concatenate([aug_X, aug_Xp], axis=-1)
    return aug_X

def tdca_feature(
        X: ndarray, templates: ndarray, W: ndarray, M: ndarray, Ps: ndarray, l: int, 
        n_components: int = 1, training=False):
    rhos = []
    for Xk, P in zip(templates, Ps):
        a = xiang_dsp_feature(
                W, M, aug_2(X, P.shape[0], l, P, training=training), n_components=n_components)
        b = Xk[:n_components, :]
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return rhos

class TDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self,
            l: int,
            n_components: int = 1):
        self.l = l
        self.n_components = n_components

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray]):
        X -= np.mean(X, axis=-1, keepdims=True)
        self.classes_ = np.unique(y)
        self.Ps_ = [proj_ref(Yf[i]) for i in range(len(self.classes_))]

        aug_X, aug_Y = [], []
        for i, label in enumerate(self.classes_):
            aug_X.append(
                aug_2(
                    X[y==label], self.Ps_[i].shape[0], self.l, self.Ps_[i], training=True))
            aug_Y.append(y[y==label])

        aug_X = np.concatenate(aug_X, axis=0)
        aug_Y = np.concatenate(aug_Y, axis=0)
        self.W_, _, self.M_, _ = xiang_dsp_kernel(aug_X, aug_Y)

        self.templates_ = np.stack([
            np.mean(xiang_dsp_feature(self.W_, self.M_, aug_X[aug_Y==label], n_components=self.W_.shape[1]), axis=0) for label in self.classes_
            ])
        return self
        
    def transform(self, X: ndarray):
        n_components = self.n_components
        X -= np.mean(X, axis=-1, keepdims=True)
        X = X.reshape((-1, *X.shape[-2:]))
        rhos = [
            tdca_feature(
                tmp, self.templates_, self.W_, self.M_, self.Ps_, self.l, 
                n_components=n_components) for tmp in X]
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

class FBTDCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        l: int,
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.l = l
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TDCA(l, n_components=n_components),
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




