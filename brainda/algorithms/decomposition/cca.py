# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/29
# License: MIT License
"""
CCA and its variants.
"""
from typing import Optional, Union, List, Tuple, Dict
from itertools import combinations
from functools import partial

import numpy as np
from scipy.linalg import eigh, cholesky, inv

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ShuffleSplit
from sklearn.pipeline import make_pipeline, clone
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from joblib import Parallel, delayed

from .base import robust_pattern, FilterBank

class ExtendCCA(BaseEstimator, TransformerMixin):
    """Extend CCA in paper [1]_.

    Modified from https://github.com/edwin465/SSVEP-MSCCA-MSTRCA/blob/master/extendedCCA.m

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer interface[J]. Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.
    """
    def __init__(self, 
            n_components: int = 1,
            return_r2: bool = True, 
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.return_r2 = return_r2
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        if Yf is None:
            raise ValueError("Yf must be supplied")
        if len(self.classes_) != len(Yf):
            raise ValueError("Yf should match the number of classes")
        X, y = np.copy(X), np.copy(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)

        self.estimators_xxk_ = [CCA(n_components=self.n_components, max_iter=1000) for _ in range(len(self.classes_))]
        self.estimators_xyk_ = [CCA(n_components=self.n_components, max_iter=1000) for _ in range(len(self.classes_))]
        self.estimators_xkyk_ = [CCA(n_components=self.n_components, max_iter=1000) for _ in range(len(self.classes_))]
        self.Xk_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])
        self.Yk_ = Yf
        return self

    def transform(self, X):
        rho = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self._fit_transform)(X, k) for k in range(len(self.classes_))), axis=1)
        if self.return_r2:
            R2 = np.sign(rho)*np.square(rho)
            return np.sum(R2, axis=-1)
        else:
            return np.reshape(rho, (rho.shape[0], -1))

    def _fit_transform(self, X, k):
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)

        Yk = self.Yk_[k]
        Xk = self.Xk_[k]
        rho = np.zeros((len(X), 5))
        for i, Xi in enumerate(X):
            self.estimators_xyk_[k].fit(Xi.T, Yk.T)
            self.estimators_xxk_[k].fit(Xi.T, Xk.T)
            self.estimators_xkyk_[k].fit(Xk.T, Yk.T)
            xscore, yscore = self.estimators_xyk_[k].transform(Xi.T, Yk.T)
            rho[i, 0] = self._pearson_features(xscore.T, yscore.T)
            yscore = self.estimators_xyk_[k].transform(Xk.T)
            rho[i, 2] = self._pearson_features(xscore.T, yscore.T)
            xscore = self.estimators_xxk_[k].transform(Xi.T)
            yscore = self.estimators_xxk_[k].transform(Xk.T)
            rho[i, 1] = self._pearson_features(xscore.T, yscore.T)
            xscore = self.estimators_xkyk_[k].transform(Xi.T)
            yscore = self.estimators_xkyk_[k].transform(Xk.T)
            rho[i, 3] = self._pearson_features(xscore.T, yscore.T)
            xscore = self.estimators_xxk_[k].transform(Xk.T)
            rho[i, 4] = self._pearson_features(xscore.T, yscore.T)
        return rho

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

class EnsembleExtendCCA(FilterBank):
    """Filterbank Extend CCA in paper [1]_.

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer interface[J]. Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.
    """
    def __init__(self,
            n_components: int = 1,
            return_r2: bool = True, 
            n_jobs: Optional[int] = None,
            filterbank: Optional[List[ndarray]] = None,
            filterweights: Optional[ndarray] = None):
        self.n_components = n_components
        self.return_r2 = return_r2
        self.n_jobs = n_jobs
        self.filterbank = filterbank
        self.filterweights = filterweights
        if filterweights is not None:
            if filterbank is None:
                self.filterweights = None
            else:
                if len(filterweights) != len(filterbank):
                    raise ValueError("the len of filterweights must be the same as that of filterbank")
        super().__init__(
            ExtendCCA(
                n_components=n_components,
                return_r2=return_r2,
                n_jobs=n_jobs),
            filterbank=filterbank)

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        # transform filterbank
        X = self.transform_filterbank(X)
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(X))]
        for i, estimator in enumerate(self.estimators_):
            estimator.fit(X[i], y, Yf=Yf)
        return self

    def transform(self, X: ndarray):
        features = super().transform(X)
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            return np.sum(features*self.filterweights[np.newaxis, :, np.newaxis], axis=1)




