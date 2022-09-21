# -*- coding: utf-8 -*-
"""
Least-squares Transformation (LST).

See https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e.
"""
import numpy as np
from numpy import ndarray
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

def lst_kernel(S: ndarray, T: ndarray):
    P = T@S.T@pinv(S@S.T)
    return P

class LST(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray):
        X = X.reshape((-1, *X.shape[-2:])) # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y==label], axis=0) for label in self.classes_]
        return self

    def transform(self, X: ndarray, y: ndarray):
        X = np.copy(X)
        X = X.reshape((-1, *X.shape[-2:])) # n_trials, n_channels, n_samples
        Ts = np.zeros_like(X)
        for i, label in enumerate(self.classes_):
            Ts[y==label] = self.T_[i]
        P = np.stack(
            Parallel(n_jobs=self.n_jobs)(delayed(lst_kernel)(S, T) for S, T in zip(X, Ts)))
        X = P@X
        return X
