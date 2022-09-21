# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/29
# License: MIT License
"""
CCA and its variants.
"""
from typing import Optional, List
from functools import partial

import numpy as np
from scipy.linalg import eigh, pinv, qr
from scipy.stats import pearsonr
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC
from joblib import Parallel, delayed

from .base import FilterBankSSVEP

def _ged_wong(
        Z: ndarray, 
        D: Optional[ndarray] = None, P: Optional[ndarray] = None, 
        n_components=1,
        method='type1'):
    A = Z
    if D is not None:
        A = D.T@A
    if P is not None:
        A = P.T@A
    A = A.T@A
    if method == 'type1':
        B = Z
        if D is not None:
            B = D.T@Z
        B = B.T@B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == 'type2':
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)
    ind = np.argsort(D)[::-1]
    D, W = D[ind], W[:, ind]
    return D[:n_components], W[:, :n_components]

def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])
    Q, R = qr(Yf.T, mode='economic')
    P = Q@Q.T
    Z = X.T
    _, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    V = pinv(R)@Q.T@X.T@U # V for Yf
    return U, V

def _scca_feature(X: ndarray, Yf: ndarray, n_components: int = 1):
    rhos = []
    for Y in Yf:
        U, V = _scca_kernel(X, Y)
        a = U[:, :n_components].T@X
        b = V[:, :n_components].T@Y
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)

class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            n_jobs: Optional[int] =None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, 
            X: Optional[ndarray] = None, 
            y: Optional[ndarray] = None, 
            Yf: Optional[ndarray] = None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Yf = self.Yf_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_scca_feature, n_components=n_components))(a, Yf) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)
        return labels

class FBSCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            SCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs
        )

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            features = np.mean(features, axis=1)
        labels = np.argmax(features, axis=-1)
        return labels

def _itcca_feature(
        X: ndarray, templates: ndarray, 
        Us: Optional[ndarray] = None, n_components: int = 1, method: str = 'itcca1'):
    rhos = []
    if method == 'itcca1':
        for Xk in templates:
            U, V = _scca_kernel(X, Xk)
            a = U[:, :n_components].T@X
            b = V[:, :n_components].T@Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    elif method == 'itcca2':
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T@X
            b = U[:, :n_components].T@Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)

class ItCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            method: str = 'itcca2',
            n_jobs: Optional[int] =None):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, 
            X: Optional[ndarray], 
            y: Optional[ndarray], 
            Yf: Optional[ndarray] = None):
        if self.method == 'itcca2' and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])
        if self.method == 'itcca2':
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = Yf
            self.Us_, self.Vs_ = zip(*[_scca_kernel(self.templates_[i], self.Yf_[i]) for i in range(len(self.classes_))])
            self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        method = self.method
        Us = None
        if method == 'itcca2':
            Us = self.Us_
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_itcca_feature, Us=Us, n_components=n_components, method=method))(a, templates) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBItCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = 'itcca2',
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ItCCA(n_components=n_components, method=method, n_jobs=1),
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

def _mscca_feature(
        X: ndarray, templates: ndarray, 
        U: Optional[ndarray] = None, n_components: int = 1):
    rhos = []
    for Xk in zip(templates):
        a = U[:, :n_components].T@X
        b = U[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)

class MsCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Note: MsCCA heavily depends on Yf, thus the phase information should be included when designs Yf.
    
    """
    def __init__(self, 
            n_components: int = 1, 
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, 
            X: Optional[ndarray], 
            y: Optional[ndarray], 
            Yf: Optional[ndarray] = None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.U_, self.V_ = _scca_kernel(
            np.concatenate(self.templates_, axis=-1), np.concatenate(self.Yf_, axis=-1))
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        U = self.U_
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_mscca_feature, U=U, n_components=n_components))(a, templates) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBMsCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsCCA(n_components=n_components, n_jobs=1),
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

def _ecca_feature(
    X: ndarray, templates: ndarray, Yf: ndarray, 
    Us: Optional[ndarray] = None, n_components: int = 1):
    if Us is None:
        Us, _ = zip(*[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))])
        Us = np.stack(Us)
    rhos = []
    for Xk, Y, U3 in zip(templates, Yf, Us):
        rho = []
        # 14a, 14d
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T@X
        b = V1[:, :n_components].T@Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        a = U1[:, :n_components].T@X
        b = U1[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        # 14b
        U2, _ = _scca_kernel(X, Xk)
        a = U2[:, :n_components].T@X
        b = U2[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        # 14c
        a = U3[:, :n_components].T@X
        b = U3[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        rho = np.array(rho)
        rho = np.sum(np.sign(rho)*(rho**2))
        rhos.append(rho)
    return rhos

class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, 
            X: Optional[ndarray], 
            y: Optional[ndarray], 
            Yf: Optional[ndarray] = None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(*[_scca_kernel(self.templates_[i], self.Yf_[i]) for i in range(len(self.classes_))])
        self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_ecca_feature, Us=Us, n_components=n_components))(a, templates, Yf) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBECCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ECCA(n_components=n_components, n_jobs=1),
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

def _ttcca_template(
    X: ndarray, y: ndarray, y_sub: Optional[ndarray] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    labels = np.unique(y)
    if y_sub is None:
        templates = np.stack([np.mean(X[y==label], axis=0) for label in labels])
    else:
        subjects = np.unique(y_sub)
        templates = 0
        for sub_id in subjects:
            templates += np.stack([np.mean(X[(y==label)&(y_sub==sub_id)], axis=0) for label in labels])
        templates /= len(subjects)
    return templates

def _ttcca_feature(
    X: ndarray, templates: ndarray, Yf: ndarray, 
    Us: Optional[ndarray] = None, n_components: int = 1):
    if Us is None:
        Us, _ = zip(*[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))])
        Us = np.stack(Us)
    rhos = []
    for Xk, Y, U2 in zip(templates, Yf, Us):
        rho = []
        # rho1
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T@X
        b = V1[:, :n_components].T@Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        # rho3
        a = U1[:, :n_components].T@X
        b = U1[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        # rho2
        a = U2[:, :n_components].T@X
        b = U2[:, :n_components].T@Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho.append(pearsonr(a, b)[0])
        rho = np.array(rho)
        rho = np.sum(np.sign(rho)*(rho**2))
        rhos.append(rho)
    return rhos

class TtCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, 
            X: Optional[ndarray], 
            y: Optional[ndarray], 
            Yf: Optional[ndarray] = None,
            y_sub: Optional[ndarray] = None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        self.templates_ = _ttcca_template(X, y, y_sub=y_sub)

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(*[_scca_kernel(self.templates_[i], self.Yf_[i]) for i in range(len(self.classes_))])
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_ttcca_feature, Us=Us, n_components=n_components))(a, templates, Yf) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBTtCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TtCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None, y_sub: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf, y_sub=y_sub)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels

def _msetcca_kernel1(X: ndarray):
    """Multi-set CCA1 (MsetCCA1).

    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P@P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    X = np.reshape(X, (-1, N))
    T = U.T@X
    return U, T

def _msetcca_kernel2(X: ndarray, Xk: ndarray, Yf: ndarray):
    C, N = X.shape
    M = 3
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P@P.T
    Z = block_diag([X, Yf, Xk]).T
    D, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    return U[:C, :], D

def _msetcca_feature2(
        X: ndarray, templates: ndarray, Yf: ndarray, 
        n_components: int = 1):
    feat = []
    for Xk, Y in zip(templates, Yf):
        U, D = _msetcca_kernel2(X, Xk, Y)
        A = U[:, :n_components].T@X
        B = U[:, :n_components].T@Xk
        rho = np.array([pearsonr(a, b)[0] for a, b in zip(A, B)])
        rho = D[0]*np.sign(rho)*(rho**2)
        feat.append(rho)
    feat = np.concatenate(feat, axis=-1)
    return feat

class MsetCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            method: str = 'msetcca2', 
            n_jobs: Optional[ndarray] = -1):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, 
            X: ndarray, 
            y: ndarray, 
            Yf: Optional[ndarray] = None):
        if self.method == 'msetcca2' and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        if self.method == 'msetcca2':
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = Yf
            feat = self.transform(X)
            self.clf_ = SVC()
            self.clf_.fit(feat, y)
        elif self.method == 'msetcca1':
            self.Us_, self.Ts_ = zip(*[_msetcca_kernel1(X[y==label]) for label in self.classes_])
            self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        if self.method == 'msetcca1':
            Ts = self.Ts_
            rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_scca_feature, n_components=n_components))(a, Ts) for a in X)
            rhos = np.stack(rhos)
        elif self.method == 'msetcca2':
            templates = self.templates_
            Yf = self.Yf_
            rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_msetcca_feature2, n_components=n_components))(a, templates, Yf) for a in X)
            rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        feat = self.transform(X)
        if self.method == 'msetcca1':
            labels = self.classes_[np.argmax(feat, axis=-1)]
        elif self.method == 'msetcca2':
            labels = self.clf_.predict(feat)
        return labels

class FBMsetCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = 'msetcca2',
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCA(n_components=n_components, method=method, n_jobs=-1),
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

def _msetccar_kernel(X: ndarray, Yf: ndarray):
    """Multi-set CCA1 with reference signals (MsetCCA-R).

    X: (n_trials, n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = min(C, Yf.shape[0])
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode='economic')
    P = P@Q@Q.T@P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    X = np.reshape(X, (-1, N))
    T = U.T@X
    return U, T

class MsetCCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            n_jobs: Optional[ndarray] = -1):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, 
            X: ndarray, 
            y: ndarray, 
            Yf: Optional[ndarray] = None):
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_, self.Ts_ = zip(*[_msetccar_kernel(X[y==label], self.Yf_[i]) for i, label in enumerate(self.classes_)])
        self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components

        Ts = self.Ts_
        rhos = Parallel(n_jobs=self.n_jobs)(delayed(partial(_scca_feature, n_components=n_components))(a, Ts) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

class FBMsetCCAR(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCAR(n_components=n_components, n_jobs=-1),
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

def _trca_kernel(X: ndarray):
    """TRCA.
    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P@P.T
    Z = np.hstack(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    return U

def _trca_feature(
        X: ndarray, templates: ndarray, 
        Us: Optional[ndarray] = None,
        n_components: int = 1,
        ensemble: bool = True):
    rhos = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T@X
            b = U[:, :n_components].T@Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        for Xk in templates:
            a = U.T@X
            b = U.T@Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return rhos

class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            ensemble: bool = True,
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs
    
    def fit(self, 
            X: ndarray, 
            y: ndarray,
            Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        self.Us_ = np.stack([_trca_kernel(X[y==label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_trca_feature, Us=Us, n_components=n_components, ensemble=ensemble))(a, templates) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

class FBTRCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCA(n_components=n_components, ensemble=ensemble, n_jobs=1),
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

def _trcar_kernel(
        X: ndarray, Yf: ndarray):
    """TRCA-R.
    X: (n_trials, n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode='economic')
    P = P@Q@Q.T@P.T
    Z = np.hstack(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components) # U for X
    return U

class TRCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            ensemble: bool = True,
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs
    
    def fit(self, 
            X: ndarray, 
            y: ndarray,
            Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_ = np.stack([_trcar_kernel(X[y==label], self.Yf_[i]) for i, label in enumerate(self.classes_)])
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_trca_feature, Us=Us, n_components=n_components, ensemble=ensemble))(a, templates) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

class FBTRCAR(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCAR(n_components=n_components, ensemble=ensemble, n_jobs=1),
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


