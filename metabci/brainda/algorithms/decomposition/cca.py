# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/29
# License: MIT License
"""
CCA and its variants.
"""
from typing import Optional, List, cast, Tuple, Any
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
from abc import abstractmethod, ABCMeta
from math import pow
from math import sqrt
from scipy import linalg as sLA


def _ged_wong(
    Z: ndarray,
    D: Optional[ndarray] = None,
    P: Optional[ndarray] = None,
    n_components=1,
    method="type1",
):
    if method != "type1" and method != "type2":
        raise ValueError("not supported method type")

    A = Z
    if D is not None:
        A = D.T @ A
    if P is not None:
        A = P.T @ A
    A = A.T @ A
    if method == "type1":
        B = Z
        if D is not None:
            B = D.T @ Z
        B = B.T @ B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == "type2":
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)

    D_exist = cast(ndarray, D)
    ind = np.argsort(D_exist)[::-1]
    D_exist, W = D_exist[ind], W[:, ind]
    return D_exist[:n_components], W[:, :n_components]


def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    Z = X.T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    V = pinv(R) @ Q.T @ X.T @ U  # V for Yf
    return U, V


def _scca_feature(X: ndarray, Yf: ndarray, n_components: int = 1):
    rhos = []
    for Y in Yf:
        U, V = _scca_kernel(X, Y)
        a = U[:, :n_components].T @ X
        b = V[:, :n_components].T @ Y
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        Yf: Optional[ndarray] = None,
    ):
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
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Yf) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)
        return labels


class FBSCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            SCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = np.argmax(features, axis=-1)
        return labels


def _itcca_feature(
    X: ndarray,
    templates: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
    method: str = "itcca1",
):
    rhos = []
    if method == "itcca1":
        for Xk in templates:
            U, V = _scca_kernel(X, Xk)
            a = U[:, :n_components].T @ X
            b = V[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    elif method == "itcca2":
        Us = cast(ndarray, Us)
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class ItCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        n_components: int = 1,
        method: str = "itcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        if self.method == "itcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )
        if self.method == "itcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = cast(ndarray, Yf)
            self.Us_, self.Vs_ = zip(
                *[
                    _scca_kernel(self.templates_[i], self.Yf_[i])
                    for i in range(len(self.classes_))
                ]
            )
            self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        method = self.method
        Us = None
        if method == "itcca2":
            Us = self.Us_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(_itcca_feature, Us=Us, n_components=n_components, method=method)
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBItCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "itcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ItCCA(n_components=n_components, method=method, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _mscca_feature(X: ndarray, templates: ndarray, U: ndarray, n_components: int = 1):
    rhos = []
    for Xk in zip(templates):
        a = U[:, :n_components].T @ X
        b = U[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class MsCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Note: MsCCA heavily depends on Yf, thus the phase information should be included when designs Yf.

    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.U_, self.V_ = _scca_kernel(
            np.concatenate(self.templates_, axis=-1), np.concatenate(self.Yf_, axis=-1)
        )
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        U = self.U_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_mscca_feature, U=U, n_components=n_components))(
                a, templates
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBMsCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ecca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U3 in zip(templates, Yf, Us):
        rho_list = []
        # 14a, 14d
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14b
        U2, _ = _scca_kernel(X, Xk)
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14c
        a = U3[:, :n_components].T @ X
        b = U3[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ecca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBECCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ECCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ttcca_template(X: ndarray, y: ndarray, y_sub: Optional[ndarray] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    labels = np.unique(y)
    if y_sub is None:
        templates = np.stack([np.mean(X[y == label], axis=0) for label in labels])
    else:
        subjects = np.unique(y_sub)
        templates = 0
        for sub_id in subjects:
            templates += np.stack(
                [
                    np.mean(X[(y == label) & (y_sub == sub_id)], axis=0)
                    for label in labels
                ]
            )
        templates /= len(subjects)
    return templates


def _ttcca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U2 in zip(templates, Yf, Us):
        rho_list = []
        # rho1
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho3
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho2
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class TtCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray, y_sub: Optional[ndarray] = None):

        self.classes_ = np.unique(y)
        self.templates_ = _ttcca_template(X, y, y_sub=y_sub)

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ttcca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBTtCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TtCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray,  # type: ignore[override]
            y: ndarray,
            Yf: Optional[ndarray] = None,
            y_sub: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf, y_sub=y_sub)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
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
    P = P @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


def _msetcca_kernel2(X: ndarray, Xk: ndarray, Yf: ndarray):
    C, N = X.shape
    M = 3
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = block_diag([X, Yf, Xk]).T
    D, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U[:C, :], D


def _msetcca_feature2(
    X: ndarray, templates: ndarray, Yf: ndarray, n_components: int = 1
):
    feat = []
    for Xk, Y in zip(templates, Yf):
        U, D = _msetcca_kernel2(X, Xk, Y)
        A = U[:, :n_components].T @ X
        B = U[:, :n_components].T @ Xk
        rho = np.array([pearsonr(a, b)[0] for a, b in zip(A, B)])
        rho = D[0] * np.sign(rho) * (rho**2)
        feat.append(rho)
    feat = np.concatenate(feat, axis=-1)
    return feat


class MsetCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        n_components: int = 1,
        method: str = "msetcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        if self.method == "msetcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        if self.method == "msetcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = Yf
            feat = self.transform(X)
            self.clf_ = SVC()
            self.clf_.fit(feat, y)
        elif self.method == "msetcca1":
            self.Us_, self.Ts_ = zip(
                *[_msetcca_kernel1(X[y == label]) for label in self.classes_]
            )
            self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        if self.method == "msetcca1":
            Ts = self.Ts_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_scca_feature, n_components=n_components))(a, Ts)
                for a in X
            )
            rhos = np.stack(rhos)
        elif self.method == "msetcca2":
            templates = self.templates_
            Yf = self.Yf_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_msetcca_feature2, n_components=n_components))(
                    a, templates, Yf
                )
                for a in X
            )
            rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        if self.method == "msetcca1":
            labels = self.classes_[np.argmax(feat, axis=-1)]
        elif self.method == "msetcca2":
            labels = self.clf_.predict(feat)
        return labels


class FBMsetCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "msetcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCA(n_components=n_components, method=method),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
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
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


class MsetCCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = 1):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_, self.Ts_ = zip(
            *[
                _msetccar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components

        Ts = self.Ts_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Ts) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBMsetCCAR(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCAR(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
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
    P = P @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


def _trca_feature(
    X: ndarray,
    templates: ndarray,
    Us: ndarray,
    n_components: int = 1,
    ensemble: bool = True,
):
    rhos = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        for Xk in templates:
            a = U.T @ X
            b = U.T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return rhos


class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        self.Us_ = np.stack([_trca_kernel(X[y == label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCA(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _trcar_kernel(X: ndarray, Yf: ndarray):
    """TRCA-R.
    X: (n_trials, n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


class TRCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_ = np.stack(
            [
                _trcar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCAR(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCAR(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
<<<<<<< HEAD
        return labels
=======
        return labels

def sign_sta(
    x: float) -> float:
    """Standardization of decision coefficient based on sign(x).

    Args:
        x (float)

    Returns:
        y (float): y=sign(x)*x^2
    """
    x = np.real(x)
    return (abs(x)/x)*(x**2)

def combine_feature(
    features: List[ndarray],
    func: Optional[Any] = sign_sta) -> ndarray:
    """Coefficient-level integration.

    Args:
        features (List[float or int or ndarray]): Different features.
        func (function): Quantization function.

    Returns:
        coef (the same type with elements of features): Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef

def combine_fb_feature(
    features: List[ndarray]) -> float:
    """Coefficient-level integration specially for filter-bank design.

    Args:
        features (List[ndarray]): Coefficient matrices of different sub-bands.

    Returns:
        coef (float): Integrated coefficients.

    """
    coef = np.zeros_like(features[0])
    for nf,feature in enumerate(features):
        coef += (pow(nf+1, -1.25) + 0.25) * (feature**2)
    return coef

def pick_subspace(
    descend_order: List[Tuple[int,float]],
    e_val_sum: float,
    ratio: float) -> int:
    """Config the number of subspaces.

    Args:
        descend_order (List[Tuple[int,float]]): See it in solve_gep() or solve_ep().
        e_val_sum (float): Trace of covariance matrix.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.

    Returns:
        n_components (int): The number of subspaces.
    """
    temp_val_sum = 0
    for n_components,do in enumerate(descend_order):  # n_sp: n_subspace
        temp_val_sum += do[-1]
        if temp_val_sum > ratio*e_val_sum:
            return n_components+1

def solve_gep(
    A: ndarray,
    B: ndarray,
    n_components: Optional[int] = None,
    ratio: Optional[float] = None,
    mode: Optional[str] = 'Max') -> ndarray:
    """Solve generalized problems | generalized Rayleigh quotient:
        f(w)=wAw^T/(wBw^T) -> Aw = lambda Bw -> B^{-1}Aw = lambda w

    Args:
        A (ndarray): (m,m).
        B (ndarray): (m,m).
        n_components (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    e_val_sum = np.sum(e_val)
    descend_order = sorted(enumerate(e_val), key=lambda x:x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if not n_components:
        n_components = pick_subspace(descend_order, e_val_sum, ratio)
    if mode == 'Min':
        return np.real(e_vec[:,w_index][:,n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:,w_index][:,:n_components].T)

def pearson_corr(
        X: ndarray,
        Y: ndarray) -> float:
    """Pearson correlation coefficient (1-D or 2-D).

    Args:
        X (ndarray): (..., n_points)
        Y (ndarray): (..., n_points). The dimension must be same with X.

    Returns:
        corrcoef (float)
    """
    # check if not zero_mean():
    # X,Y = zero_mean(X), zero_mean(Y)
    cov_xy = np.sum(X * Y)
    var_x = np.sum(X ** 2)
    var_y = np.sum(Y ** 2)
    corrcoef = cov_xy / sqrt(var_x * var_y)
    return corrcoef

# %% Basic TRCA object
class BasicTRCA(metaclass=ABCMeta):
    def __init__(self,
                 standard: Optional[bool] = True,
                 ensemble: Optional[bool] = True,
                 n_components: Optional[int] = 1,
                 ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def fit(self,
            X_train: ndarray,
            y_train: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass

    @abstractmethod
    def transform(self,
                  X_test: ndarray) -> Tuple[ndarray]:
        """Calculating decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
        """
        pass

    @abstractmethod
    def predict(self,
                X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        pass


class BasicFBTRCA(metaclass=ABCMeta):
    def __init__(self,
                 standard: Optional[bool] = True,
                 ensemble: Optional[bool] = True,
                 n_components: Optional[int] = 1,
                 ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def fit(self,
            X_train: ndarray,
            y_train: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass

    def transform(self,
                  X_test: ndarray) -> Tuple[ndarray]:
        """Using filter-bank algorithms to calculate decision coefficients.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
        """
        # apply model.predict() method in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_erou = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb] = fb_results[0]
            self.fb_erou[nb] = fb_results[2]

        # integration of multi-bands' results
        self.rou = combine_fb_feature(self.fb_rou)
        self.erou = combine_fb_feature(self.fb_erou)

        return self.rou, self.erou

    def predict(self,
                X_test: ndarray) -> Tuple[ndarray]:
        """Calculating the prediction labels based on the decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Nt*Nte,). Predict labels of sc-TRCA.
            y_ensemble (ndarray): (Nt*Nte,). Predict labels of sc-eTRCA.
        """
        # basic information
        n_test = X_test.shape[1]
        self.fb_y_standard = [[] for nb in range(self.n_bands)]
        self.fb_y_ensemble = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_y_standard[nb] = fb_results[1]
            self.fb_y_ensemble[nb] = fb_results[3]

        # integration of multi-bands' results
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        self.rou, self.erou = self.transform(X_test)
        for nte in range(n_test):
            self.y_standard[nte] = np.argmax(self.rou[nte, :])
            self.y_ensemble[nte] = np.argmax(self.erou[nte, :])
        return self.y_standard, self.y_ensemble


def sctrca_compute(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict[str, Any]:
    """(Ensemble) similarity-constrained TRCA (sc-(e)TRCA).

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: sc-(e)TRCA model (dict).
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average template.
        S (ndarray): (Ne,Nc,Nc). Covariance of template.
        u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for EEG signal.
        v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal signal.
        u_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for EEG signal.
        v_concat (ndarray): (Ne*Nk,2*Nh). Concatenated filter for sinusoidal signal.
        uX (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for EEG signal.
        vY (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for sinusoidal signal.
        euX (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for EEG signal.
        evY (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for sinusoidal signal.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_2harmonics = sine_template.shape[1]  # 2*Nh

    S = np.zeros((n_events, n_chans + n_2harmonics, n_chans + n_2harmonics))  # (Ne,Nc+2Nh,Nc+2Nh)
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        avg_template[ne] = np.mean(X_temp, axis=0)  # (Nc,Np)

        YY = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        XX = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for tt in range(train_trials):
            XX += X_temp[tt] @ X_temp[tt].T
        XmXm = avg_template[ne] @ avg_template[ne].T  # (Nc,Nc)
        XmY = avg_template[ne] @ sine_template[ne].T  # (Nc,2Nh)

        # block covariance matrix S: [[S11,S12],[S21,S22]]
        S[ne, :n_chans, :n_chans] = XmXm  # S11
        S[ne, :n_chans, n_chans:] = (1 - 1 / train_trials) * XmY  # S12
        S[ne, n_chans:, :n_chans] = S[ne, :n_chans, n_chans:].T  # S21
        S[ne, n_chans:, n_chans:] = YY  # S22

        # block covariance matrix Q: blkdiag(Q1,Q2)
        for ntr in range(n_train[ne]):
            Q[ne, :n_chans, :n_chans] += X_temp[ntr] @ X_temp[ntr].T  # Q1
        Q[ne, n_chans:, n_chans:] = train_trials * YY  # Q2

    # GEP | train spatial filters
    u, v, ndim, correct = [], [], [], [False for ne in range(n_events)]
    for ne in range(n_events):
        spatial_filter = solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        u.append(spatial_filter[:, :n_chans])  # (Nk,Nc)
        v.append(spatial_filter[:, n_chans:])  # (Nk,2Nh)
    u_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
    start_idx = 0
    for ne, dims in enumerate(ndim):
        u_concat[start_idx:start_idx + dims] = u[ne]
        v_concat[start_idx:start_idx + dims] = v[ne]
        start_idx += dims

    # signal templates
    uX, vY = [], []  # Ne*(Nk,Np)
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ avg_template[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ avg_template[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)

    # sc-(e)TRCA model
    model = {
        'Q': Q, 'S': S,
        'u': u, 'v': v, 'u_concat': u_concat, 'v_concat': v_concat,
        'uX': uX, 'vY': vY, 'euX': euX, 'evY': evY, 'correct': correct
    }
    return model


# %% similarity constrained (e)TRCA | sc-(e)TRCA
class SC_TRCA(BasicTRCA):
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et) for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble
        }

        # train sc-TRCA models & templates
        model = sctrca_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = model['Q'], model['S']
        self.u, self.v = model['u'], model['v']
        self.u_concat, self.v_concat = model['u_concat'], model['v_concat']
        self.uX, self.vY = model['uX'], model['vY']
        self.euX, self.evY = model['euX'], model['evY']
        self.correct = model['correct']
        return self

    def transform(self,
                  X_test: ndarray) -> Tuple[ndarray]:
        """Using sc-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-TRCA.
                Not empty when self.standard is True.
            erou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-eTRCA.
                Not empty when self.ensemble is True.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching (2-step)
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou)
        self.rou_sin = np.zeros_like(self.rou)
        self.erou = np.zeros_like(self.rou)
        self.erou_eeg = np.zeros_like(self.rou)
        self.erou_sin = np.zeros_like(self.rou)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_standard = self.u[nem] @ X_test[nte]
                    self.rou_eeg[nte, nem] = pearson_corr(
                        X=temp_standard,
                        Y=self.uX[nem]
                    )
                    self.rou_sin[nte, nem] = pearson_corr(
                        X=temp_standard,
                        Y=self.vY[nem]
                    )
                    self.rou[nte, nem] = combine_feature([
                        self.rou_eeg[nte, nem],
                        self.rou_sin[nte, nem]
                    ])

        if self.ensemble:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_ensemble = self.u_concat @ X_test[nte]
                    self.erou_eeg[nte, nem] = pearson_corr(
                        X=temp_ensemble,
                        Y=self.euX[nem]
                    )
                    self.erou_sin[nte, nem] = pearson_corr(
                        X=temp_ensemble,
                        Y=self.evY[nem]
                    )
                    self.erou[nte, nem] = combine_feature([
                        self.erou_eeg[nte, nem],
                        self.erou_sin[nte, nem]
                    ])

        return self.rou, self.erou

    def predict(self,
                X_test: ndarray) -> Tuple[ndarray]:
        """Calculating the prediction labels based on the decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Nt*Nte,). Predict labels of sc-TRCA.
            y_ensemble (ndarray): (Nt*Nte,). Predict labels of sc-eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        event_type = self.train_info['event_type']
        self.rou, self.erou = self.transform(X_test)
        if self.standard:
            for nte in range(n_test):
                self.y_standard[nte] = event_type[np.argmax(self.rou[nte, :])]
        if self.ensemble:
            for nte in range(n_test):
                self.y_ensemble[nte] = event_type[np.argmax(self.erou[nte, :])]

        return self.y_standard, self.y_ensemble


class FB_SC_TRCA(BasicFBTRCA):
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train filter-bank sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = X_train.shape[0]

        # train sc-TRCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template
            )
        return self
>>>>>>> b292979bf48d01f6d88e086f96b2b071c1f810f3
