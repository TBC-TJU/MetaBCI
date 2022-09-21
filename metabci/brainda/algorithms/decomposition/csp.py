# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
"""
Common Spatial Patterns and his happy little buddies!

"""
from copy import deepcopy
from typing import Union, Optional, List, Dict, Tuple
from functools import partial

import numpy as np
from numpy import ndarray
from scipy.linalg import eigh, pinv, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ShuffleSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import make_pipeline

from .base import robust_pattern, FilterBank
from ..utils.covariance import nearestPD, covariances

def csp_kernel(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """The kernel in CSP algorithm based on paper [1]_.

    Parameters
    ----------
    X: ndarray
        eeg data, shape (n_trials, n_channels, n_samples).
    y: ndarray
        labels of X, shape (n_trials,).

    Returns
    -------
    W: ndarray
        Spatial filters, shape (n_channels, n_filters).
    D: ndarray
        Eigenvalues of spatial filters, shape (n_filters,).
    A: ndarray
        Spatial patterns, shape (n_channels, n_patterns).

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = X - np.mean(X, axis=-1, keepdims=True)
    if len(labels) != 2:
        raise ValueError("the current kernel is for 2-class problem.")
    
    C1 = covariances(X[y==labels[0]])
    C2 = covariances(X[y==labels[1]])

    # # trace normalization
    # # this operation equals to trial normalization
    # C1 = C1 / np.trace(C1, axis1=-1, axis2=-2)[:, np.newaxis, np.newaxis]
    # C2 = C2 / np.trace(C2, axis1=-1, axis2=-2)[:, np.newaxis, np.newaxis]

    C1 = np.mean(C1, axis=0)
    C2 = np.mean(C2, axis=0)
    Cc = C1 + C2
    # check positive-definiteness
    Cc = nearestPD(Cc)
    # generalized eigenvalue problem
    D, W = eigh(C1, Cc)
    ix = np.argsort(D)[::-1]
    W = W[:, ix]
    D = D[ix]

    A = robust_pattern(W, C1, W.T@C1@W)

    return W, D, A

def csp_feature(W: ndarray, X: ndarray,
        n_components: int = 2) -> ndarray:
    """Return CSP features in paper [1]_.

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    n_components : int, optional
        the first k components to use, usually even number, by default 2

    Returns
    -------
    ndarray
        features of shape (n_trials, n_features)

    Raises
    ------
    ValueError
        n_components should less than the number of channels

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    W, X = np.copy(W), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    
    eps = np.finfo(X.dtype).eps
    X = X - np.mean(X, axis=-1, keepdims=True)
    # normalized variance
    features = np.mean(np.square(np.matmul(W[:, :n_components].T, X)), axis=-1)
    features = features / (np.sum(features, axis=-1, keepdims=True) + eps)
    # log-transformation
    features = np.log(np.clip(features, eps, None))
    return features

def _rjd(X, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization based on jacobi angle.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    eps : float, optional
        Tolerance for stopping criterion (default 1e-8).
    n_iter_max : int, optional
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_channels, n_filters), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    This is a direct implementation of the Cardoso AJD algorithm [1]_ used in
    JADE. The code is a translation of the matlab code provided in the author
    website.

    References
    ----------
    .. [1] Cardoso, Jean-Francois, and Antoine Souloumiac. Jacobi angles for simultaneous diagonalization. SIAM journal on matrix analysis and applications 17.1 (1996): 161-164.

    """

    # reshape input matrix
    A = np.concatenate(X, 0).T

    # init variables
    m, nm = A.shape
    V = np.eye(m)
    encore = True
    k = 0

    while encore:
        encore = False
        k += 1
        if k > n_iter_max:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = np.arange(p, nm, m)
                Iq = np.arange(q, nm, m)

                # computation of Givens angle
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton +
                                         np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                encore = encore | (np.abs(s) > eps)
                if (np.abs(s) > eps):
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

    D = np.reshape(A, (m, int(nm / m), m)).transpose(1, 0, 2)
    return V, D

def _ajd_pham(X, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization based on pham's algorithm.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    eps : float, optional 
        Tolerance for stoping criterion (default 1e-6).
    n_iter_max : int, optional
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_channels, n_filters), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    This is a direct implementation of the PHAM's AJD algorithm [1]_.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive definite Hermitian matrices." SIAM Journal on Matrix Analysis and Applications 22, no. 4 (2001): 1136-1152.

    """
     # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V.T, D

def _uwedge(X, init=None, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization algorithm UWEDGE.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    init : None | ndarray, optional
        Initialization for the diagonalizer, shape (n_channels, n_channels).
    eps : float, optional
        Tolerance for stoping criterion (default 1e-7).
    n_iter_max : int
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    W_est : ndarray
        The diagonalizer, shape (n_filters, n_channels), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    Uniformly Weighted Exhaustive Diagonalization using Gauss iteration
    (U-WEDGE). Implementation of the AJD algorithm by Tichavsky and Yeredor [1]_ [2]_.
    This is a translation from the matlab code provided by the authors.

    References
    ----------
    .. [1] P. Tichavsky, A. Yeredor and J. Nielsen, "A Fast Approximate Joint Diagonalization Algorithm Using a Criterion with a Block Diagonal Weight Matrix", ICASSP 2008, Las Vegas.
    .. [2] P. Tichavsky and A. Yeredor, "Fast Approximate Joint Diagonalization Incorporating Weight Matrices" IEEE Transactions of Signal Processing, 2009.
    
    """
    L, d, _ = X.shape

    # reshape input matrix
    M = np.concatenate(X, 0).T

    # init variables
    d, Md = M.shape
    iteration = 0
    improve = 10

    if init is None:
        E, H = np.linalg.eig(M[:, 0:d])
        W_est = np.dot(np.diag(1. / np.sqrt(np.abs(E))), H.T)
    else:
        W_est = init

    Ms = np.array(M)
    Rs = np.zeros((d, L))

    for k in range(L):
        ini = k*d
        Il = np.arange(ini, ini + d)
        M[:, Il] = 0.5*(M[:, Il] + M[:, Il].T)
        Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
        Rs[:, k] = np.diag(Ms[:, Il])

    crit = np.sum(Ms**2) - np.sum(Rs**2)
    while (improve > eps) & (iteration < n_iter_max):
        B = np.dot(Rs, Rs.T)
        C1 = np.zeros((d, d))
        for i in range(d):
            C1[:, i] = np.sum(Ms[:, i:Md:d]*Rs, axis=1)

        D0 = B*B.T - np.outer(np.diag(B), np.diag(B))
        A0 = (C1 * B - np.dot(np.diag(np.diag(B)), C1.T)) / (D0 + np.eye(d))
        A0 += np.eye(d)
        W_est = np.linalg.solve(A0, W_est)

        Raux = np.dot(np.dot(W_est, M[:, 0:d]), W_est.T)
        aux = 1./np.sqrt(np.abs(np.diag(Raux)))
        W_est = np.dot(np.diag(aux), W_est)

        for k in range(L):
            ini = k*d
            Il = np.arange(ini, ini + d)
            Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
            Rs[:, k] = np.diag(Ms[:, Il])

        crit_new = np.sum(Ms**2) - np.sum(Rs**2)
        improve = np.abs(crit_new - crit)
        crit = crit_new
        iteration += 1

    D = np.reshape(Ms, (d, L, d)).transpose(1, 0, 2)
    return W_est.T, D

ajd_methods = {
    'rjd': _rjd, 
    'ajd_pham': _ajd_pham, 
    'uwedge': _uwedge
}

def _check_ajd_method(method):
    """Check if a given method is valid.

    Parameters
    ----------
    method : callable object or str
        Could be the name of ajd_method or a callable method itself.

    Returns
    -------
    method: callable object
        A callable ajd method.
    """
    if callable(method):
        pass
    elif method in ajd_methods.keys():
        method = ajd_methods[method]
    else:
        raise ValueError(
            """%s is not an valid method ! Valid methods are : %s or a
             callable function""" % (method, (' , ').join(ajd_methods.keys())))
    return method

def ajd(X: ndarray, method: str ='uwedge') -> Tuple[ndarray, ndarray]:
    """Wrapper of AJD methods.
    
    Parameters
    ----------
    X : ndarray
        Input covariance matrices, shape (n_trials, n_channels, n_channels)
    method : str, optional
        AJD method (default uwedge).
    
    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_channels, n_filters), usually n_filters == n_channels.
    D : ndarray
        The mean of quasi diagonal matrices, shape (n_channels,).
    """
    method = _check_ajd_method(method)
    V, D = method(X)
    D = np.diag(np.mean(D, axis=0))
    ind = np.argsort(D)[::-1]
    D = D[ind]
    V = V[:, ind]
    return V, D

def gw_csp_kernel(X: ndarray, y: ndarray,
        ajd_method: str = 'uwedge') -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Grosse-Wentrup AJD method based on paper [1]_.

    Parameters
    ----------
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples).
    y : ndarray
        labels, shape (n_trials).
    ajd_method : str, optional
        ajd methods, 'uwedge' 'rjd' and 'ajd_pham', by default 'uwedge'.

    Returns
    -------
    W: ndarray
        Spatial filters, shape (n_channels, n_filters).
    D: ndarray
        Eigenvalues of spatial filters, shape (n_filters,).
    A: ndarray
        Spatial patterns, shape (n_channels, n_patterns).
    mutual_info: ndarray
        Mutual informaiton values, shape (n_filters).

    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial patterns and information theoretic feature extraction." Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = X - np.mean(X, axis=-1, keepdims=True)

    Cx = []
    for label in labels:
        C = covariances(X[y==label])
        # trace normalization
        C = C / np.trace(C, axis1=-1, axis2=-2)[:, np.newaxis, np.newaxis]
        Cx.append(np.mean(C, axis=0))
    Cx = np.stack(Cx)
    W, D = ajd(Cx, method=ajd_method)
    # Ctot = np.mean(Cx, axis=0)
    # W = W / np.sqrt(np.diag(W.T@Ctot@W))
    W = W / np.sqrt(D)

    # compute mutual information values
    Pc = [np.mean(y == label) for label in labels]
    mutual_info = []
    for j in range(W.shape[-1]):
        a = 0
        b = 0
        for i in range(len(labels)):
            # tmp = np.dot(np.dot(W[j], self.C_[i]), W[j].T)
            tmp = W[:, j].T@Cx[i]@W[:, j]
            a += Pc[i] * np.log(np.sqrt(tmp))
            b += Pc[i] * (tmp ** 2 - 1)
        mi = - (a + (3.0 / 16) * (b ** 2))
        mutual_info.append(mi)
    mutual_info = np.array(mutual_info)
    ix = np.argsort(mutual_info)[::-1]
    W = W[:, ix]
    mutual_info = mutual_info[ix]
    D = D[ix]
    A = robust_pattern(W, Cx[0], W.T@Cx[0]@W)
    return W, D, A, mutual_info

class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Pattern.

    if n_components is None, auto finding the best number of components with gridsearch. The upper searching limit is determined by max_components, default is half of the number of channels.
    """
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None):
        self.n_components = n_components
        self.max_components = max_components

    def fit(self, X: ndarray, y: ndarray):
        self.classes_ = np.unique(y)
        self.W_, self.D_, self.A_ = csp_kernel(X, y)
        # resorting with 0.5 threshold
        self.D_ = np.abs(self.D_ - 0.5)
        ind = np.argsort(self.D_, axis=-1)[::-1]
        self.W_, self.D_, self.A_ = self.W_[:, ind], self.D_[ind], self.A_[:, ind]

        # auto-tuning
        if self.n_components is None:
            estimator = make_pipeline(*[CSP(n_components=self.n_components), SVC()])
            if self.max_components is None:
                params = {'csp__n_components': np.arange(1, self.W_.shape[1]+1)}
            else:
                params = {'csp__n_components': np.arange(1, self.max_components+1)}
            
            n_splits = np.min(np.unique(y, return_counts=True)[1])
            n_splits = 5 if n_splits > 5 else n_splits
            
            gs = GridSearchCV(estimator,
                param_grid=params, scoring='accuracy', 
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True), refit=False, n_jobs=-1, verbose=False)
            gs.fit(X, y)
            self.best_n_components_ = gs.best_params_['csp__n_components']
        return self

    def transform(self, X: ndarray):
        n_components = self.best_n_components_ if self.n_components is None else self.n_components
        return csp_feature(self.W_, X, n_components=n_components)

class MultiCSP(BaseEstimator, TransformerMixin):
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None,
            multiclass: str = 'ovr', ajd_method: str ='uwedge'):
        self.n_components = n_components
        self.max_components = max_components
        self.multiclass = multiclass
        self.ajd_method = ajd_method

    def fit(self, X: ndarray, y: ndarray):
        self.classes_ = np.unique(y)

        if self.multiclass == 'ovr':
            self.estimator_ = OneVsRestClassifier(
                make_pipeline(*[
                CSP(n_components=self.n_components, max_components=self.max_components), SVC()
                ]), n_jobs=-1)
            self.estimator_.fit(X, y)

        elif self.multiclass == 'ovo':
            self.estimator_ = OneVsOneClassifier(
                make_pipeline(*[
                CSP(n_components=self.n_components, max_components=self.max_components), SVC()
                ]), n_jobs=-1)
            # patching avoiding 2d array check
            self.estimator_._validate_data = partial(self.estimator_._validate_data, allow_nd=True)
            self.estimator_.fit(X, y)

        elif self.multiclass == 'grosse-wentrup':
            self.W_, _, self.A_, self.mutualinfo_values_ = gw_csp_kernel(
                X, y, ajd_method=self.ajd_method)
            if self.n_components is None:
                estimator = make_pipeline(*[
                    MultiCSP(n_components=self.n_components, multiclass='grosse-wentrup', ajd_method=self.ajd_method), SVC()
                ])
                if self.max_components is None:
                    params = {'multicsp__n_components': np.arange(1, self.W_.shape[1]+1)}
                else:
                    params = {'multicsp__n_components': np.arange(1, self.max_components+1)}

                n_splits = np.min(np.unique(y, return_counts=True)[1])
                n_splits = 5 if n_splits > 5 else n_splits
                gs = GridSearchCV(estimator,
                    param_grid=params, scoring='accuracy', 
                    cv=StratifiedKFold(n_splits=n_splits, shuffle=True), refit=False, n_jobs=-1, verbose=False)
                gs.fit(X, y)
                self.best_n_components_ = gs.best_params_['multicsp__n_components']
        else:
            raise ValueError("not a valid multiclass strategy")
        return self

    def transform(self, X: ndarray):
        if self.multiclass == 'grosse-wentrup':
            n_components = self.best_n_components_ if self.n_components is None else self.n_components
            features = csp_feature(self.W_, X, n_components=n_components)
        else:
            features = np.concatenate([est[0].transform(X) for est in self.estimator_.estimators_], axis=-1)
        return features

def spoc_kernel(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Source Power Comodulation (SPoC) based on paper [1]_.

    It is a continous CSP-like method.

    Parameters
    ----------
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials)

    Returns
    -------
    W: ndarray
        Spatial filters, shape (n_channels, n_filters).
    D: ndarray
        Eigenvalues of spatial filters, shape (n_filters,).
    A: ndarray
        Spatial patterns, shape (n_channels, n_patterns).

    References
    ----------
    .. [1] Sven Dähne, Frank C. Meinecke, Stefan Haufe, Johannes Höhne, Michael Tangermann, Klaus-Robert Müller, and Vadim V. Nikulin. SPoC: a novel framework for relating the amplitude of neuronal oscillations to behaviorally relevant parameters. NeuroImage, 86:111–122, 2014. doi:10.1016/j.neuroimage.2013.07.079.
    """
    X, weights = np.copy(X), np.copy(y)
    eps = np.finfo(X.dtype).eps
    X = X - np.mean(X, axis=-1, keepdims=True)
    weights = weights - np.mean(weights)
    weights = weights / np.std(weights)
    Cx = covariances(X)
    # trace normalization
    Cx = Cx / np.trace(Cx, axis1=-1, axis2=-2)[:, np.newaxis, np.newaxis]
    C = np.mean(Cx, axis=0)
    Cz = np.mean(weights[:, np.newaxis, np.newaxis]*Cx, axis=0)
    
    # check positive-definiteness
    C = nearestPD(C)
    Cz = nearestPD(Cz)

    # TODO: direct copy from pyriemann, need verify
    D, W = eigh(Cz, C)
    ind = np.argsort(D)[::-1]
    D = D[ind]
    W = W[:, ind]

    A = robust_pattern(W, Cz, W.T@Cz@W)
    return W, D, A

class SPoC(BaseEstimator, TransformerMixin):
    """Source Power Comodulation (SPoC).

    For continuous data, not verified.

    """
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None):
        self.n_components = n_components
        self.max_components = max_components

    def fit(self, X: ndarray, y: ndarray):
        self.W_, self.D_, self.A_ = spoc_kernel(X, y)

        # auto-tuning
        if self.n_components is None:
            estimator = make_pipeline(*[SPoC(n_components=self.n_components), Ridge(alpha=0.5)])
            if self.max_components is None:
                params = {'spoc__n_components': np.arange(1, self.W_.shape[1]+1)}
            else:
                params = {'spoc__n_components': np.arange(1, self.max_components+1)}

            test_size = 0.2 if len(y) > 5 else 1/len(y)

            gs = GridSearchCV(estimator,
                param_grid=params, scoring='neg_root_mean_squared_error', 
                cv=ShuffleSplit(n_splits=5, test_size=test_size), refit=False, n_jobs=-1, verbose=False)
            gs.fit(X, y)
            self.best_n_components_ = gs.best_params_['spoc__n_components']

    def transform(self, X: ndarray):
        n_components = self.best_n_components_ if self.n_components is None else self.n_components
        return csp_feature(self.W_, X, n_components=n_components)

class FBCSP(FilterBank):
    """FBCSP.

    FilterBank CSP based on paper [1]_.

    References
    ----------
    .. [1] Ang K K, Chin Z Y, Zhang H, et al. Filter bank common spatial pattern (FBCSP) in brain-computer interface[C]//2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008: 2390-2397.
    """
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None,
            n_mutualinfo_components: Optional[int] = None,
            filterbank: Optional[List[ndarray]] = None):
        self.n_components = n_components
        self.max_components = max_components
        self.n_mutualinfo_components = n_mutualinfo_components
        self.filterbank = filterbank
        super().__init__(CSP(n_components=n_components, max_components=max_components), filterbank=filterbank)

    def fit(self, X: ndarray, y: ndarray):
        super().fit(X, y)
        features = super().transform(X)
        if self.n_mutualinfo_components is None:
            estimator = make_pipeline(*[
                SelectKBest(score_func=mutual_info_classif, k='all'), 
                SVC()
            ])
            params = {'selectkbest__k': np.arange(1, features.shape[1]+1)}
            
            n_splits = np.min(np.unique(y, return_counts=True)[1])
            n_splits = 5 if n_splits > 5 else n_splits

            gs = GridSearchCV(estimator,
                param_grid=params, scoring='accuracy', 
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True), refit=False, n_jobs=-1, verbose=False)
            gs.fit(features, y)
            self.best_n_mutualinfo_components_ = gs.best_params_['selectkbest__k']
            self.selector_ = SelectKBest(
                score_func=mutual_info_classif, k=self.best_n_mutualinfo_components_)
        else:
            self.selector_ = SelectKBest(
                score_func=mutual_info_classif, k=self.n_mutualinfo_components)
        self.selector_.fit(features, y)
        return self

    def transform(self, X: ndarray):
        features = super().transform(X)
        features = self.selector_.transform(features)
        return features

class FBMultiCSP(FilterBank):
    def __init__(self,
            n_components: Optional[int] = None,
            max_components: Optional[int] = None,
            multiclass: str = 'ovr', ajd_method: str ='uwedge',
            n_mutualinfo_components: Optional[int] = None,
            filterbank: Optional[List[ndarray]] = None):
        self.n_components = n_components
        self.max_components = max_components
        self.multiclass = multiclass
        self.ajd_method = ajd_method
        self.n_mutualinfo_components = n_mutualinfo_components
        self.filterbank = filterbank
        self.n_mutualinfo_components = n_mutualinfo_components
        super().__init__(MultiCSP(n_components=n_components, max_components=max_components, multiclass=multiclass, ajd_method=ajd_method),filterbank=filterbank)

    def fit(self, X: ndarray, y: ndarray):
        super().fit(X, y)
        features = super().transform(X)
        if self.n_mutualinfo_components is None:
            estimator = make_pipeline(*[
                SelectKBest(score_func=mutual_info_classif, k='all'), 
                SVC()
            ])
            params = {'selectkbest__k': np.arange(1, features.shape[1]+1)}

            n_splits = np.min(np.unique(y, return_counts=True)[1])
            n_splits = 5 if n_splits > 5 else n_splits

            gs = GridSearchCV(estimator,
                param_grid=params, scoring='accuracy', 
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True), refit=False, n_jobs=-1, verbose=False)
            gs.fit(features, y)
            self.best_n_mutualinfo_components_ = gs.best_params_['selectkbest__k']
            self.selector_ = SelectKBest(
                score_func=mutual_info_classif, k=self.best_n_mutualinfo_components_)
        else:
            self.selector_ = SelectKBest(
                score_func=mutual_info_classif, k=self.n_mutualinfo_components)
        self.selector_.fit(features, y)
        return self

    def transform(self, X: ndarray):
        features = super().transform(X)
        features = self.selector_.transform(features)
        return features

