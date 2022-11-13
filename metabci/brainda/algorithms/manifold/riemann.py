# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
"""
Riemannian Geometry for BCI.
"""
from typing import Union, List, Tuple, Dict, Optional, Callable
import numpy as np
from numpy import ndarray
from numpy.lib.function_base import cov
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.linalg import eigvalsh, inv, eigh, pinv

from ..utils.covariance import (nearestPD, covariances, sqrtm, invsqrtm, logm, expm, powm)

def logmap(Pi: ndarray, P: ndarray, n_jobs: Optional[int] = None):
    """Logarithm map from the positive-definite space to the tangent space.

    Logarithm map projects :math:`\mathbf{P}_i \in \mathcal{M}` to the tangent space point 
    :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` at :math:`\mathbf{P} \in \mathcal{M}`.
    
    Parameters
    ----------
    Pi : ndarray
        SPD matrix.
    P : ndarray
        Reference point.
    n_jobs: int, optional
        the number of jobs to use.

    Returns
    -------
    Si : ndarray
        Tangent space point (in matrix form).
    """
    P12 = sqrtm(P, n_jobs=n_jobs)
    iP12 = invsqrtm(P, n_jobs=n_jobs)
    wPi = np.matmul(np.matmul(iP12, Pi), iP12)
    Si = np.matmul(np.matmul(P12, logm(wPi, n_jobs=n_jobs)), P12)
    return Si

def expmap(Si: ndarray, P: ndarray, n_jobs: Optional[int] = None):
    """Exponential map from the tangent space to the positive-definite space.

    Exponential map projects :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` bach to the manifold
    :math:`\mathcal{M}`.
    
    Parameters
    ----------
    Si : ndarray
        Tangent space point (in matrix form).       
    P : ndarray
        Reference point.
    n_jobs: int, optional
        the number of jobs to use.

    Returns
    -------
    Pi : ndarray
        SPD matrix.
    """
    P12 = sqrtm(P, n_jobs=n_jobs)
    iP12 = invsqrtm(P, n_jobs=n_jobs)
    wSi = np.matmul(np.matmul(iP12, Si), iP12)
    Pi = np.matmul(np.matmul(P12, expm(wSi, n_jobs=n_jobs)), P12)
    return Pi

def geodesic(P1: ndarray, P2: ndarray, t: float, n_jobs: Optional[int] = None):
    """Geodesic.
    
    The geodesic curve between any two SPD matrices :math:`\mathbf{P}_1,\mathbf{P}_2 \in \mathcal{M}`.

    Parameters
    ----------
    P1 : ndarray
        SPD matrix.
    P2 : ndarray
        SPD matrix, the same shape of P1.
    t : float
        :math:`0 \leq t \leq 1`.
    n_jobs: int, optional
        the number of jobs to use.
    
    Returns
    -------
    phi : ndarray
        SPD matrix on the geodesic curve between P1 and P2.
    """
    p1_shape = P1.shape
    p2_shape = P2.shape
    P1 = P1.reshape((-1, *p1_shape[-2:]))
    P2 = P2.reshape((-1, *p2_shape[-2:]))
    P12 = sqrtm(P1, n_jobs=n_jobs)
    iP12 = invsqrtm(P1, n_jobs=n_jobs)
    wP2 = np.matmul(np.matmul(iP12, P2), iP12)
    phi = np.matmul(np.matmul(P12, powm(wP2, t, n_jobs=n_jobs)), P12)
    return phi

def distance_riemann(A: ndarray, B: ndarray, n_jobs: Optional[int] = None):
    """Riemannian distance between two covariance matrices A and B.

    Parameters
    ----------
    A : ndarray
        First positive-definite matrix, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    B : ndarray
        Second positive-definite matrix.

    Returns
    -------
    ndarray | float
        Riemannian distance between A and B.

    Notes
    -----
    .. math::
            d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.
    """
    def _single_distance_riemann(A, B):
        dist = np.sqrt(
           np.sum(np.log(eigvalsh(A, B))**2) 
        )
        return dist

    A = A.reshape((-1, *A.shape[-2:]))
    B = B.reshape((-1, *B.shape[-2:]))
    
    if A.shape[0] == 1:
        A = np.broadcast_to(A, B.shape)
    elif B.shape[0] == 1:
        B = np.broadcast_to(B, A.shape)
    
    d = Parallel(n_jobs=n_jobs)(delayed(_single_distance_riemann)(a, b) for a, b in zip(A, B))
    d = np.array(d)
    return d

def _get_sample_weight(sample_weight, N):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = np.ones(N)
    if len(sample_weight) != N:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight

def mean_riemann(covmats, tol=1e-11, maxiter=300, init=None, sample_weight=None, n_jobs=None):
    """Return the mean covariance matrix according to the Riemannian metric.

    Parameters
    ----------
    covmats : ndarray
        Covariance matrices set, shape (n_trials, n_channels, n_channels).
    tol : float, optional
        The tolerance to stop the gradient descent (default 1e-8).
    maxiter : int, optional
        The maximum number of iteration (default 50).
    init : None|ndarray, optional
        A covariance matrix used to initialize the gradient descent (default None), if None the arithmetic mean is used.
    sample_weight : None|ndarray, optional
        The weight of each sample (efault None), if None weights are 1 otherwise weights are normalized.

    Returns
    -------
    C : ndarray
        The Riemannian mean covariance matrix.
    
    Notes
    -----
    The procedure is similar to a gradient descent minimizing the sum of riemannian distance to the mean.

    .. math::
        \mathbf{C} = \\arg \min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    where :math:\delta_R is riemann distance.
    """
    # init
    sample_weight = _get_sample_weight(sample_weight, len(covmats))
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C, n_jobs=1)
        iC12 = invsqrtm(C, n_jobs=1)

        J = logm(np.matmul(np.matmul(iC12, covmats), iC12), n_jobs=n_jobs)
        J = np.sum(sample_weight[:, np.newaxis, np.newaxis]*J, axis=0)
        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit

        C = C12@expm(nu*J, n_jobs=1)@C12
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
    return C

def vectorize(Si: ndarray):
    """vectorize tangent space points.

    Parameters
    ----------
    Si : ndarray
        points in the tangent space, shape (n_trials, n_channels, n_channels)

    Returns
    -------
    ndarray
        vectorized version of Si, shape (n_trials, n_channels*(n_channels+1)/2)
    """
    Si = Si.reshape((-1, *Si.shape[-2:]))
    n_channels = Si.shape[-1]
    ind = np.triu_indices(n_channels, k=0)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + np.eye(n_channels))[ind]
    vSi = Si[:, ind[0], ind[1]]*coeffs
    return vSi

def unvectorize(vSi: ndarray):
    """unvectorize tangent space points.

    Parameters
    ----------
    vSi : ndarray
        vectorized version of Si, shape (n_trials, n_channels*(n_channels+1)/2)

    Returns
    -------
    ndarray
        points in the tangent space, shape (n_trials, n_channels, n_channels)
    """
    n_trials, n_features = vSi.shape
    n_channels = int((np.sqrt(1 + 8 * n_features) - 1) / 2)
    ind = np.triu_indices(n_channels, k=0)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + 2*np.eye(n_channels))[ind]
    vSi = vSi / coeffs
    Si = np.zeros((n_trials, n_channels, n_channels))
    Si[:, ind[0], ind[1]] = vSi
    Si = Si + np.transpose(Si, (0, 2, 1))
    return Si

def tangent_space(Pi: ndarray, P: ndarray, n_jobs: Optional[int] = None):
    """Logarithm map projects SPD matrices to the tangent vectors.
    
    Parameters
    ----------
    Pi : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    P : ndarray
        Reference point.
    
    Returns
    -------
    vSi : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    """
    Si = logmap(Pi, P, n_jobs=n_jobs)
    vSi = vectorize(Si)
    return vSi

def untangent_space(vSi: ndarray, P: ndarray, n_jobs: Optional[int] = None):
    """Logarithm map projects SPD matrices to the tangent vectors.
    
    Parameters
    ----------
    vSi : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    P : ndarray
        Reference point.
    
    Returns
    -------
    Pi : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    """
    Si = unvectorize(vSi)
    Pi = expmap(Si, P, n_jobs=n_jobs)
    return Pi

def mdrm_kernel(X: ndarray, y: ndarray,
        sample_weight: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
    """Minimum Distance to Riemannian Mean.

    Parameters
    ----------
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials)
    sample_weight : Optional[ndarray], optional
        sample weights, by default None
    n_jobs : Optional[int], optional
        the number of jobs to use, by default None

    Returns
    -------
    ndarray
        centroids of each class, shape (n_class, n_channels, n_channels).
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    Cx = covariances(X, estimator='lwf', n_jobs=n_jobs)
    sample_weight = np.ones((len(X))) if sample_weight is None else sample_weight

    Centroids =Parallel(n_jobs=n_jobs)(
        delayed(mean_riemann)(Cx[y==label], sample_weight=sample_weight[y==label]) for label in labels)
    return np.stack(Centroids)

class FGDA(BaseEstimator, TransformerMixin):
    """
    Fisher Geodesic Discriminat Analysis.
    """
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        Pi = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        self.lda_ = LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto')
        self.lda_.fit(vSi, y)
        W = self.lda_.coef_.copy()
        self.W_ = W.T@pinv(W@W.T)@W # n_feat by n_feat
        return self

    def transform(self, X):
        Pi = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        vSi = vSi@self.W_
        Pi = untangent_space(vSi, self.P_)
        return Pi

class MDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray,
            sample_weight: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        self.centroids_ = mdrm_kernel(X, y, sample_weight=sample_weight, n_jobs=self.n_jobs)
        return self

    def _transform_distance(self, X: ndarray):
        Cx = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        dist = np.stack([distance_riemann(Cx, centroid, n_jobs=self.n_jobs) for centroid in self.centroids_]).T
        return dist

    def transform(self, X: ndarray):
        return self._transform_distance(X)

    def predict(self, X: ndarray):
        dist = self._transform_distance(X)
        return self.classes_[np.argmin(dist, axis=1)]

    def predict_proba(self, X: ndarray):
        return softmax(-1*self._transform_distance(X))

class FgMDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray,
            sample_weight: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        self.fgda_ = FGDA(n_jobs=self.n_jobs)
        Cx = self.fgda_.fit_transform(X, y)
        sample_weight = np.ones((len(X))) if sample_weight is None else sample_weight
        Centroids =Parallel(n_jobs=self.n_jobs)(
            delayed(mean_riemann)(Cx[y==label], sample_weight=sample_weight[y==label]) for label in self.classes_)
        self.centroids_ = np.stack(Centroids)
        return self

    def _transform_distance(self, X: ndarray):
        Cx = self.fgda_.transform(X)
        dist = np.stack([distance_riemann(Cx, centroid, n_jobs=self.n_jobs) for centroid in self.centroids_]).T
        return dist

    def transform(self, X: ndarray):
        return self._transform_distance(X)

    def predict(self, X: ndarray):
        dist = self._transform_distance(X)
        return self.classes_[np.argmin(dist, axis=1)]

class TSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=LogisticRegression(), n_jobs=None):
        self.clf = clf
        self.n_jobs = n_jobs

        if not isinstance(self.clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')
    
    def fit(self, X: ndarray, y: ndarray):
        Pi = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        self.clf.fit(vSi, y)
        return self

    def predict(self, X: ndarray):
        Pi = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        return self.clf.predict(vSi)

    def predict_proba(self, X: ndarray):
        Pi = covariances(X, estimator='lwf', n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        return self.clf.predict_proba(vSi)        

class Alignment(BaseEstimator, TransformerMixin):
    """Riemannian/Euclidean Alignment.
    """
    def __init__(self,
            align_method: str = 'euclid',
            cov_method: str = 'lwf',
            n_jobs: Optional[int] = None):
        self.align_method = align_method
        self.cov_method = cov_method
        self.n_jobs = n_jobs
    
    def fit(self, X: ndarray, y: Optional[ndarray] = None):
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if self.align_method == 'euclid':
            self.iC12_ = self._euclid_center(X)
        elif self.align_method == 'riemann':
            self.iC12_ = self._riemann_center(X)
        else:
            raise ValueError("non-supported aligning method.")
        
        return self

    def transform(self, X):
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        X = np.matmul(self.iC12_, X)
        return X

    def _euclid_center(self, X):
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        C = np.mean(Cs, axis=0)
        return invsqrtm(C)
    
    def _riemann_center(self, X):
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        C = mean_riemann(Cs, n_jobs=self.n_jobs)
        return invsqrtm(C)     

class RecursiveAlignment(BaseEstimator, TransformerMixin):
    """Recursive Riemannian/Euclidean Alignment.
    """
    def __init__(self,
            align_method: str = 'euclid',
            cov_method: str = 'lwf',
            n_jobs: Optional[int] = None):
        self.align_method = align_method
        self.cov_method = cov_method
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        if not hasattr(self, 'iC12_'):
            self.iC12_ = np.eye(X.shape[1])
            self.C_ = np.eye(X.shape[1])
            self.n_tracked = 0
        X = self._recursive_fit_transform(X, Cs)
        return X

    def _recursive_fit_transform(self, X, Cs):
        for i in range(len(X)):
            if self.align_method == 'euclid':
                self._recursive_euclid_center(Cs[i])
            elif self.align_method == 'riemann':
                self._recursive_riemann_center(Cs[i])
            else:
                raise ValueError("non-supported aligning method.")
            if self.n_tracked == 1:
                X[i] = X[i]/np.std(X[i], axis=(-2, -1), keepdims=True)
            else:
                X[i] = self.iC12_@X[i]
        return X

    def _recursive_euclid_center(self, C):
        self.n_tracked += 1
        alpha = 1/(self.n_tracked)
        self.C_ = (1-alpha)*self.C_ + alpha*C
        self.iC12_ = invsqrtm(self.C_)

    def _recursive_riemann_center(self, C):
        self.n_tracked += 1
        alpha = 1/(self.n_tracked)
        self.C_ = geodesic(self.C_, C, alpha, n_jobs=1)
        self.iC12_ = invsqrtm(self.C_)

