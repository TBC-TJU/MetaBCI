# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
"""
Riemannian Procrustes Analysis.
Modified from https://github.com/plcrodrigues/RPA
"""
from typing import Union, List, Tuple, Dict, Optional, Callable
from functools import partial
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.extmath import softmax
from joblib import Parallel, delayed
from scipy.linalg import eigvalsh, inv, eigh

import autograd.numpy as anp
try:
    from pymanopt.manifolds import Rotations
except:
    from pymanopt.manifolds import SpecialOrthogonalGroup as Rotations
from pymanopt import Problem
try:
    from pymanopt.solvers import SteepestDescent
except:
    from pymanopt.optimizers import SteepestDescent
from ..utils.covariance import (nearestPD, covariances, sqrtm, invsqrtm, logm, expm, powm)
from .riemann import mean_riemann, distance_riemann

def get_recenter(X: ndarray,
        cov_method: str = 'cov',
        mean_method: str = 'riemann',
        n_jobs: Optional[int] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    C = covariances(X, estimator=cov_method, n_jobs=n_jobs)
    if mean_method == 'riemann':
        M = mean_riemann(C, n_jobs=n_jobs)
    elif mean_method == 'euclid':
        M = np.mean(C, axis=0)
    iM12 = invsqrtm(M)
    return iM12

def recenter(X: ndarray, iM12: ndarray):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    return iM12@X    

def get_rescale(X: ndarray,
        cov_method: str = 'cov',
        n_jobs: Optional[int] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    C = covariances(X, estimator=cov_method, n_jobs=n_jobs)
    M = mean_riemann(C, n_jobs=n_jobs)
    d = np.mean(np.square(distance_riemann(C, M, n_jobs=n_jobs)))
    scale = np.sqrt(1/d)
    return M, scale

def rescale(X: ndarray, M: ndarray, scale: float,
        cov_method: str = 'cov', 
        n_jobs: Optional[int] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    C = covariances(X, estimator=cov_method, n_jobs=n_jobs)
    iM12 = invsqrtm(M)
    M12= sqrtm(M)
    A = iM12@C@iM12
    B = M12@powm(A, (scale-1)/2)@iM12
    X = B@X
    return X

def _cost_euc(R: ndarray, Mt: ndarray, Ms: ndarray, 
        weights: Optional[ndarray] = None):
    if weights is None:
        weights = anp.ones(len(Mt))

    cost = 0
    for i, a in enumerate(zip(Ms, Mt)):
        Msi, Mti = a
        Mti = anp.dot(R, anp.dot(Mti, R.T))
        cost += weights[i]*anp.square(anp.linalg.norm(Mti-Msi))
    # cost = anp.linalg.norm(Mt-Ms, ord='fro', axis=(-2, -1))
    return cost

def _cost_rie(R: ndarray, Mt: ndarray, Ms: ndarray,
        weights: Optional[ndarray] = None):
    if weights is None:
        weights = anp.ones(len(Mt))
    Mt = anp.matmul(R, anp.matmul(Mt, R.T))
    # distance_riemann not implemented in autograd, must provide egrad
    cost = anp.square(distance_riemann(Ms, Mt))
    return anp.dot(cost, weights)

def _egrad_rie(R: ndarray, Mt: ndarray, Ms: ndarray,
        weights: Optional[ndarray] = None):
    if weights is None:
        weights = anp.ones(len(Mt))
    # I dont't understand the code!!!
    iMt12 = invsqrtm(Mt)
    Ms12 = sqrtm(Ms)
    term_aux = anp.matmul(R, anp.matmul(Mt, R.T))
    term_aux = anp.matmul(iMt12, anp.matmul(term_aux, iMt12))
    g = 4*np.matmul(np.matmul(iMt12, logm(term_aux)), np.matmul(Ms12, R))
    g = g*weights[:, np.newaxis, np.newaxis]
    return anp.sum(g, axis=0)

def _procruster_cost_function_euc(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = anp.linalg.norm(t1-t2)**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)

def _procruster_cost_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = distance_riemann(t1, t2)[0]**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)

def _procruster_egrad_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))
    
    g = []
    for Mti, Msi, wi in zip(Mt, Ms, weights):
        iMti12 = invsqrtm(Mti)
        Msi12 = sqrtm(Msi)
        term_aux = anp.dot(R, anp.dot(Msi, R.T))
        term_aux = anp.dot(iMti12, anp.dot(term_aux, iMti12))
        gi = 4 * anp.dot(anp.dot(iMti12, logm(term_aux)), anp.dot(Msi12, R))
        g.append(gi * wi)

    g = anp.sum(g, axis=0)

    return g

def _get_rotation_matrix(Mt: ndarray, Ms: ndarray,
        weights: Optional[ndarray] = None,
        metric: str ='euclid'):
    Mt = Mt.reshape(-1, *Mt.shape[-2:])
    Ms = Ms.reshape(-1, *Ms.shape[-2:])

    n = Mt[0].shape[0]
    manifolds = Rotations(n)

    if metric == 'euclid':
        # cost = partial(_cost_euc, Mt=Mt, Ms=Ms, weights=weights)
        cost = partial(_procruster_cost_function_euc, Mt=Mt, Ms=Ms)
        problem = Problem(manifold=manifolds, cost=cost, verbosity=0)
    elif metric == 'riemann':
        # cost = partial(_cost_rie, Mt=Mt, Ms=Ms, weights=weights)    
        # egrad = partial(_egrad_rie, Mt=Mt, Ms=Ms, weights=weights)
        cost = partial(_procruster_cost_function_rie, Mt=Mt, Ms=Ms)
        egrad = partial(_procruster_egrad_function_rie, Mt=Mt, Ms=Ms)
        problem = Problem(manifold=manifolds, cost=cost, egrad=egrad, verbosity=0) 

    solver = SteepestDescent(mingradnorm=1e-3)

    Ropt = solver.solve(problem)

    return Ropt

def get_rotate(
        Xs: ndarray, ys: ndarray, 
        Xt: ndarray, yt: ndarray,
        cov_method: str = 'cov',
        metric: str = 'euclid',
        n_jobs: Optional[int] = None):
    slabels = np.unique(ys)
    tlabels = np.unique(yt)
    Xs = np.reshape(Xs, (-1, *Xs.shape[-2:]))
    Xt = np.reshape(Xt, (-1, *Xt.shape[-2:]))
    Xs = Xs - np.mean(Xs, axis=-1, keepdims=True)
    Xt = Xt - np.mean(Xt, axis=-1, keepdims=True)
    Cs = covariances(Xs, estimator=cov_method, n_jobs=n_jobs)
    Ct = covariances(Xt, estimator=cov_method, n_jobs=n_jobs)

    Ms = np.stack([mean_riemann(Cs[ys==label]) for label in slabels])
    Mt = np.stack([mean_riemann(Ct[yt==label]) for label in tlabels])

    Ropt = _get_rotation_matrix(Mt, Ms, metric=metric)
    return Ropt

def rotate(Xt: ndarray, Ropt: ndarray):
    Xt = np.reshape(Xt, (-1, *Xt.shape[-2:]))
    Xt = Xt - np.mean(Xt, axis=-1, keepdims=True)
    return Ropt@Xt








