# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/22
# License: MIT License
"""Manifold Embedded Knowledge Transfer.

Manifold embedded knowledge transfer (MEKT) transfers features from
the tangent space of a positive definite manifold through the fusion
of classical transfer methods in transfer learning.

MEKT[1] can be mainly divided into feature extraction part and domain adaptation part.

In the feature extraction section, MEKT chooses to perform Riemann alignment
on the covariance matrix of each individual, so that the Riemann center points
of each individual's data are located in the identity matrix, and extract the
tangent vector of the sample as the main feature.

In the domain adaptation part, MEKT solves from four aspects: minimizing joint
probability distribution differences, source domain separability, target domain
local consistency, and regularization constraints.

.. [1] Zhang W, Wu D. Manifold embedded knowledge transfer for brain-computer interfaces [J].IEEE
       Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28 (5): 1117–1127.

souce code of MEKT: https://github.com/chamwen/MEKT.git

"""
import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.stats import f_oneway
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..manifold import tangent_space, mean_riemann
from ..utils.covariance import Covariance, invsqrtm


def anova_dimension_reduction(Xs, ys, d):
    """Dimension reduction in MEKT.

    MEKT use len(ys) as d.

    Parameters
    ----------
    Xs: ndarray
        features, shape (n_trials, n_features).
    ys: ndarray
        labels, shape (n_trials,).
    d: int
        reduce to dimension d

    Returns
    -------
    f_ix: ndarray
        the index of selected features, shape (d,).

    """

    labels = np.sort(np.unique(ys))
    n_samples, n_features = Xs.shape
    f_values = np.zeros((n_features,))

    for i in range(n_features):
        groups = []
        for label in labels:
            groups.append(Xs[ys == label, i])

        f_values[i], _ = f_oneway(*groups)

    f_ix = np.sort(np.argsort(f_values)[::-1][:d])
    return f_ix


def source_discriminability(Xs, ys):
    """Source features discriminability.

    Parameters
    ----------
    Xs : ndarray
        source features, shape (n_trials, n_features).
    ys : ndarray
        labels, shape (n_trials).

    Returns
    -------
    Sw: ndarray
        within-class scatter matrix, shape (n_features, n_features).
    Sb: ndarray
        between-class scatter matrix, shape (n_features, n_features).

    """

    classes = np.unique(ys)
    n_samples, n_features = Xs.shape
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    mean_total = np.mean(Xs, axis=0, keepdims=True)

    for c in classes:
        Xi = Xs[ys == c, :]
        mean_class = np.mean(Xi, axis=0, keepdims=True)
        Sw = Sw + (Xi - mean_class).T @ (Xi - mean_class)
        Sb = Sb + (mean_class.T - mean_total.T) @ (mean_class - mean_total) * len(Xi)

    return Sw, Sb


def graph_laplacian(Xs, k=10, t=1):
    """Graph Laplacian Matrix.

    Currently with heat kernel implemented.

    Parameters
    ----------
    Xs: ndarray
        features, shape (n_trials, n_samples).
    k: int
        k nearest neighbors, by default 10.
    t: int
        heat kernel parameter, by default 1.

    Returns
    -------
    L: ndarray
        unnormalized laplacian kernel, shape (n_trials, n_trials).
    D: ndarray
        degree matrix, L = D - W, shape (n_trials, n_trials).

    """

    # compute pairwise distance
    pair_dist = squareform(pdist(Xs, metric="euclidean"))

    # knn
    # MEKT has self-connection, W[0,0] = 1
    ix = np.argsort(pair_dist, axis=-1)[:, : k + 1]

    # heat kernel
    heat_W = np.exp(-np.square(pair_dist) / (2 * np.square(t)))
    W = np.zeros((Xs.shape[0], Xs.shape[0]))

    for i, ind in enumerate(ix):
        W[i, ind] = heat_W[i, ind]

    W = np.maximum(W, W.T)

    D = np.diag(np.sum(W, axis=-1))
    L = D - W

    return L, D


def scatter_matrix(X, y):
    """Compute between-class scatter matrix.

    Parameters
    ----------
    X : ndarray
        features, shape (n_trials, n_features).
    y : ndarray
        labels, shape (n_trials,).

    Returns
    -------
    Sb: ndarray
        between-class scatter matrix, shape (n_features, n_features).

    """

    classes = np.unique(y)
    M = np.mean(X, axis=0, keepdims=True)
    Sb = np.zeros((X.shape[-1], X.shape[-1]))
    for class_id in classes:
        nk = len(X[y == class_id])
        Mk = np.mean(X[y == class_id], keepdims=True)
        Sb += nk * (Mk - M).T @ (Mk - M)
    return Sb


def dte(Xs, Xt, ys):
    """Domain Transferiability Estimation.

    Parameters
    ----------
    Xs: ndarray
        source features, shape (n_source_trials, n_features).
    Xt: ndarray
        target features, shape (n_traget_trials, n_features).
    ys: ndarray
        source labels, shape (n_source_trials,).

    Returns
    -------
    dis: float
        discriminability of Ds.
    dif: float
        difference of Ds and Dt.

    """

    Sb = scatter_matrix(Xs, ys)
    dis = np.linalg.norm(Sb, 1)
    Sb = scatter_matrix(
        np.concatenate((Xs, Xt), axis=0),
        np.concatenate((np.zeros(len(Xs)), np.ones(len(Xt)))),
    )
    dif = np.linalg.norm(Sb, 1)
    return dis, dif


def choose_multiple_subjects(Xs, Xt, ys, y_subjects, k=1):
    """choose k most appropriate subjects according to dte.

    Parameters
    ----------
    Xs: ndarray
        source features, shape (n_trials*n_subjects, n_features).
    Xt: ndarray
        target features, shape (n_trials, n_features).
    ys: ndarray
        source labels, shape (n_trials*n_subjects, n_features).
    y_subjects: ndarray
        subject labels, shape (n_trials*n_subjects,).
    k : int
        k subjects, by default 1.

    Returns
    -------
    subject_ix: ndarray
        selected subject boolean index, shape (n_trials*n_subjects,).
    selected_subjects: ndarray
        selected subject ids, shape (k,).

    """

    subjects = np.unique(y_subjects)
    ranks = []
    for subject in subjects:
        dis, dif = dte(Xs[y_subjects == subject], Xt, ys[y_subjects == subject])
        # Note: original code map dif to [1, 0], dis to [0, 1] and  multiply them
        r = dis / (dif + np.finfo(np.float).resolution)
        ranks.append(r)

    ranks = np.argsort(ranks)[::-1]
    subject_ix = np.zeros((len(y_subjects)), dtype=np.bool)
    selected_subjects = subjects[ranks[:k]]
    for subject in selected_subjects:
        subject_ix = np.logical_or(subject_ix, y_subjects == subject)

    return subject_ix, selected_subjects


def mekt_feature(X, covariance_type):
    """Covariance Matrix Centroid Alignment and Tangent Space Feature Extraction.

    Parameters
    ----------
    X : ndarray
        EEG data, shape (n_trials, n_channels, n_timepoints).
    covariance_type: str
        Covariance category, default to 'lwf'


    Returns
    -------
    featureX: ndarray
        feature of X, shape (n_trials, n_feature).

    """

    covest = Covariance(estimator=covariance_type)
    X = covest.transform(X)
    # Covariance Matrix Centroid Alignment
    M = mean_riemann(X)
    iM12 = invsqrtm(M)
    C = iM12 @ X @ iM12.T
    # Tangent Space Feature Extraction
    featureX = tangent_space(C, np.eye(M.shape[0]))

    return featureX


def mekt_kernel(Xs, Xt, ys, d=10, max_iter=5, alpha=0.01, beta=0.1, rho=20, k=10, t=1):
    """Find the projection matrix to make the distribution of the source
       and target domains as close as possible after projection.

    Parameters
    ----------
    Xs: ndarray
        source features, shape (n_source_trials, n_features).
    Xt: ndarray
        target features, shape (n_target_trials, n_features).
    ys: ndarray
        source labels, shape (n_source_trials,).
    d: int
        selected d projection vectors, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.

    Returns
    -------
    A: ndarray
        projection matrix for source, shape (n_features, d).
    B: ndarray
        projection matrix for target, shape (n_features, d).

    """

    ns_samples, ns_features = Xs.shape
    nt_samples, nt_features = Xt.shape

    # source domain discriminability
    Sw, Sb = source_discriminability(Xs, ys)
    P = np.zeros((2 * ns_features, 2 * ns_features))
    P[:ns_features, :ns_features] = Sw
    P0 = np.zeros((2 * ns_features, 2 * ns_features))
    P0[:ns_features, :ns_features] = Sb

    # target locality
    L, D = graph_laplacian(Xt, k=k, t=t)  # should be (n_samples, n_samples)
    iD12 = invsqrtm(D)
    L = iD12 @ L @ iD12
    L = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ L @ Xt)

    Q = np.block(
        [
            [np.eye(ns_features), -1 * np.eye(nt_features)],
            [-1 * np.eye(ns_features), 2 * np.eye(nt_features)],
        ]
    )

    Ht = np.eye(nt_samples) - (1 / nt_samples) * np.ones((nt_samples, 1)) @ np.ones(
        (1, nt_samples)
    )
    S = block_diag(np.zeros((ns_features, ns_features)), Xt.T @ Ht @ Xt)

    classes = np.sort(np.unique(ys))
    onehot_enc = OneHotEncoder(categories=[classes], sparse=False)
    Ns = onehot_enc.fit_transform(np.reshape(ys, (-1, 1))) / len(ys)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    yt = clf.fit(Xs, ys).predict(Xt)  # initial predict label

    X = block_diag(Xs, Xt)
    Emin_temp = alpha * P + beta * L + rho * Q
    Emax = S + alpha * P0 + 1e-3 * np.eye(ns_features + nt_features)
    for _ in range(max_iter):
        # update fake yt
        Nt = onehot_enc.fit_transform(np.reshape(yt, (-1, 1))) / len(yt)

        # calculate R: joint probability distribution shift
        M = np.block([[Ns @ Ns.T, -Ns @ Nt.T], [-Nt @ Ns.T, Nt @ Nt.T]])
        R = X.T @ M @ X

        # generalized eigen-decompostion
        Emin = Emin_temp + R

        w, V = eigh(Emin, Emax)

        A = V[:ns_features, :d]
        B = V[ns_features:, :d]

        # embedding
        Zs = Xs @ A
        Zt = Xt @ B

        yt = clf.fit(Zs, ys).predict(Zt)

    return A, B


class MEKT(BaseEstimator, TransformerMixin):
    """
    Manifold Embedded Knowledge Transfer(MEKT) [1]_.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-01-22

    update log:
        2021-01-22 by Swolf<swolfforever@gmail.com>

        2023-12-09 by heoohuan <heoohuan@163.com>（Add code annotation）

    Parameters
    ----------
    subspace_dim: int
        Selected projection vector, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        Covariance category, by default 'lwf'.

    Attributes
    ----------
    subspace_dim: int
        Selected projection vector, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        covariance category, by default 'lwf'.
    A_: ndarray
        first type center, shape(n_class, n_channels, n_channels).
    B_: ndarray
       second type center, shape(n_class, n_channels, n_channels).

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Zhang W, Wu D. Manifold embedded knowledge transfer for brain-computer interfaces
       [J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28 (5): 1117–1127.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using MEKT

       from brainda.algorithms.transfer_learning import MEKT
       mekt = MEKT(max_iter=5)
       source_features, target_features = mekt.fit_transform(Xs, ys, Xt)

    """

    def __init__(
        self,
        subspace_dim: int = 10,
        max_iter: int = 5,
        alpha: float = 0.01,
        beta: float = 0.1,
        rho: float = 20,
        k: int = 10,
        t: int = 1,
        covariance_type="lwf",
    ):
        self.subspace_dim = subspace_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.k = k
        self.t = t
        self.covariance_type = covariance_type

    def fit_transform(self, Xs, ys, Xt):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        Xs: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        ys: ndarray
            Label, shape(n_trials,).
        Xt: ndarray
            Target of EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        -------
        source_features: ndarray
            source domain features, shape(n_trials, n_features).
        target_features: ndarray
            target domain features, shape(n_trials, n_features).

        """
        featureXs = mekt_feature(Xs, self.covariance_type)
        featureXt = mekt_feature(Xt, self.covariance_type)
        self.A_, self.B_ = mekt_kernel(
            featureXs,
            featureXt,
            ys,
            d=self.subspace_dim,
            max_iter=self.max_iter,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            k=self.k,
            t=self.t,
        )
        source_features = featureXs @ self.A_
        target_features = featureXt @ self.B_
        return source_features, target_features
