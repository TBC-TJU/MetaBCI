# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
"""
Riemannian Geometry for BCI.
"""
from typing import Optional
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.linalg import eigvalsh, pinv

from ..utils.covariance import covariances, sqrtm, invsqrtm, logm, expm, powm


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

    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}
    where :math:`\lambda_i` are the joint eigenvalues of A and B.

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

    """

    def _single_distance_riemann(A, B):
        dist = np.sqrt(np.sum(np.log(eigvalsh(A, B)) ** 2))
        return dist

    A = A.reshape((-1, *A.shape[-2:]))
    B = B.reshape((-1, *B.shape[-2:]))

    if A.shape[0] == 1:
        A = np.broadcast_to(A, B.shape)
    elif B.shape[0] == 1:
        B = np.broadcast_to(B, A.shape)

    d = Parallel(n_jobs=n_jobs)(
        delayed(_single_distance_riemann)(a, b) for a, b in zip(A, B)
    )
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


def mean_riemann(
        covmats, tol=1e-11, maxiter=300, init=None, sample_weight=None, n_jobs=None
):
    """Return the mean covariance matrix according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of riemannian distance to the mean.

    .. math::
        \mathbf{C} = \\arg \min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    where :math:`\delta_R` is riemann distance.

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
        J = np.sum(sample_weight[:, np.newaxis, np.newaxis] * J, axis=0)
        crit = np.linalg.norm(J, ord="fro")
        h = nu * crit

        C = C12 @ expm(nu * J, n_jobs=1) @ C12
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
    coeffs = (
            np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + np.eye(n_channels)
    )[ind]
    vSi = Si[:, ind[0], ind[1]] * coeffs
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
    coeffs = (
            np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1)
            + 2 * np.eye(n_channels)
    )[ind]
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


def mdrm_kernel(
        X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None, n_jobs: int = 1
):
    """Minimum Distance to Riemannian Mean.

    Parameters
    ----------
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials)
    sample_weight : Optional[ndarray], optional
        sample weights, by default None
    n_jobs : int
        the number of jobs to use, by default 1

    Returns
    -------
    ndarray
        centroids of each class, shape (n_class, n_channels, n_channels).
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    Cx = covariances(X, estimator="lwf", n_jobs=n_jobs)
    sample_weight = np.ones((len(X))) if sample_weight is None else sample_weight

    Centroids = Parallel(n_jobs=n_jobs)(
        delayed(mean_riemann)(Cx[y == label], sample_weight=sample_weight[y == label])
        for label in labels
    )
    return np.stack(Centroids)


class FGDA(BaseEstimator, TransformerMixin):
    """
        Characteristics and uses of classes FGDA

        Authors: Swolf <swolfforever@gmail.com>

        Created on: 2021-1-23

        update log:
            2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

        Fisher Geodesic Discriminate Analysis(FGDA) is the application of Fisher Linear Discriminate Analysis
        in the Riemannian tangent space.FGDA first calculates the projection vectors of the sample covariance
        matrix of EEG signals in the Riemannian tangent space.Then, leveraging the properties of Riemannian
        tangent space as a Euclidean space, it performs discriminant feature extraction on the projected
        vectors in the tangent space based on the Fisher Linear Discriminant Analysis criterion.

        Parameters
        -----------
        n_jobs:int
           the default of n_jobs is None,meaning it will utilize all available CPUs.
        Attributes
        -----------
        lda_:discriminate_analysis.Linear Discriminate Analysis
           LDA
        P_:ndarray:shape(int,int)
           the average covariance matrix calculates from the Riemann matrix
        W_:ndarray,shape(int,int)
           the weight of LDA
        References
        ----------
        .. [1] Barachant A, Bonnet S, Congedo M, et al. Riemannian geometry applied to BCI
            classification [C].International Conference on Latent Variable Analysis and Signal Separation, 2010: 629–636

    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X:ndarray:shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        y:ndarray:shape(n_trails)
           the labels of train data
        """
        Pi = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_.fit(vSi, y)
        W = self.lda_.coef_.copy()
        self.W_ = W.T @ pinv(W @ W.T) @ W  # n_feat by n_feat
        return self

    def transform(self, X):
        """
        Calculate the FGDA from the parameters stored in self

        Parameters
        ----------
        X:ndarray:shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        Pi:ndarray
           the projection matrix

        """
        Pi = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        vSi = vSi @ self.W_
        Pi = untangent_space(vSi, self.P_)
        return Pi


class MDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    """ Characteristics and uses of classes  MDRM

        Authors: Swolf <swolfforever@gmail.com>

        Date: 2021-1-23

        update log:
            2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

        Minimum Distance to Riemannian Mean(MDRM) is a decoding algorithm based on Riemann distance metric.
        MDRM calculates the covariance matrix of EEG signals, estimates the Riemannian centroids for each class,
        then determines the class of a test sample by computing the minimum distance between the test data's covariance
        matrix and the mean point.

        Parameters
        ----------
        n_jobs:int
           n_jobs the default is None,meaning it will utilize all available CPUs.
        Attributes
        ----------
        classes_:ndarray,shape(int)
            class labels
        centroids_:ndarray,shape(int,float,float)
            Riemannian centroid of two classes

        References
        ----------
        .. [1] Barachant A, Bonnet S, Congedo M, et al. Riemannian geometry applied to BCI
            classification [C].International Conference on Latent Variable Analysis and Signal Separation, 2010: 629–636

        Tip
        ----
        ..  code-block:: python
            :linenos:
            :caption: An example using MDRM

            from metabci.brainda.algorithms.mainfold import MDRM
            estimator = MDRM()
            p_labels = estimator.fit(X[train_ind],y[train_ind]).predict(X[test_ind])

    """

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None):
        """
        Train the model

        Parameters
        ----------
        X: ndarray,shape(n_trails,n_channels,n_samples)
            train data: EEG signals

        y: ndarray,shape(n_trails)
            label of train data

        sample_weight: ndarray
            the weight of the samples which is optional,the default is None

        Returns
        -------
        self:the model
        """
        self.classes_ = np.unique(y)
        self.centroids_ = mdrm_kernel(
            X, y, sample_weight=sample_weight, n_jobs=self.n_jobs
        )
        return self

    def _transform_distance(self, X: ndarray):
        """
        Calculate the Riemann distance

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
            train data: EEG signals

        Returns
        -------
        dist:ndarray
            the Riemann distance
        """

        Cx = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        dist = np.stack(
            [
                distance_riemann(Cx, centroid, n_jobs=self.n_jobs)
                for centroid in self.centroids_
            ]
        ).T
        return dist

    def transform(self, X: ndarray):
        """
        Calculate the Riemann distance of each class using the parameters from self.

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
            train data: EEG signals

        Returns
        -------
        self._transform_distance(X):
            the Riemann distance of each class
        """
        return self._transform_distance(X)

    def predict(self, X: ndarray):
        """
        Predict the label

        Parameters
        ----------
        X:ndarray, shape(n_trials, n_channels, n_samples)
           train data: EEG signals

        Returns
        -------
        self.classes_[np.argmin(dist, axis=1)]:ndarray,shape(n_trails)
            predicted labels

        """
        dist = self._transform_distance(X)
        return self.classes_[np.argmin(dist, axis=1)]

    def predict_proba(self, X: ndarray):
        """
        Predict label probabilities

        Parameters
        ----------
        X:ndarray, shape(n_trials, n_channels, n_samples)
            train data: EEG signals

        Returns
        -------
        softmax(-1 * self._transform_distance(X)):ndarray,shape(n_trails)
            the probabilities of the predicted labels

        """
        return softmax(-1 * self._transform_distance(X))


class FgMDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Characteristics and uses of classes FgMDRM

    Authors: Swolf <swolfforever@gmail.com>

    Date: 2021-1-23

    update log:
        2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

    The Fisher Geodesic Minimum Distance to Riemannian Mean(FGMDRM)  algorithm is a fusion of
    MDRM and FGDA.The algorithm first employs FGDA in the tangent space to filter the data,extracting
    key discriminative features,removing irrelevant noise components. Subsequently, the extracted
    discriminative features are remapped back to the manifold space. The covariance matrix of the
    filtered sample space is then calculated based on MDRM to determine the Riemannian centroids for
    each class. The classification of test data is performed based on the minimum distance principle.

    Parameters
    ----------
    n_jobs:int
        the default of n_jobs is None,meaning it will utilize all available CPUs.

    Attributes
    ----------
    n_jobs:int
        the default of n_jobs is None,meaning it will utilize all available CPUs.
    classes_:ndarray,shape(int)
        the class of samples
    centroids_:ndarray,shape(int,float,float)
        Riemannian centroid of two classes
    fgda_:algorithms.mainfold.riemann.FGDA
        Fisher Geodesic Discriminate Analysis(FGDA)

    References
    ---------
    .. [1] Barachant A, Bonnet S, Congedo M, et al. Riemannian geometry applied to BCI
        classification [C].International Conference on Latent Variable Analysis and Signal Separation, 2010: 629–636

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: An example using FgMDRM

       from metabci.brainda.algorithms.mainfold import FgMDRM
       estimator = FgMDRM()
       p_labels = estimator.fit(X[train_ind],y[train_ind]).predict(X[test_ind])

    """

    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None):
        """
        Train the model.

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        y:ndarray,shape(n_trails)
           the labels if train data

        sample_weight:ndarray
           the weight of the model

        """
        self.classes_ = np.unique(y)
        self.fgda_ = FGDA(n_jobs=self.n_jobs)
        Cx = self.fgda_.fit_transform(X, y)
        sample_weight = np.ones((len(X))) if sample_weight is None else sample_weight
        Centroids = Parallel(n_jobs=self.n_jobs)(
            delayed(mean_riemann)(
                Cx[y == label], sample_weight=sample_weight[y == label]
            )
            for label in self.classes_
        )
        self.centroids_ = np.stack(Centroids)
        return self

    def _transform_distance(self, X: ndarray):
        """
        Calculate the Riemann distance

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        dist:ndarray
           Riemann distance
        """
        Cx = self.fgda_.transform(X)
        dist = np.stack(
            [
                distance_riemann(Cx, centroid, n_jobs=self.n_jobs)
                for centroid in self.centroids_
            ]
        ).T
        return dist

    def transform(self, X: ndarray):
        """
        Calculate the Riemann distance of each class from the parameters stored in self

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        Self._transform_distance(X)
            the Riemann distance of each class

        """
        return self._transform_distance(X)

    def predict(self, X: ndarray):
        """
        Predict the labels

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        self.classes_[np.argmin(dist, axis=1)]:ndarray,shape(n_trails)
           predicted labels

        """
        dist = self._transform_distance(X)
        return self.classes_[np.argmin(dist, axis=1)]


class TSClassifier(BaseEstimator, ClassifierMixin):
    """
    Characteristics and uses of classes TSClassifier

    Authors: Swolf <swolfforever@gmail.com>

    Date: 2021-1-23

    update log:
        2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

    The Tangent Space Classifier (TSClassifier) is a general term for classifiers constructed in the Riemannian
    tangent space,which is treated as a Euclidean space. Methods such as LDA (Linear Discriminant Analysis),
    SVM (Support Vector Machine),Logistic Regression, and others are employed to build classifiers in this
    Riemannian tangent space.

    Parameters
    ----------
    n_jobs:int
      the default of n_jobs is None,meaning it will utilize all available CPUs.
    clf:linear_model._logistic.LogisticRegression
       Logistic Regression

    Attributes
    ----------
    n_jobs:int
      the default of n_jobs is None,meaning it will utilize all available CPUs.
    clf:linear_model._logistic.LogisticRegression
       Logistic Regression
    P_:ndarray,shape(int,int)
       The average covariance matrix returned according to the Riemann matrix

    References
    ----------
    .. [1] Barachant A, Bonnet S, Congedo M, et al. Multiclass brain–computer interface classification
        by Riemannian geometry [J].IEEE Transactions on Biomedical Engineering, 2011, 59 (4): 920–928.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: An example using TSClassifier

       from metabci.brainda.algorithms.manifold import TSClassifier
       estimator = TSClassifier()
       p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])

    """

    def __init__(self, clf=LogisticRegression(), n_jobs=None):
        self.clf = clf
        self.n_jobs = n_jobs

        if not isinstance(self.clf, ClassifierMixin):
            raise TypeError("clf must be a ClassifierMixin")

    def fit(self, X: ndarray, y: ndarray):
        """
        Train the model

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals
        y:ndarray,shape(n_trails)
           the labels if train data

        """
        Pi = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        self.clf.fit(vSi, y)
        return self

    def predict(self, X: ndarray):
        """
        Predict labels

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        self.clf.predict(vSi):ndarray,shape(n_trails)\
           predicted labels

        """
        Pi = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        return self.clf.predict(vSi)

    def predict_proba(self, X: ndarray):
        """
        Predict label probabilities

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        self.clf.predict_proba(vSi):ndarray,shape(n_trails)
            predicted label probabilities

        """
        Pi = covariances(X, estimator="lwf", n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        return self.clf.predict_proba(vSi)


class Alignment(BaseEstimator, TransformerMixin):
    """
    Characteristics and uses of classes Alignment

    Authors: Swolf <swolfforever@gmail.com>

    Date: 2021-1-23

    update log:
        2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

    Riemannian Alignment (RA) uses the Riemannian mean of the covariance matrix of all trials as the reference matrix,
    so that the center point of the whitened covariance matrix is located in the identity matrix.
    By performing RA processing on each subject's data, the center point of the covariance matrix for all individuals
    can be aligned. Euclidean Alignment (EA) replaces the Riemann mean covariance matrix with the Euclidean mean
    covariance matrix.

    Parameters
    ----------
    n_jobs:int
       the default is None
    align_method:str
       choose the alignment method:'riemann' or 'euclid'
    cov_method:str
       covariance estimators:'lwf'

    Attributes
    ----------
    n_jobs:int
       the default is None
    align_method:str
       choose the alignment method:'riemann' or 'euclid'
    cov_method:str
       covariance estimators:'lwf'
    iC12_:ndarray,shape(int,int)
       aligned Riemann/Euclidean center

    References
    ----------
    .. [1] Zanini P, Congedo M, Jutten C, et al. Transfer learning: A Riemannian geometry framework with applications
        to brain–computer interfaces [J].
        IEEE Transactions on Biomedical Engineering, 2017, 65 (5): 1107–1116.

    .. [2] He H, Wu D. Transfer learning for Brain–Computer interfaces: A Euclidean space data alignment approach [J].
        IEEE Transactions on Biomedical Engineering, 2019, 67 (2): 399–410.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: An example using Alignment

        from metabci.brainda.algorithms.manifold import Alignment
        estimator = Alignment(align_method='riemann')
        filterX = estimator.fit(X).transform(X)
    """

    def __init__(
            self,
            align_method: str = "euclid",
            cov_method: str = "lwf",
            n_jobs: Optional[int] = None,
    ):
        self.align_method = align_method
        self.cov_method = cov_method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: Optional[ndarray] = None):
        """
        Train the model,calculate the aligned center

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
           train data: EEG signals

        Returns
        -------
        self
        """
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if self.align_method == "euclid":
            self.iC12_ = self._euclid_center(X)
        elif self.align_method == "riemann":
            self.iC12_ = self._riemann_center(X)
        else:
            raise ValueError("non-supported aligning method.")

        return self

    def transform(self, X):
        """
        Obtain the aligned individual data

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
            EEG data

        Returns
        -------
        X:ndarray,shape(n_trails,n_channels,n_samples)
            aligned EEG data
        """
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        X = np.matmul(self.iC12_, X)
        return X

    def _euclid_center(self, X):
        """
        Calculate the Euclidean center

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels,n_samples)
            EEG data

        Returns
        -------
        invsqrtm(C):ndarray
             the Euclidean center

        """
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        C = np.mean(Cs, axis=0)
        return invsqrtm(C)

    def _riemann_center(self, X):
        """
        Calculate the Riemann center

        Parameters
        ----------
         X:ndarray,shape(n_trails,n_channels,n_samples)
            EEG data

        Returns
        -------
        invsqrtm(C):ndarray
            the Riemann center

        """
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        C = mean_riemann(Cs, n_jobs=self.n_jobs)
        return invsqrtm(C)


class RecursiveAlignment(BaseEstimator, TransformerMixin):
    """
    Characteristics and uses of classes RecursiveAlignment

    Authors: Swolf <swolfforever@gmail.com>

    Date: 2021-1-23

    update log:
        2023-12-18 by Yuwei Liu<liuyuwei20010905@163.com> add code annotation

    In order to overcome the problem that the trial data gradually appear in chronological order under the online experiment,
    there is no initial sample size estimation center, and the calculation process of the Riemann center is complex,
    and it takes a lot of time to recalculate the Riemannian center in the feedback stage, the Recursive Riemannian Alignment
    (rRA) and Recursive Euclidean Alignment (rEA) suitable for the online stage were proposed.

    Parameters
    ----------
    n_jobs:int
       the default is None
    align_method:str
       choose the alignment method:'riemann' or 'euclid'
    cov_method:str
       covariance estimators:'lwf'

    Attributes
    ----------
    n_jobs:int
       the default is None
    align_method:str
       choose the alignment method:'riemann' or 'euclid'
    cov_method:str
       covariance estimators:'lwf'
    iC12_:ndarray,shape(int,int)
       aligned Riemann/Euclidean center
    n_tracked:int
       the number of iterations
    C_:ndarray
       the Euclid or Riemann center after iteration

    References
    ----------
    .. [1] Xu Lichao, Xu Minpeng, Ke Yufeng, An Xingwei, Liu Shuang, Ming Dong*. Cross-Dataset Variability Problem
        in EEG Decoding with Deep Learning[J].
        Frontiers in Human Neuroscience, 2020, 14: 103

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: An example using RecursiveAlignment

       from metabci.brainda.algorithms.manifold import RecursiveAlignment
       estimator = RecursiveAlignment(align_method='riemann')
       filterX = estimator.fit(X).transform(X)
    """

    def __init__(
            self,
            align_method: str = "euclid",
            cov_method: str = "lwf",
            n_jobs: Optional[int] = None,
    ):
        self.align_method = align_method
        self.cov_method = cov_method
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        recursive alignment

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels.n_samples)
            EEG data
        Returns
        -------
        self

        """
        return self

    def transform(self, X):
        """
        obtain the subject's data after recursive alignment

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels.n_samples)
            EEG data

        Returns
        -------
        X:ndarray,shape(n_trails,n_channels.n_samples)
            the individual data after recursive alignment
        """
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Cs = covariances(X, estimator=self.cov_method, n_jobs=self.n_jobs)
        if not hasattr(self, "iC12_"):
            self.iC12_ = np.eye(X.shape[1])
            self.C_ = np.eye(X.shape[1])
            self.n_tracked = 0
        X = self._recursive_fit_transform(X, Cs)
        return X

    def _recursive_fit_transform(self, X, Cs):
        """
        obtain the subject's data after recursive alignment

        Parameters
        ----------
        X:ndarray,shape(n_trails,n_channels.n_samples)
            EEG data
        Cs:ndarray
            the covariance matrix of X

        Returns
        -------
        X:ndarray
            the individual data after recursive alignment

        """
        for i in range(len(X)):
            if self.align_method == "euclid":
                self._recursive_euclid_center(Cs[i])
            elif self.align_method == "riemann":
                self._recursive_riemann_center(Cs[i])
            else:
                raise ValueError("non-supported aligning method.")
            if self.n_tracked == 1:
                X[i] = X[i] / np.std(X[i], axis=(-2, -1), keepdims=True)
            else:
                X[i] = self.iC12_ @ X[i]
        return X

    def _recursive_euclid_center(self, C):
        """
        Calculate the euclid center after recursive alignment

        Parameters
        ----------
        C:ndarray
           the euclid center calculated in the offline period

        """
        self.n_tracked += 1
        alpha = 1 / (self.n_tracked)
        self.C_ = (1 - alpha) * self.C_ + alpha * C
        self.iC12_ = invsqrtm(self.C_)

    def _recursive_riemann_center(self, C):
        """
        Calculate the riemann center after recursive alignment

        Parameters
        ----------
        C:ndarray
           the riemann center calculated in the offline period

        """
        self.n_tracked += 1
        alpha = 1 / (self.n_tracked)
        self.C_ = geodesic(self.C_, C, alpha, n_jobs=1)
        self.iC12_ = invsqrtm(self.C_)
