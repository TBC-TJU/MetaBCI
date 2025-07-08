# -*- coding: utf-8 -*-
"""
Least squares transformation (LST)[1] is an EEG transfer learning algorithm
that minimizes the least squares regression error between source domain
transfer samples and target domain samples.The original paper[1] has proven that
this algorithm can effectively transfer SSVEP datasets from different collection
devices and subjects, reducing BCI system calibration time.

.. [1] Chiang K-J, Wei C-S, Nakanishi M, et al. Boosting template-based SSVEP
    decoding by cross domain transfer learning [J]. Journal of Neural Engineering,
    2021, 18 (1): 016002.

souce paper of LST: https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e.

"""
import numpy as np
from numpy import ndarray
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


def lst_kernel(S: ndarray, T: ndarray):
    """Calculate LST transformation matrix.

    Parameters
    ----------
    S:ndarray
        source features, shape (n_source_trials, n_features).
    T:ndarray
        target features, shape (n_target_trials, n_features).

    Returns
    -------
    P: ndarray
        projection matrix for target, shape (n_features, d).

    """

    P = T @ S.T @ pinv(S @ S.T)
    return P


class LST(BaseEstimator, TransformerMixin):
    """LST converter [1]_.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-01-24

    update log:
        2021-01-24 by Swolf<swolfforever@gmail.com>

        2023-12-09 by heoohuan <heoohuan@163.com>（Add code annotation）


    Parameters
    ----------
    n_jobs: int
        n_jobs defaults to None, which means using all CPUs.

    Attributes
    ----------
    n_jobs: int
        n_jobs defaults to None, which means using all CPUs.
    T_: list
        Average template for different classes of data.
    classes_: ndarray
        Data category, shape(int).

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Chiang K-J, Wei C-S, Nakanishi M, et al. Boosting template-based SSVEP decoding
       by cross domain transfer learning [J]. Journal of Neural Engineering, 2021, 18 (1): 016002.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using LST

        from brainda.algorithms.transfer_learning import LST
        LST_estimator = LST()
        LST_estimator.fit(Xt[train_ind], yt[train_ind])
        Xs_transform = LST_estimator.transform(Xs, ys)

    """

    def __init__(self, n_jobs=None):
        """
         Parameters
         ----------
         n_jobs: int
            n_jobs defaults to None, which means using all CPUs.

         """
        self.T_ = None
        self.classes_ = None
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray):
        """Model training.

         Parameters
         ----------
         X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
         y：ndarry
            Label, shape(n_trials,).

         """
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y == label], axis=0) for label in self.classes_]
        return self

    def transform(self, X: ndarray, y: ndarray):
        """ Obtain transformed source data.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,).

        Returns
        -------
        X : ndarray
            Data after LST conversion, shape(n_trials, n_channels, n_samples).

        """
        X = np.copy(X)
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        Ts = np.zeros_like(X)
        for i, label in enumerate(self.classes_):
            Ts[y == label] = self.T_[i]
        P = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(lst_kernel)(S, T) for S, T in zip(X, Ts)
            )
        )
        X = P @ X
        return X
