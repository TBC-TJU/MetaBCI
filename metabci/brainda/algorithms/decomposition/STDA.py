
"""
    The Spatial-Temporal Discriminant Analysis (STDA) algorithm maximizes
    the discriminability of the projected features between target and non-target classes
    by alternately and synergistically optimizing the spatial and temporal dimensions of the EEG
    in order to learn two projection matrices. Using the learned two projection matrices to
    transform each of the constructed spatial-temporal two-dimensional samples into new one-dimensional
    samples with significantly lower dimensions effectively improves the covariance matrix parameter estimation
    and enhances the generalization ability of the learned classifiers under small training sample sets.

    author: Jin Han

    email: jinhan9165@gmail.com

    Created on: 2022-05

    update log:
        2023/12/08 by Yin ZiFan, promise010818@gmail.com, update code annotation

    Refer: [1] Zhang, Yu, et al. "Spatial-temporal discriminant analysis for ERP-based brain-computer interface."
            IEEE Transactions on Neural Systems and Rehabilitation Engineering 21.2 (2013): 233-243.

    Application: Spatial-Temporal Discriminant Analysis (STDA)

    """

import warnings

import numpy as np
from numpy import ndarray
from scipy import linalg as LA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


def lda_kernel(X1: ndarray, X2: ndarray):
    """
    Linear Discriminant analysis kernel that is appliable to binary problems.

    Parameters
    ----------
    X1: ndarray of shape (n_samples, n_features)
        samples for class 1 (i.e. positive samples)

    X2: ndarray of shape (n_samples, n_features)
        samples for class 2 (i.e. negative samples)

    Returns
    -------
    weight_vec: ndarray of shape (1, n_features)
        weight vector.

    lda_threshold: float

    Note
    ----
    The test samples should be formatted as (n_samples, n_features).
        test sample is positive, if W @ test_sample.T > lda_thold.
        test sample is negative, if W @ test_sample.T <= lda_thold.
    """

    # mean feature vectors
    avg_feats1, avg_feats2 = X1.mean(axis=0, keepdims=True), X2.mean(
        axis=0, keepdims=True
    )

    # within-class scatter matrix
    X1_tmp, X2_tmp = (X1 - avg_feats1), (X2 - avg_feats2)
    Sw = X1_tmp.T @ X1_tmp + X2_tmp.T @ X2_tmp

    weight_vec = (LA.inv(Sw) @ (avg_feats1 - avg_feats2).T).T
    lda_threshold = weight_vec @ (avg_feats1.T + avg_feats2.T) / 2

    return weight_vec, lda_threshold.item()


def lda_proba(test_samples: ndarray, weight_vec: ndarray, lda_threshold: float):
    """Calculate decision value.

    Parameters
    ----------
    test_samples: 2-D, (n_samples, n_features)
    weight_vec: from LDA_kernel.
    lda_threshold: from LDA_kernel.

    Returns
    -------
    proba: ndarray of shape (n_samples,)
    """
    proba = weight_vec @ test_samples.T

    return proba.squeeze()


class STDA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Spatial-Temporal Discriminant Analysis (STDA).
    Note that the parameters naming are exactly the same as in the paper for convenient application.

    Parameters
    ----------
    L: int
        the number of eigenvectors retained for projection matrices.

    max_iter: int, default=400
        Max iteration times.

    eps: float, default=1e-5, also can be 1e-10.
        Error to guarantee convergence.
        Error = norm2(W(n) - W(n-1)), see more details in paper[1].

    Attributes
    ----------
    W1: ndarray of shape (D1, self.L)
        Weight vector. Actually, D1=n_chs.

    W2: ndarray of shape (D2, self.L)
        Weight vector. Actually, D2=n_features.

    iter_times: int
        Iteration times of STDA.

    wf: ndarray of shape (1, L*L)
        Weight vector of LDA after the raw features are projected by STDA.


    References
    ----------
    [1] Zhang, Yu, et al. "Spatial-temporal discriminant analysis for ERP-based brain-computer interface."
            IEEE Transactions on Neural Systems and Rehabilitation Engineering 21.2 (2013): 233-243.

    Tip
    ----
    .. code-block:: python
       :caption: A example using STDA

        import numpy as np
        from metabci.brainda.algorithms.decomposition import STDA
        Xtrain2 = np.random.randint(-10, 10, (100*2, 16, 19))
        y2 = np.hstack((np.ones(100, dtype=int), np.ones(100, dtype=int) * -1))
        Xtest2 = np.random.randint(-10, 10, (4, 16, 19))
        clf3 = STDA()
        clf3.fit(Xtrain2, y2)
        z=clf3.transform(Xtest2)
        print(clf3.transform(Xtest2))
    """

    def __init__(self, L: int = 6, max_iter: int = 400, eps: float = 1e-5):
        self.L = L
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X, y):
        """
        Fit Spatial-Temporal Discriminant Analysis (STDA) model.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_chs, n_features)
           Training data.

        y : array-like of shape (n_samples,)
            Target values. {-1, 1} or {0, 1}

        Returns
        -------
        self: object
              Fitted estimator (i.e. self.W1, self.W2).
        """
        self.classes_ = np.unique(y)
        _, n_chs, n_features = X.shape
        n_classes = len(self.classes_)

        if self.L > n_chs or self.L > n_features:
            raise ValueError(
                "The parameter L must be less than n_sample or n_features."
            )

        if n_classes != 2:
            raise NotImplementedError(
                "Only for binary problem. Multi-class STDA is not tested."
            )

        loc = [
            np.argwhere(y == self.classes_[idx_class]).squeeze()
            for idx_class in range(n_classes)
        ]
        X1, X2 = X[loc[0]], X[loc[1]]  # X1: negative samples. X2: positive samples.

        n_samples_c1, n_samples_c2 = X1.shape[0], X2.shape[0]
        # W1, W2 = np.ones((n_chs, self.L)), np.ones((n_features, self.L))
        W1, W2 = [], []
        W2.append(np.ones((n_features, n_features)))
        self.iter_times = 0
        while 1:
            self.iter_times += 1
            for k in range(1, n_classes + 1):
                if k == 1:
                    Y_mat_c1, Y_mat_c2 = np.matmul(X1, W2[-1]), np.matmul(X2, W2[-1])
                elif k == 2:
                    Y_mat_c1, Y_mat_c2 = np.matmul(W1[-1].T, X1).transpose(
                        (0, 2, 1)
                    ), np.matmul(W1[-1].T, X2).transpose((0, 2, 1))

                Y_bar_c1, Y_bar_c2 = Y_mat_c1.mean(axis=0), Y_mat_c2.mean(axis=0)
                Y_bar_all = ((Y_bar_c1 * n_samples_c1) + (Y_bar_c2 * n_samples_c2)) / (
                    n_samples_c1 + n_samples_c2
                )

                # construct Sb
                y1_tmp, y2_tmp = Y_bar_c1 - Y_bar_all, Y_bar_c2 - Y_bar_all
                Sb = n_samples_c1 * (y1_tmp @ y1_tmp.T) + n_samples_c2 * (
                    y2_tmp @ y2_tmp.T
                )

                # construct Sw
                Y_mat_c1 -= Y_bar_all
                Y_mat_c2 -= Y_bar_all
                Sw = (
                    np.matmul(Y_mat_c1, Y_mat_c1.transpose((0, 2, 1)))
                    + np.matmul(Y_mat_c2, Y_mat_c2.transpose((0, 2, 1)))
                ).sum(axis=0)

                # eig_vals, eig_vecs = LA.eig(np.dot(LA.inv(Sw), Sb))
                eig_vals, eig_vecs = LA.eigh(Sb, Sw)
                loc_idx = eig_vals.argsort()[::-1]  # descending order

                eig_vals = eig_vals[loc_idx]

                if k == 1:
                    W1.append(
                        eig_vecs[:, loc_idx][:, : self.L]
                    )  # return indices in ascending order and reverse
                else:
                    W2.append(eig_vecs[:, loc_idx][:, : self.L])

            # stop criterion
            if self.iter_times >= 4:
                if (
                    LA.norm(W1[-1] - W1[-2], ord=2) < self.eps
                    and LA.norm(W2[-1] - W2[-2], ord=2) < self.eps
                ) or (self.iter_times > self.max_iter):
                    self.W1, self.W2 = W1[-1], W2[-1]

                    f_c1 = np.matmul(np.matmul(self.W1.T, X1), self.W2).reshape(
                        n_samples_c1, -1
                    )
                    f_c2 = np.matmul(np.matmul(self.W1.T, X2), self.W2).reshape(
                        n_samples_c2, -1
                    )

                    self.wf, _ = lda_kernel(f_c2, f_c1)

                    return self

                if self.iter_times > 200:
                    warnings.warn(
                        "The alternating iteration has been performed many times (>200). "
                        "Model may be un-convergence."
                    )

    def transform(self, Xtest):
        """Project data and Get the decision values.

        Parameters
        ----------
        Xtest: ndarray of shape (n_samples, n_features).
            Input test data.

        Returns
        -------
        H_dv: ndarray of shape (n_samples, )
            decision values.
        """
        n_samples = Xtest.shape[0]
        f_hat = np.matmul(np.matmul(self.W1.T, Xtest), self.W2).reshape(n_samples, -1)

        H_dv = f_hat @ self.wf.T

        return H_dv.squeeze()


if __name__ == "__main__":
    clf_stda = STDA(L=6)

    # X = np.random.randn(1080*2, 16, 19)
    X = np.random.randint(-100, 100, (1080 * 2, 16, 19))
    y = np.hstack((np.ones(1080, dtype=int), np.ones(1080, dtype=int) * -1))

    clf_stda.fit(X, y)
