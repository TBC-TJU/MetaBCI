
"""
    Shrinkage Linear Discriminant Analysis (SKLDA) algorithm, through the optimization of
    local features to achieve the purpose ofreducing the dimensionality of the data,
    can improve the small sample problem of the LDA algorithm to some extent.

    author: OrionHan

    email: jinhan9165@gmail.com

    Created on: date (e.g.2022-02-15)

    update log:
        2023/12/08 by Yin ZiFan, promise010818@gmail.com, update code annotation

    Refer: [1] Blankertz, et al. "Single-trial analysis and classification of ERP components—a tutorial."
           NeuroImage 56.2 (2011): 814-825.

    Application:

"""

import numpy as np
from numpy import ndarray
from scipy import linalg as LA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class SKLDA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Shrinkage Linear discriminant analysis (SKLDA) for BCI.

    Attributes
    ----------
    avg_feats1: ndarray of shape (n_features,)
        mean feature vector of class 1.

    avg_feats2: ndarray of shape (n_features,)
        mean feature vector of class 2.

    sigma_c1: ndarray of shape (n_features, n_features)
        empirical covariance matrix of class 1.

    sigma_c2: ndarray of shape (n_features, n_features)
        empirical covariance matrix of class 2.

    D: int, (=n_features)
        the dimensionality of the feature space.

    nu_c1: float
        for sigma penalty calculation in class 1.

    nu_c2: float
        for sigma penalty calculation in class 2.

    classes_: ndarray
        Class labels.

    n_features: int
        Number of features of the training data.

    n_samples_c2: int
        Number of samples in class 2.

    n_samples_c1: int
        Number of samples in class 1.

    Tip
    ----
    .. code-block:: python
       :caption: A example using SKLDA

        import numpy as np
        from metabci.brainda.algorithms.decomposition import SKLDA
        Xtrain = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        Xtest = np.array([[-0.8, -1], [-1.2, -1], [1.2, 1], [0.5, 2]])
        clf2 = SKLDA()
        clf2.fit(Xtrain, y)
        print(clf2.transform(Xtest))
    """

    def __init__(self):
        pass

    def fit(self, X: ndarray, y: ndarray):
        """Train the model, Fit SKLDA.

        Parameters
        ----------
        X1: ndarray of shape (n_samples, n_features)
            samples for class 1 (i.e. positive samples)

        X2: ndarray of shape (n_samples, n_features)
            samples for class 2 (i.e. negative samples)

        X: array-like of shape (n_samples, n_features)
           Training data.

        y : array-like of shape (n_samples,)
            Target values, {-1, 1} or {0, 1}.

        Returns
        -------
        self: object
            Some parameters (sigma_c1, sigma_c2, D) of SKLDA.

        """
        self.classes_ = np.unique(y)
        _, self.n_features = X.shape
        n_classes = len(self.classes_)

        # Extract samples of two classes
        loc = [
            np.argwhere(y == self.classes_[idx_class]).squeeze()
            for idx_class in range(n_classes)
        ]
        X1, X2 = (
            X[loc[1], :],
            X[loc[0], :],
        )  # X1: positive samples. X2: negative samples.

        self.n_samples_c1, self.n_samples_c2 = X1.shape[0], X2.shape[0]
        # n_sum = self.n_samples_c1 + self.n_samples_c2

        # mean feature vectors
        self.avg_feats1, self.avg_feats2 = X1.mean(axis=0, keepdims=True), X2.mean(
            axis=0, keepdims=True
        )

        # within-class scatter matrix
        X1_tmp, X2_tmp = (X1 - self.avg_feats1), (X2 - self.avg_feats2)
        self.sigma_c1 = X1_tmp.T @ X1_tmp
        self.sigma_c2 = X2_tmp.T @ X2_tmp

        # Sw = self.sigma_c1 + self.sigma_c2

        # Shrinkage parameters
        self.D = X1.shape[1]

        return self

    def transform(self, Xtest: ndarray):
        """Project data and Get the decision values.

        Parameters
        ----------
        Xtest: ndarray of shape (n_samples, n_features).
            Input test data.

        Returns
        -------
        proba: ndarray of shape (n_samples,)
            decision values of all test samples.

        """
        # Shrinkage parameters
        self.nu_c1, self.nu_c2 = (
            np.trace(self.sigma_c1) / self.D,
            np.trace(self.sigma_c2) / self.D,
        )

        # --------------------- Estimate lambda-------------------------------#
        # 1. One of the terms in the denominator
        cov2_c1, cov2_c2 = self.sigma_c1**2, self.sigma_c2**2
        sum_sij2_c1 = cov2_c1.sum() - cov2_c1.trace()
        sum_sij2_c2 = cov2_c2.sum() - cov2_c2.trace()

        # 2. Another term in the denominator
        denom_c1 = np.sum((self.sigma_c1.diagonal() - self.nu_c1) ** 2)
        denom_c2 = np.sum((self.sigma_c2.diagonal() - self.nu_c2) ** 2)

        # 3. numerator
        n_samples_test = Xtest.shape[0]
        Xtest_c1, Xtest_c2 = Xtest - self.avg_feats1, Xtest - self.avg_feats2
        z_mat_c1, z_mat_c2 = np.zeros((n_samples_test, self.D, self.D)), np.zeros(
            (n_samples_test, self.D, self.D)
        )
        for idx_feats in range(self.D):
            z_mat_c1[:, idx_feats, :] = np.multiply(
                Xtest_c1, Xtest_c1[:, idx_feats][:, np.newaxis]
            )
            z_mat_c2[:, idx_feats, :] = np.multiply(
                Xtest_c2, Xtest_c2[:, idx_feats][:, np.newaxis]
            )

        numerator_c1 = z_mat_c1.reshape((n_samples_test, -1)).var(axis=1)
        numerator_c2 = z_mat_c2.reshape((n_samples_test, -1)).var(axis=1)

        # lambda
        lambda_c1 = (
            self.n_samples_c1
            / (self.n_samples_c1 - 1) ** 2
            * numerator_c1
            / (sum_sij2_c1 + denom_c1)
        )  # element-wise computation
        lambda_c2 = (
            self.n_samples_c2
            / (self.n_samples_c2 - 1) ** 2
            * numerator_c2
            / (sum_sij2_c2 + denom_c2)
        )
        # --------------------- End ----------------------------------------#

        # estimate covariance
        n_samples_train = self.n_samples_c1 + self.n_samples_c2
        weight_vec = np.empty((n_samples_test, self.D))
        proba = np.zeros(n_samples_test)
        for idx_test in range(n_samples_test):
            sigma_c1_new = (1 - lambda_c1[idx_test]) * self.sigma_c1 + lambda_c1[
                idx_test
            ] * self.nu_c1 * np.eye(self.D)
            sigma_c2_new = (1 - lambda_c2[idx_test]) * self.sigma_c2 + lambda_c2[
                idx_test
            ] * self.nu_c2 * np.eye(self.D)
            Sw_new = sigma_c1_new * (
                self.n_samples_c1 / n_samples_train
            ) + sigma_c2_new * (self.n_samples_c2 / n_samples_train)
            weight_vec[idx_test, :] = (
                LA.inv(Sw_new) @ (self.avg_feats1 - self.avg_feats2).T
            ).T

            proba[idx_test] = weight_vec[idx_test, :] @ Xtest[idx_test, :]

        return proba
