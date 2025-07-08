# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
from functools import partial
from typing import Union, Optional, Callable
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from joblib import Parallel, delayed
from scipy.linalg import eigh

estimator = Callable[[ndarray], ndarray]


def isPD(B: ndarray) -> bool:
    """Returns true when input matrix is positive-definite, via Cholesky decompositon method.

    Parameters
    ----------
    B : ndarray
        Any matrix, shape (N, N)

    Returns
    -------
    bool
        True if B is positve-definite.

    Notes
    -----
        Use numpy.linalg rather than scipy.linalg. In this case, scipy.linalg has unpredictable behaviors.
    """

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A: ndarray) -> ndarray:
    """Find the nearest positive-definite matrix to input.

    Parameters
    ----------
    A : ndarray
        Any square matrxi, shape (N, N)

    Returns
    -------
    A3 : ndarray
        positive-definite matrix to A

    Notes
    -----
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1]_, which
    origins at [2]_.

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988):
           https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    print("Replace current matrix with the nearest positive-definite matrix.")

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `numpy.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    eye = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += eye * (-mineig * k**2 + spacing)
        k += 1

    return A3


def _lwf(X: ndarray) -> ndarray:
    """Wrapper for sklearn ledoit wolf covariance estimator.

    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).

    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X: ndarray) -> ndarray:
    """Wrapper for sklearn oas covariance estimator.

    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).

    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C, _ = oas(X.T)
    return C


def _cov(X: ndarray) -> ndarray:
    """Wrapper for sklearn sample covariance estimator.

    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).

    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C = empirical_covariance(X.T)
    return C


def _mcd(X: ndarray) -> ndarray:
    """Wrapper for sklearn mcd covariance estimator.

    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).

    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    _, C, _, _ = fast_mcd(X.T)
    return C


estimators = {
    "cov": _cov,
    "lwf": _lwf,
    "oas": _oas,
    "mcd": _mcd,
}


def _check_est(est: Union[str, estimator]) -> estimator:
    """Check if a given estimator is valid.

    Parameters
    ----------
    est : callable object or str
        Could be the name of estimator or a callable estimator itself.

    Returns
    -------
    est: callable object
        A callable estimator.
    """
    if callable(est):
        pass
    elif est in estimators.keys():
        est = estimators[est]
    else:
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function"""
            % (est, (" , ").join(estimators.keys()))
        )
    return est


def covariances(
    X: ndarray, estimator: Union[str, estimator] = "cov", n_jobs: int = 1
) -> ndarray:
    """Estimation of covariance matrix.

    Parameters
    ----------
    X : ndarray
        EEG signal, shape (..., n_channels, n_samples).
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). For regularization,
        consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int or None, optional
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).

    Returns
    -------
    covmats : ndarray
        covariance matrices, shape (..., n_channels, n_channels)

    See Also
    --------
    covariances_erp
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    shape = X.shape
    X = np.reshape(X, (-1, shape[-2], shape[-1]))

    parallel = Parallel(n_jobs=n_jobs)
    est = _check_est(estimator)
    covmats = parallel(delayed(est)(x) for x in X)

    covmats = np.reshape(covmats, (*shape[:-2], shape[-2], shape[-2]))
    return covmats


class Covariance(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix.

    Parameters
    ----------
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). For regularization,
        consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int or None, optional
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).

    See Also
    --------
    ERPCovariance
    """

    def __init__(self, estimator="cov", n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Not used, only for compatibility with sklearn API.

        Parameters
        ----------
        X : ndarray
            EEG signal, shape (..., n_channels, n_samples).
        y : ndarray
            Labels.

        Returns
        -------
        self : Covariance instance
            The Covariance instance.
        """
        return self

    def transform(self, X):
        """Transform EEG to covariance matrix.

        Parameters
        ----------
        X : ndarray
            EEG signal, shape (..., n_channels, n_samples).

        Returns
        -------
        covmats : ndarray
            Estimated covariances, shape (..., n_channels, n_channels)
        """
        covmats = covariances(X, estimator=self.estimator, n_jobs=self.n_jobs)
        return covmats


def matrix_operator(
    Ci: ndarray, operator: estimator, n_jobs: Optional[int] = None
) -> ndarray:
    """Apply operator to any matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive definite matrix.
    operator : callable object
        Operator function or callable object.
    n_jobs: int, optional
        the number of jobs to use.

    Returns
    -------
    Co : ndarray
        Operated matrix.

    Raises
    ------
    ValueError
        If Ci is not positive definite.

    Notes
    -----
    .. math::
        \mathbf{Ci} = \mathbf{V} \left( \mathbf{\Lambda} \\right) \mathbf{V}^T \\\\
        \mathbf{Co} = \mathbf{V} operator\left( \mathbf{\Lambda} \\right) \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """

    def _single_matrix_operator(Ci: ndarray, operator: estimator) -> ndarray:
        eigvals, eigvects = eigh(Ci, check_finite=False)
        eigvals = np.diag(operator(eigvals))
        Co = eigvects @ eigvals @ eigvects.T
        return Co

    ori_shape = Ci.shape
    Ci = Ci.reshape((-1, *ori_shape[-2:]))
    Co = Parallel(n_jobs=n_jobs)(
        delayed(_single_matrix_operator)(C, operator) for C in Ci
    )
    Co = np.stack(Co)
    Co = Co.reshape((*ori_shape,))
    return Co


def sqrtm(Ci: ndarray, n_jobs: Optional[int] = None):
    """Return the matrix square root of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Square root matrix of Ci.

    Notes
    -----
    .. math::
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.sqrt, n_jobs=n_jobs)


def logm(Ci: ndarray, n_jobs: Optional[int] = None):
    """Return the matrix logrithm of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Logrithm matrix of Ci.

    Notes
    -----
    .. math::
        \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.log, n_jobs=n_jobs)


def expm(Ci: ndarray, n_jobs: Optional[int] = None):
    """Return the matrix exponential of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Exponential matrix of Ci.

    Notes
    -----
    .. math::
        \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.exp, n_jobs=n_jobs)


def invsqrtm(Ci: ndarray, n_jobs: Optional[int] = None):
    """Return the inverse matrix square root of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Inverse matrix square root of Ci.

    Notes
    -----
    .. math::
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """

    def isqrt(x):
        return 1.0 / np.sqrt(x)

    return matrix_operator(Ci, isqrt, n_jobs=n_jobs)


def powm(Ci: ndarray, alpha: float, n_jobs: Optional[int] = None):
    """Return the matrix power of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.
    alpha : float
        Exponent.

    Returns
    -------
    ndarray
        Power matrix of Ci.

    Notes
    -----
    .. math::
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    power = partial(lambda x, alpha=None: x**alpha, alpha=alpha)
    return matrix_operator(Ci, power, n_jobs=n_jobs)
