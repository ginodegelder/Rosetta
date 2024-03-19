# -*- coding: utf-8 -*-

"""
    Functions to generate covariance function and their inverse
"""

import numpy as np
from scipy import linalg

# =============================================================================
# try:
#     import numba as nb
#     njit = nb.jit(nopython=True)
# except (ImportError, NameError):
#     print("Numba not available")
# =============================================================================

def njit(func):
    return func


@njit
def icovar_diagonal(n, sigma):
    diag = np.ones(n)/sigma**2
    return np.diag(diag)


# @njit
def k_exponential(r, corr_l, gamma=2, truncate=None):
    """
    Exponential kernel for building covariance matrix

    Parameters
    ----------
    r : float
        Distance between two points

    corr_l : float
        Correlation length

    gamma : float
        Exponent

    truncate : float
        Distance at which the kernel is truncated (set to 0)

    Returns
    -------
    k : float
        k(r)
    """
    if isinstance(truncate, (int, float)) and r > float(truncate):
        return 0.
    else:
        return np.exp(- 0.5*(r/corr_l)**gamma)


# @njit
def exponential_covar_1d(n, sigma, corr_l, dx=1, gamma=2, truncate=None):
    """
    Create a covariance matrix with gaussian kernel for a 1D
    regularly-sampled signal.

    Parameters
    ----------
    n : int
        Size of the signal

    sigma : float
        Standard deviation at distance 0.

    corr_l : float
        Correlation length

    dx : float
        Distance between adjacent samples in the signal

    gamma : float
        Value of the exponent in the exponential kernel

    truncate : float
        Distance at which the kernel is truncated (set to 0)

    Returns
    -------
    covar : 2D numpy array
        The covariance matrix
    """
    covar = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            r = abs(int(j-i))*float(dx)
            covar[i, j] = k_exponential(
                r, corr_l, gamma=gamma, truncate=truncate) * sigma**2
    return covar
