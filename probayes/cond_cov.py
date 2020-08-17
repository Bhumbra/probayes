"""
A covariance-matrix-based conditional sampling class.
"""
import numpy as np
import scipy.stats
from probayes.vtypes import isunitsetint, isscalar, uniform

#-------------------------------------------------------------------------------
class CondCov:

  # Protected
  _mean = None # Means 
  _cov = None  # Covariance matrix
  _n = None    # len(means)
  _scom = None # Schur complements
  _inv = None  # Inversion of covariance matrix
  _stdv = None # Conditional standard deviations
  _coef = None # Regression coefficients
  _lims = None  # Recentered limits (Nx2 array)
  _cdfs = None  # cdf-limits

#-------------------------------------------------------------------------------
  def __init__(self, mean, cov, lims):
    self._mean = np.atleast_1d(mean)
    self._cov = np.atleast_2d(cov)
    self._lims = np.atleast_2d(lims) - np.expand_dims(self._mean, -1)
    self._n = len(self._mean)
    self._inv = np.linalg.inv(self._cov)
    assert len(self._cov) == self._n, \
        "Means and covariance matrix incommensurate"
    self._stdv = np.empty(self._n, dtype=float)
    self._coef = [None] * self._n
    for i in range(self._n):
      cov = self._cov[i]
      cov = np.array([cov[j] for j in range(self._n) if j != i])
      self._coef[i] = cov * self._inv[i, i]
      self._stdv[i] = np.sqrt(np.abs(self._cov[i,i] - np.sum(cov * self._coef[i])))

    self._cdfs = np.array([scipy.stats.norm.cdf(lim, loc=0., scale=self._stdv[i]) \
                          for i, lim in enumerate(self._lims)])

#-------------------------------------------------------------------------------
  def interp(self, *args):
    # args in order of mean - one must be a unitsetint
    idx = None
    dmu = np.zeros(self._n, dtype=float)
    for i, arg in enumerate(args):
      if isunitsetint(arg):
        if idx is None:
          idx = i
        else:
          raise ValueError("Only one argument can be interpolated at a time")
      else:
        dmu = arg - self._mean[i]
    assert idx is not None, "No variable specified for interpolation"
    lims = self._cdfs[idx]
    number = list(args[idx])[0]
    cdf = uniform(lims[0], lims[1], number)
    mean = self._mean[idx] + np.sum(self._coef[idx] * dmu)
    return scipy.stats.norm.ppf(cdf, loc=mean, scale=self._stdv[i])

#-------------------------------------------------------------------------------
