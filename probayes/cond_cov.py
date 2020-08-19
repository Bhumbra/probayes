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
  _stdv = None # Conditional standard deviations
  _coef = None # Regression coefficients
  _lims = None  # Recentered limits (Nx2 array)
  _cdfs = None  # cdf-limits

#-------------------------------------------------------------------------------
  def __init__(self, mean, cov, lims):
    self._mean = np.atleast_1d(mean)
    self._cov = np.atleast_2d(cov)
    self._inv = np.linalg.inv(self._cov)
    self._lims = np.atleast_2d(lims) - np.expand_dims(self._mean, -1)
    self._n = len(self._mean)
    assert len(self._cov) == self._n, \
        "Means and covariance matrix incommensurate"
    self._stdv = np.empty(self._n, dtype=float)
    self._coef = [None] * self._n
    for i in range(self._n):
      ll  = np.delete(self._cov[:, i].reshape([self._n, 1]), (i), axis=0)
      ru  = np.delete(self._cov[i, :].reshape([1, self._n]), (i), axis=1)
      cov = np.delete(np.delete(self._cov, (i), axis=1), (i), axis=0)
      self._coef[i] = ru.dot(np.linalg.inv(cov))
      self._stdv[i] = np.sqrt(self._cov[i, i] - float(self._coef[i].dot(ll)))
    self._cdfs = np.array([scipy.stats.norm.cdf(lim, loc=0., scale=self._stdv[i]) \
                          for i, lim in enumerate(self._lims)])

#-------------------------------------------------------------------------------
  def interp(self, *args, cond_pdf=False):
    # args in order of mean - one must be a unitsetint
    idx = None
    dmu = np.empty((self._n, 1), dtype=float)
    vals = [arg for arg in args]
    for i, arg in enumerate(args):
      if isunitsetint(arg):
        if idx is None:
          idx = i
        else:
          raise ValueError("Only one argument can be interpolated at a time")
      else:
        dmu[i] = arg - self._mean[i]
    assert idx is not None, "No variable specified for interpolation"
    dmu = np.delete(dmu, (idx), axis=0)
    lims = self._cdfs[idx]
    number = list(args[idx])[0]
    cdf = uniform(lims[0], lims[1], number)
    mean = self._mean[idx] + float(self._coef[idx].dot(dmu))
    stdv = self._stdv[idx]
    vals = scipy.stats.norm.ppf(cdf, loc=mean, scale=stdv)
    if not cond_pdf:
      return vals
    return vals, scipy.stats.norm.pdf(vals, loc=mean, scale=stdv)


#-------------------------------------------------------------------------------
