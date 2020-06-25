"""
An abstract probability class supporting probability distributions without
any specification of a variable alphabet set.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import warnings
import numpy as np
import scipy.stats

#-------------------------------------------------------------------------------
NEARLY_POSITIVE_ZERO = 1.175494e-38
NEARLY_NEGATIVE_INF = -3.4028236e38
NEARLY_POSITIVE_INF =  3.4028236e38
LOG_NEARLY_POSITIVE_INF = np.log(NEARLY_POSITIVE_INF)
SCIPY_STATS_DISTS = (scipy.stats.rv_continuous, scipy.stats.rv_discrete)

#-------------------------------------------------------------------------------
def is_scipy_stats_dist(ptype, scipy_stats_dists=SCIPY_STATS_DISTS):
  return isinstance(ptype, scipy_stats_dists)

#-------------------------------------------------------------------------------
def nominal_prob(x, p):
  x, p = np.atleast_1d(x), float(p)
  prob = np.tile(1.-p, x.shape)
  prob[x.astype(bool)] = p
  return prob

#-------------------------------------------------------------------------------
def log_prob(prob):
  logs = np.tile(NEARLY_NEGATIVE_INF, prob.shape)
  ok = prob >= NEARLY_POSITIVE_ZERO
  logs[ok] = np.log(prob[ok])
  return logs

#-------------------------------------------------------------------------------
def exp_logs(logs):
  prob = np.tile(NEARLY_POSITIVE_INF, logs.shape)
  ok = logs <= LOG_NEARLY_POSITIVE_INF
  prob[ok] = np.exp(logs[ok])
  return prob

#-------------------------------------------------------------------------------
def log_offset(ptype=None):
  if ptype is None:
    return 0.
  if isinstance(ptype, str):
    return float(ptype)
  if ptype <= 0.:
    return ptype
  return np.log(sc)

#-------------------------------------------------------------------------------
def rescale(prob, *args):
  ptype, rtype = None, None
  if len(args) == 0: 
    return prob
  elif len(args) ==  1: 
    rtype = args[0]
  else: 
    ptype, rtype = args[0], args[1]
  if ptype == rtype:
    return prob
  if ptype in ['ln', 'log']:
    ptype = str(float(0))
  elif type(ptype) is float and ptype <= 0.:
    ptype = str(abs(ptype))
  if rtype in ['ln', 'log']:
    rtype = str(float(0))
  elif type(rtype) is float and rtype <= 0.:
    rtype = str(abs(rtype))
  if ptype == rtype:
    return prob
  
  p_log = isinstance(ptype, str)
  r_log = isinstance(rtype, str)

  # Support non-logarithmic conversion (maybe used to avoid logging zeros)
  if not p_log and not r_log:
    coef = ptype / rtype
    if coef == 1.:
      return prob
    else:
      return coef * prob

  # For floating point precision, perform other operations in log-space
  if not p_log: prob = log_prob(prob)
  d_offs = log_offset(ptype) - log_offset(rtype)
  if d_offs != 0.: prob = prob + d_offs
  if r_log:
    return prob
  return exp_logs(prob)

#-------------------------------------------------------------------------------
class _Prob (ABC):

  # Protected
  _prob = None      # Probability distribution function
  _ptype = None     # Probability type (can be a scipy.stats.dist object)
  _prob_args = None #
  _prob_kwds = None #
  _pfun = None      # 2-length tuple of cdf/icdf
  _pfun_args = None
  _pfun_kwds = None

  # Private
  __pset = None     # Set of pdfs/logpdfs/cdfs/icdfs
  __callable = None # Flag for callable function

#-------------------------------------------------------------------------------
  def __init__(self, prob=None, ptype=None, *args, **kwds):
    self.set_prob(prob, ptype, *args, **kwd)

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None, *args, **kwds):
    self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)
    self.__pset = None

    # Handle SciPy distributions first
    if is_scipy_stats_dist(self._prob):
      self.__pset = self._prob
      self._prob = None

    # Now nominal probabilities
    elif isinstance(prob, (float, int, bool)) and ptype is None and \
         not len(args) and not len(kwds):
      self._prob, self._prob_args = nominal_prob, tuple([float(prob)])

    # And the rest
    else:
      self._prob = prob;

    # Set ptype and distinguish between non-callable and callable self._prob
    self.set_ptype(ptype) # this defaults self._pfun
    self.__callable = callable(self._prob)
    if self._prob is not None and not self.__callable:
      if not isinstance(self._prob, np.ndarray) or self._prob.ndim < 1:
        self._prob = np.atleast_1d(self._prob)
      assert not len(self._prob_args), "Optional arguments requires callable prob"
      assert not len(self._prob_kwds), "Optional keywords requires callable prob"
    return self.ret_callable()

#-------------------------------------------------------------------------------
  def set_ptype(self, ptype=None):
    """
    Positive denotes a normalising coefficient.
    If a string, denotes log probability offset ('log' or 'ln' means '0.0').
    May also be scipy.stats.distribution variable type to set everything else.
    """
    self._ptype = str(float(0)) if ptype in ['log', 'ln'] else ptype
    if self._ptype is float and self._ptype < 0.:
      self._ptype = str(self._ptype)

    # Probe pset to set functions based on ptype setting
    if self.__pset:
      assert self._prob is None, "Cannot use scipy.stats.dist while also setting prob"
      if not isinstance(self._ptype, str):
        if hasattr(self.__pset, 'pdf'):
          self._prob = self.__pset.pdf
        elif hasattr(self.__pset, 'pmf'):
          self._prob = self.__pset.pmf
        else: 
          warnings.warn("Cannot find probability function for {}"\
                        .format(self.__pset))
      else:
        if hasattr(self.__pset, 'logpdf'):
          self._prob = self.__pset.logpdf
        elif hasattr(self.__pset, 'logpmf'):
          self._prob = self.__pset.logpmf
        else: 
          warnings.warn("Cannot find log probability function for {}"\
                        .format(self.__pset))
      if hasattr(self.__pset, 'cdf') and  hasattr(self.__pset, 'ppf'):
        self.set_pfun((self.__pset.cdf, self.__pset.ppf), 
                      *self._prob_args, **self._prob_kwds)
      else:
        warnings.warn("Cannot find cdf and ppf functions for {}"\
                      .format(self._ptype))
        self.set_pfun()
    else:
      if self._ptype is not None:
        assert self._prob is not None, "Cannot specify ptype without setting prob"
      self.set_pfun()

    return self._ptype

#-------------------------------------------------------------------------------
  def set_pfun(self, pfun=None, *args, **kwds):
    self._pfun = pfun
    self._pfun_args = tuple(args)
    self._pfun_kwds = dict(kwds)
    if self._pfun is not None:
      message = "Input pfun be a two-sized tuple of callable functions"
      assert isinstance(self._pfun, tuple), message
      assert len(self._pfun) == 2, message
      assert callable(self._pfun[0]), message
      assert callable(self._pfun[1]), message

#-------------------------------------------------------------------------------
  def ret_callable(self):
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def ret_pset(self):
    return self.__pset

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None, **kwds):
    """ keys can include ptype """

    # Callable and non-callable evaluations
    prob = self._prob
    if self.__callable:
      prob = prob(values, *self._prob_args, **self._prob_kwds)
    else:
      assert values is None, \
          "Cannot evaluate from values from an uncallable probability function"
    if 'ptype' in kwds:
      return self.rescale(probs, kwds['ptype'])
    return prob

#-------------------------------------------------------------------------------
  def rescale(self, probs, **kwds):
    if 'ptype' not in kwds:
      return probs
    return rescale(probs, self._ptype, kwds['ptype'])

#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
