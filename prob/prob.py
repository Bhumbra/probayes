"""
An abstract probability class supporting probability distributions without
any specification of a variable alphabet set.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import warnings
import numpy as np
import scipy.stats
from prob.ptypes import eval_ptype, rescale

#-------------------------------------------------------------------------------
SCIPY_STATS_CONT = {scipy.stats.rv_continuous}
SCIPY_STATS_DISC = {scipy.stats.rv_discrete}
SCIPY_STATS_DIST = SCIPY_STATS_CONT.union(SCIPY_STATS_DISC)

#-------------------------------------------------------------------------------
def is_scipy_stats_cont(arg, scipy_stats_cont=SCIPY_STATS_CONT):
  return isinstance(arg, tuple(scipy_stats_cont))

#-------------------------------------------------------------------------------
def is_scipy_stats_dist(arg, scipy_stats_dist=SCIPY_STATS_DIST):
  return isinstance(arg, tuple(scipy_stats_dist))

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
  __scalar = None
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
    self.__scalar = None

    # Handle SciPy distributions first
    if is_scipy_stats_dist(self._prob):
      self.__pset = self._prob
      self._prob = None

    # Then scalars probabilities
    elif isinstance(prob, (float, int, bool)): 
      self._prob = float(self._prob)

    # And the rest
    else:
      self._prob = prob;

    # Set ptype and distinguish between non-callable and callable self._prob
    self.set_ptype(ptype) # this defaults self._pfun
    self.__callable = callable(self._prob)
    if self.__callable:
      self.__scalar = False
    elif self._prob is not None:
      self.__scalar = np.isscalar(self._prob)
      assert not len(self._prob_args), "Optional arguments requires callable prob"
      assert not len(self._prob_kwds), "Optional keywords requires callable prob"

      # Convert non-scalar foreign types to 1D Numpy array
      if isinstance(self._prob, (list, tuple, np.ndarray)) and not self.__scalar:
          self._prob = np.atleast_1d(self._prob)

    return self.__callable

#-------------------------------------------------------------------------------
  def set_ptype(self, ptype=None):
    """
    Positive denotes a normalising coefficient.
    If zero or negative, denotes log probability offset ('log' or 'ln' means '0.0').
    May also be scipy.stats.distribution variable type to set everything else.
    """
    self._ptype = eval_ptype(ptype)

    # Probe pset to set functions based on ptype setting
    if self.__pset:
      assert self._prob is None, "Cannot use scipy.stats.dist while also setting prob"
      if not np.complex(self._ptype):
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
    elif self._ptype != 1.:
      assert self._prob is not None, "Cannot specify ptype without setting prob"
      self.set_pfun()

    return self._ptype

#-------------------------------------------------------------------------------
  def ret_pset(self):
    return self.__pset

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
  def ret_scalar(self):
    return self.__scalar

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
