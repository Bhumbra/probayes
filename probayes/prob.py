"""
A probability class supporting probability distributions without specification 
of a variable set.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import warnings
import numpy as np
import scipy.stats
from probayes.vtypes import isscalar
from probayes.pscales import eval_pscale, rescale, iscomplex
from probayes.func import Func

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
class Prob:

  # Protected
  _prob = None      # Probability distribution function
  _pscale = None    # Probability type (can be a scipy.stats.dist object)
  _pfun = None      # 2-length tuple of cdf/icdf

  # Private
  __pset = None     # Set of pdfs/logpdfs/cdfs/icdfs
  __scalar = None   # Flag for being a scalar
  __callable = None # Flag for callable function

#-------------------------------------------------------------------------------
  def __init__(self, prob=None, pscale=None, *args, **kwds):
    self.set_prob(prob, pscale, *args, **kwd)

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, pscale=None, *args, **kwds):
    self._pfun = None
    pset = prob if is_scipy_stats_dist(prob) else None
    self.__scalar = None
    self.__callable = None

    # Handle SciPy distributions and scalars, 
    if pset is not None:
      prob = None        # needed to pass set_pset assertion
      self.__pset = pset # needed to pass set_pscale assertion
    elif isscalar(prob): 
      prob = float(prob)
    self._prob = prob 

    # Set pscalar before pset
    self.set_pscale(pscale) # this defaults self._pfun

    # Create functional interface for prob
    if pset is not None:
      self.set_pset(pset, *args, **kwds)
    elif  self._prob is not None:
      self._prob = Func(self._prob, *args, **kwds)
    else:
      return

    # Set pscale and distinguish between non-callable and callable self._prob
    self.__isscalar = self._prob.ret_isscalar()
    self.__callable = self._prob.ret_callable()

    return self.__callable

#-------------------------------------------------------------------------------
  def set_pscale(self, pscale=None):
    """
    Positive denotes a normalising coefficient.
    If zero or negative, denotes log probability offset ('log' or 'ln' means '0.0').
    May also be scipy.stats.distribution variable type to set everything else.
    """
    self._pscale = eval_pscale(pscale)

    # Probe pset to set functions based on pscale setting
    if self.__pset is None:
      if self._pscale != 1.:
        assert self._prob is not None, \
            "Cannot specify pscale without setting prob"
      self.set_pfun()

    return self._pscale

#-------------------------------------------------------------------------------
  def set_pset(self, pset, *args, **kwds):
    self.__pset = pset if is_scipy_stats_dist(pset) else None
    if self.__pset is None:
      return
    assert self._prob is None, "Cannot use scipy.stats.dist while also setting prob"
    if not iscomplex(self._pscale):
      if hasattr(self.__pset, 'pdf'):
        self._prob = Func(self.__pset.pdf, *args, **kwds)
      elif hasattr(self.__pset, 'pmf'):
        self._prob = Func(self.__pset.pmf, *args, **kwds)
      else: 
        warnings.warn("Cannot find probability function for {}"\
                      .format(self.__pset))
    else:
      if hasattr(self.__pset, 'logpdf'):
        self._prob = Func(self.__pset.logpdf, *args, **kwds)
      elif hasattr(self.__pset, 'logpmf'):
        self._prob = Func(self.__pset.logpmf, *args, **kwds)
      else: 
        warnings.warn("Cannot find log probability function for {}"\
                      .format(self.__pset))
    if hasattr(self.__pset, 'cdf') and  hasattr(self.__pset, 'ppf'):
      self.set_pfun((self.__pset.cdf, self.__pset.ppf), *args, **kwds)
    else:
      warnings.warn("Cannot find cdf and ppf functions for {}"\
                    .format(self._pscale))
      self.set_pfun()

#-------------------------------------------------------------------------------
  def ret_pset(self):
    return self.__pset

#-------------------------------------------------------------------------------
  def set_pfun(self, pfun=None, *args, **kwds):
    self._pfun = pfun
    if self._pfun is None:
      return 
    
    message = "Input pfun be a two-sized tuple of callable functions"
    assert isinstance(self._pfun, tuple), message
    assert len(self._pfun) == 2, message
    assert callable(self._pfun[0]), message
    assert callable(self._pfun[1]), message
    self._pfun = Func(self._pfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def ret_pfun(self, index=None):
    if self._pfun is None or index is None:
      return self._pfun
    return self._pfun[index]

#-------------------------------------------------------------------------------
  def ret_prob(self):
    return self._prob

#-------------------------------------------------------------------------------
  def ret_callable(self):
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    return self.__isscalar

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    return self._pscale

#-------------------------------------------------------------------------------
  def ret_pset(self):
    return self.__pset

#-------------------------------------------------------------------------------
  def rescale(self, probs, **kwds):
    if 'pscale' not in kwds:
      return probs
    return rescale(probs, self._pscale, kwds['pscale'])

#-------------------------------------------------------------------------------
  def eval_prob(self, *args, **kwds):
    """ keys can include pscale """

    # Callable and non-callable evaluations
    probs = self._prob
    if self.__callable:
      probs = probs(*args)
    else:
      assert not len(args), \
          "Cannot evaluate from values from an uncallable probability function"
      probs = probs()
    if 'pscale' in kwds:
      return self.rescale(probs, kwds['pscale'])
    return probs

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    return eval_prob(*args, **kwds)

#-------------------------------------------------------------------------------
