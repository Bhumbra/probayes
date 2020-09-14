"""
A probability class supporting probability distributions without specification 
of a variable set.
"""

#-------------------------------------------------------------------------------
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
  """ Returns if argument belongs to scipy.stats.continuous """
  return isinstance(arg, tuple(scipy_stats_cont))

#-------------------------------------------------------------------------------
def is_scipy_stats_dist(arg, scipy_stats_dist=SCIPY_STATS_DIST):
  """ Returns if argument belongs to scipy.stats.continuous or discrete """
  return isinstance(arg, tuple(scipy_stats_dist))

#-------------------------------------------------------------------------------
class Prob:
  """ A probability is quantification of degrees of belief concerning outcomes.
  Typically these outcomes are defined over the domains of one or more variables. 
  Since this is not a requirement, this class is not abstract, but it is 
  nevertheless not so useful as probayes.RV if instantiated directly. 
  This class can be used to define a probability distribution.

  :example:
  >>> import scipy.stats
  >>> import probayes as pb
  >>> norm = pb.Prob(scipy.stats.norm)
  >>> print(norm(0.))
  0.3989422804014327
  """

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
    """ Initialises the probability and pscale (see set_prob). """
    self.set_prob(prob, pscale, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, pscale=None, *args, **kwds):
    """ Sets the probability and pscale with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param pscale: represents the scale used to represent probabilities.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.

    See set_pscale() for explanation of how pscale is used.
    """
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

#-------------------------------------------------------------------------------
  def set_pscale(self, pscale=None):
    """ Sets the probability scaling constant used for probabilities.

    :param pscale: can be None, a real number, or a complex number, or 'log'

       if pscale is None (default) the normalisation constant is set as 1.
       if pscale is real, this defines the normalisation constant.
       if pscale is complex, this defines the offset for log probabilities.
       if pscale is 'log', this denotes a logarithmic scale with an offset of 0.

    :return: pscale (either as a real or complex number)
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
    """ Sets a set of probability functions if prob is a scipy.stats object.
    Normally this function should not require calling if set_prob is set.
    """
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
    """ Returns object set by set_pset() """
    return self.__pset

#-------------------------------------------------------------------------------
  def set_pfun(self, pfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_prob(). It is necessary to set these
    functions if sampling variables with non-flat distributions.

    :param pfun: two-length tuple of callable functions
    :param *args: arguments to pass to pfun functions
    :param **kwds: keywords to pass to pfun functions
    """
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
    """ Returns object set by set_pfun() """
    if self._pfun is None or index is None:
      return self._pfun
    return self._pfun[index]

#-------------------------------------------------------------------------------
  def ret_prob(self):
    """ Returns object set by set_prob() """
    return self._prob

#-------------------------------------------------------------------------------
  def ret_callable(self):
    """ Returns whether object set by set_prob() is callable """
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    """ Returns whether prob is a scalar """
    return self.__isscalar

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    """ Returns the real or complex scaling constant set for pscale """
    return self._pscale

#-------------------------------------------------------------------------------
  def rescale(self, probs, **kwds):
    """ Returns a rescaling of probs from current pscale to the values according
    to the keyword pscale=new_pscale. """
    if 'pscale' not in kwds:
      return probs
    return rescale(probs, self._pscale, kwds['pscale'])

#-------------------------------------------------------------------------------
  def eval_prob(self, *args, **kwds):
    """ Evaluates the probability inputting optional args for callable cases

    :param *args: optional arguments for callable probability objects.
    :param **kwds: optional arguments to include pscale for rescaling.

    :return: evaluated probabilities
    """
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
    """ See eval_prob() """
    return self.eval_prob(*args, **kwds)

#-------------------------------------------------------------------------------
