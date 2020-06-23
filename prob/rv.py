""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
import scipy.stats as ss
from prob.base import _P, _V, NOMINAL_VSET, nominal_prob

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
"""

#-------------------------------------------------------------------------------
class RV (_P, _V):

  # Public
  name = "rv"                # Name of the random vsetiable

  # Protected
  _id = None                 # Same as name
  _prob = None               # The probability distribution function
  _prob_args = None          # Arguments of probability distribution function
  _prob_kwds = None          # Keywords of probability distribution function
  _get = None                # Namedtuple

  # Private
  __callable = None          # Flag to denote if prob is callable

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vfun=None, 
                     prsc=None,
                     prob=None,
                     *args,
                     **kwds):
    self.set_name(name)
    self.set_vset(vset, vfun)
    self.set_prsc(prsc)
    self.set_prob(prob, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_name(self, name):
    # Identifier name required
    self.name = name
    assert isinstance(self.name, str), \
        "Mandatory RV name must be a string: {}".format(self.name)
    assert self.name.isidentifier(), \
        "RV name must ba a valid identifier: {}".format(self.name)
    self._id = self.name

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    if isinstance(prob, (float, int, bool)) and \
       isinstance(self._vset, np.ndarray) and \
       np.all(self._vset == np.atleast_1d(NOMINAL_VSET)) and not len(kwds):
      prob, args = nominal_prob, tuple([float(prob)])
    if prob is not None: self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)
    self.__callable = callable(self._prob)
    if self._prob is None:
      if self._vset is None:
        return self.__callable
      elif isinstance(self._vset, (bool, int, float)):
        self._prob = 1.
      elif isinstance(self._vset, np.ndarray):
        nvset = len(self._vset)
        self._prob = np.inf if not nvset else 1. / float(nvset)
      elif isinstance(self._vset, tuple) and len(self._vset) == 2:
        if self._vset[0] == self._vset[1]:
          self._prob = np.inf
        else:
          self._prob = 1. / (float(max(self._vset)) - float(min(self._vset)))
    elif not self.__callable:
      self._prob = np.atleast_1d(self._prob)
      assert not len(self._prob_args), "Optional arguments requires callable prob"
      assert not len(self._prob_kwds), "Optional keywords requires callable prob"
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob),len(self._vset))
    return self.__callable
   
#-------------------------------------------------------------------------------
  def eval_prob(self, samples=None, rsc=None):

    # Callable and non-callable evaluations
    probs = self._prob
    if self.__callable:
      probs = probs(samples, *self._prob_args, **self._prob_kwds)
    elif isinstance(samples, np.ndarray) and type(probs) is float:
      probs = np.tile(probs, samples.shape)
    else:
      probs = np.atleast_1d(probs).astype(float)

    # If not callable and samples is an array, check not outside variable set
    if not self.__callable and isinstance(probs, np.ndarray):
      if probs.ndim < 2 and isinstance(samples, np.ndarray):
        probs_size = probs.size
        if probs_size == samples.size:
          if isinstance(self._vset, np.ndarray):
            outside = np.array([samp not in self._vset for samp in samples], dtype=bool)
            probs[outside] = 0.
          elif isinstance(self._vset, tuple) and len(self._vset) == 2:
            lo, hi = self.get_bounds(False)
            outside = np.logical_or(samples<lo, samples>hi)
            probs[outside] = 0.

    return self.rescale(probs, rsc)

#-------------------------------------------------------------------------------
  def __call__(self, samples=None, **kwds):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    mutable = isinstance(samples, (np.ndarray, list))
    samps = self.eval_samp(samples)
    probs = self.eval_prob(samps)
    self._get = collections.namedtuple(self._id, ['samp', 'prob'], **kwds)

    # Reciprocate samps evaluated using vfun or just recall if mutable array
    vfun = self.ret_vfun()
    if mutable:
      samps = samples
      if isinstance(samps, list):
        samps = np.atleast_1d(samps) if not self._vtype else \
                np.atleast_1d(samps).astype(self._vtype)
    elif vfun:
      samps = vfun[1](samps)
    return self._get(samps, probs)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self.name + "'"

#-------------------------------------------------------------------------------
