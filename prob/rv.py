""" Random variable module """

#-------------------------------------------------------------------------------
import numpy as np
import scipy.stats as ss

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
"""

#-------------------------------------------------------------------------------
NUMPY_DTYPES = {
                 np.dtype('bool'): bool,
                 np.dtype('int'): int,
                 np.dtype('int32'): int,
                 np.dtype('int64'): int,
                 np.dtype('float'): float,
                 np.dtype('float32'): float,
                 np.dtype('float64'): float,
               }
#-------------------------------------------------------------------------------
def nominal_prob(x, p):
  x, p = np.atleast_1d(x).astype(bool), float(p)
  prob = np.tile(1.-p, x.shape)
  prob[x] = p
  return prob

#-------------------------------------------------------------------------------
NOMINAL_VSET = [False, True]
NOMINAL_PROB = nominal_prob

#-------------------------------------------------------------------------------
class RV:

  # Public
  name = "rv"                # Name of the random vsetiable

  # Protected
  _vset = None               # Variable set (array or 2-length tuple range)
  _vtype = None              # Variable data type (ideally bool, int, or float)
  _ptype = None              # Probability type (future support)
  _prob = None               # The probability distribution function
  _prob_args = None          # Arguments of probability distribution function
  _prob_kwds = None          # Keywords of probability distribution function

  # Private
  __callable = None          # Flag to denote if prob is callable

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vtype=None, 
                     ptype=None,
                     prob=None,
                     *args,
                     **kwds):
    self.set_vset(name, vset)
    self.set_ptype(ptype)
    self.set_prob(prob, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_vset(self, name, vset=None, vtype=None):

    # Name required
    self.name = str(name)
    assert len(self.name), "Random variable name mandatary"

    # Default vset to nominal, and convert a set,list to an array
    if vset is None: vset = NOMINAL_VSET
    self._vset = vset
    if isinstance(self._vset, (set, list)):
      self._vset = np.atleast_1d(self._vset) if isinstance(self._vset, list) \
                   else np.sort(np.atleast_1d(self._vset))
    elif isinstance(self._vset, tuple):
      assert len(self._vset) == 2,\
          "Tuple vset must be of length 2, not {}".format(len(self._vset))
    elif isinstance(self,_vset, range):
      self._vset = np.arange(self._vset.start, self._vset,stop, self._vset.step,
                             dtype=int)

    # If dtype specified convert vset np.array if required
    if vtype: 
      self._vtype = vtype
      if isinstance(self._vset, np.ndarray):
        self._vset = self._vset.astype(self._vtype)
      return self._vtype

    # Attempt to detect vtype if not specified
    if isinstance(self._vset, (bool, int, float)):
      self._vtype = type(self._vset)
    elif isinstance(self._vset, tuple):
      self._vtype = float
    elif isinstance(self._vset, np.ndarray):
      self._vtype = NUMPY_DTYPES.get(self._vset.dtype, None)
    return self._vtype

#-------------------------------------------------------------------------------
  def set_ptype(self, ptype=None):
    self._ptype = ptype
    return self._ptype

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    if isinstance(prob, (float, int, bool)) and \
       isinstance(self._vset, np.ndarray) and \
       np.all(self._vset == np.atleast_1d(NOMINAL_VSET)) and not len(kwds):
      prob, args = NOMINAL_PROB, tuple([float(prob)])
    if prob is not None: self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)
    self.__callable = callable(self._prob)
    if self._prob is None:
      return self.__callable
    elif not self.__callable:
      self._prob = np.atleast_1d(self._prob)
      assert not len(self._prob_args), "Optional arguments requires callable prob"
      assert not len(self._prob_kwds), "Optional keywords requires callable prob"
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob),len(self._vset))
    return self.__callable

#-------------------------------------------------------------------------------
  def sample(self, samples=None):
    # Negative samples denotes randomised

    # Convert a list of samples to a numpy array
    if isinstance(samples, list):
      samples = np.atleast_1d(samples) if not self._vtype else \
                np.atleast_1d(samples).astype(self._vtype)

    # Integer samples n values
    elif samples is None or type(samples) is int:
      if samples is None:
        assert not isinstance(self._vset, tuple),\
            "Samples must be specified for Vset: {}".format(self._vset)
        samples = len(self._vset)

      # Non-continuous support sets
      if not isinstance(self._vset, tuple):
        divisor = len(self._vset)
        if samples >= 0:
          indices = np.arange(samples, dtype=int) % divisor
        else:
          indices = np.random.permutation(-samples, dtype=int) % divisor
        samples = self._vset[indices]

      else:
        assert np.all(np.isfinite(self._vset)), \
            "Cannot evaluate {} samples for bounds: {}".format(samples, self._vset)
        lo, hi = float(np.min(self._vset)), float(np.max(self._vset))
        if samples >= 0:
          samples = np.sort(np.random.uniform(lo, hi, size=samples))
        else:
          samples = np.linspace(lo, hi, -samples)

    else:
      raise TypeError("Ambiguous samples type: ".format(type(samples)))
    return samples

#-------------------------------------------------------------------------------
  def __call__(self, samples=None):
    samples = self.sample(samples)

    # Output probs
    probs = self._prob
    if self.__callable:
      probs = probs(samples, *self._prob_args, **self._prob_kwds)
    else:
      probs = np.atleast_1d(probs).astype(float)
    return samples, probs

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": (" + self.name + ")"

#-------------------------------------------------------------------------------
