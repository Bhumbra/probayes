""" Random variable module """

import numpy as np

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every possible
realisation defined over the alphabet set $A_x$ with marginal probabilities P_x.
"""
#-------------------------------------------------------------------------------
class RV:

  # Protected
  _var = range(0, 1)   # Variable alphabet set (expressed as a numpy vector or python range)
  _name = "rv"         # Name of the random variable
  _log_fun = True          # Boolean flag denoting with _marg is in log_fun_e space
  _fun = np.zeros_like # Marginal probability distribution function
  _fun_args = None
  _fun_kwds = None 
  _inv = None          # Inverse function
  _inv_args = None
  _inv_kwds = None 

  # Private
  __callable = None    # Flag to denote if fun is callable

#-------------------------------------------------------------------------------
  def __init__(self, name, var=None, log_fun=None):
    self.set_var(name, var, log_fun)
    self.set_fun()
    self.set_inv()

#-------------------------------------------------------------------------------
  def set_var(self, name, var=None, log_fun=None):
    self._name = str(name)
    assert len(self._name), "Random variable name mandatary"
    if var is not None: self._var = var
    if log_fun is not None: self._log_fun = bool(log_fun)
    assert self._var is not None, "set_var(None) is not a valid random variable"

#-------------------------------------------------------------------------------
  def set_fun(self, fun=None, *args, **kwds):
    if fun is not None: self._fun = fun
    self._fun_args = tuple(args)
    self._fun_kwds = dict(kwds)
    self.__callable = callable(self._fun)
    if not self.__callable:
      self._fun = np.atleast_1d(self._fun)
      assert not len(_fun_args), "Optional arguments requires callable function"
      assert not len(_fun_kwds), "Optional keywords requires callable function"
    
#-------------------------------------------------------------------------------
  def set_inv(self, inv=None, *args, **kwds):
    if inv is not None: self._inv = inv
    self._inv_args = tuple(args)
    self._inv_kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def __call__(self, samples=None, 
                     randomise=False, 
                     log_probs=False,
                     epsilon=1e-10):

    # Handle non-callables
    if not self.__callable:
      assert samples is None, "Condition {} invalid for fixed functions"
      assert randomise is False, "Cannot randomise for fixed functions"
      if self._log_fun:
        if log_probs:
          return (None, self._fun)
        else:
          return (None, np.exp(self._fun))
      else:
        if log_probs:
          return (None, np.log(self._fun))
        else:
          return (None, self._fun)

    # Sample n vars or assume condition are the samples
    if type(samples) is int:
      lo = min(self._var.start, self._var.stop) + epsilon
      hi = max(self._var.start, self._var.stop) - epsilon
      if randomise:
        samples = np.sort(np.random.uniform(lo, hi, size=samples))
      else:
        samples = np.linspace(lo, hi, samples)
    else:
      samples = np.atleast_1d(samples)
      if randomise:
        samples = np.ravel(samples)
        samples = np.sort(samples[np.random.permutation(
                          len(samples))])

    # Output probs
    sampfun = self._fun(samples, *self._fun_args, **self._fun_kwds)
    if self._log_fun:
      if log_probs:
        return (samples, sampfun)
      else:
        return (samples, np.exp(sampfun))
    else:
      if log_probs:
        return (samples, np.log(sampfun))
      else:
        return (samples, sampfun)

#-------------------------------------------------------------------------------
