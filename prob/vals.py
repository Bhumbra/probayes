"""
An abstract values class that defines a variable support set along and supports
invertible transformations.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import numpy as np
from prob.vtypes import eval_vtype

#-------------------------------------------------------------------------------
DEFAULT_VSET = {False, True}

#-------------------------------------------------------------------------------
class _Vals (ABC):

  # Protected
  _vset = None      # Variable set (array or 2-length tuple range)
  _vtype = None     # Variable type
  _vfun = None      # 2-length tuple of mutually inverting functions
  _vfun_args = None
  _vfun_kwds = None

#-------------------------------------------------------------------------------
  def __init__(self, vset=None, 
                     vtype=None,
                     vfun=None,
                     *args,
                     **kwds):
    self.set_vset(vset, vtype)
    self.set_vfun(vfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_vset(self, vset=None, vtype=None):

    # Default vset to nominal
    if vset is None: 
      vset = list(DEFAULT_VSET)
    elif isinstance(vset, (set, range)):
      vset = list(vset)
    elif np.iscalar(self._vset):
      vset = [self._vset]

    # At this point, self._vset should be a list, tuple, or np.ndarray
    if vtype is None:
      vset = np.array(vset)
      vtype = eval_vtype(vset)
    else:
      vset = np.array(vset, dtype=vtype)
    self._vset = set(vset)
    self._vtype = eval_vtype(vtype)
    return self._vtype

#-------------------------------------------------------------------------------
  def set_vfun(self, vfun=None, *args, **kwds):
    self._vfun = vfun
    self._vfun_args = tuple(args)
    self._vfun_kwds = dict(kwds)

    if self._vfun is not None:
      assert self._vtype in (float, np.dtype('float32'),  np.dtype('float64')), \
          "Values transformation function only supported for floating point"
      message = "Input vfun be a two-sized tuple of callable functions"
      assert isinstance(self._vfun, tuple), message
      assert len(self._vfun) == 2, message
      assert callable(self._vfun[0]), message
      assert callable(self._vfun[1]), message

#-------------------------------------------------------------------------------
  def ret_vtype(self):
    return self._vtype

#-------------------------------------------------------------------------------
  def ret_vfun(self, option=None):
    return self._vfun

#-------------------------------------------------------------------------------
  def vfun_0(self, values, use_vfun=True):
    if self._vfun is None or not use_vfun:
      return values
    return self._vfun[0](values, *self._vfun_args, **self._vfun_kwds)

#-------------------------------------------------------------------------------
  def vfun_1(self, values, use_vfun=True):
    if self._vfun is None or not use_vfun:
      return values
    return self._vfun[1](values, *self._vfun_args, **self._vfun_kwds)

#-------------------------------------------------------------------------------
  def get_bounds(self, use_vfun=False):
    if self._vset is None:
      return None
    lo = self.vfun_0(min(self._vset), use_vfun)
    hi = self.vfun_0(max(self._vset), use_vfun)
    if use_vfun and self._vfun is not None:
      lo, hi = float(lo), float(hi)
    return lo, hi

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):

    # Default to arrays of complete sets
    if values is None:
      values = np.array(list(self._vset), dtype=self._vtype)

    # Sets may be used to sample from support sets
    elif isinstance(values, set):
      assert len(values) == 1, "Set values must contain one integer"
      number = int(list(values)[0])
      values = np.array(list(self._vset), dtype=self._vtype)

      # Non-continuous
      if self._vtype not in [float, np.dtype('float32'), np.dtype('float64')]:
        divisor = len(self._vset)
        if number >= 0:
          indices = np.arange(number, dtype=int) % divisor
        else:
          indices = np.random.permutation(-number, dtype=int) % divisor
        values = values[indices]
       
      # Continuous
      else:
        lo, hi = self.get_bounds(use_vfun=True)
        assert np.all(np.isfinite([lo, hi])), \
            "Cannot evaluate {} values for bounds: {}".format(values, vset)
        if number == 1:
          values = np.atleast_1d(0.5 * (lo+hi))
        elif number >= 0:
          values = np.linspace(lo, hi, number)
        else:
          values = np.random.uniform(lo, hi, size=-number)
        return self.vfun_1(values)

    return values
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
