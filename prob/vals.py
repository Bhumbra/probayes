"""
An abstract values class that defines a variable support set along and supports
invertible transformations.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import numpy as np

#-------------------------------------------------------------------------------
NOMINAL_VSET = [False, True]
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
    if vset is None: vset = NOMINAL_VSET

    # Support floating range limits as tuple, otherwise use numpy array
    self._vset = vset
    if isinstance(self._vset, (list, np.ndarray, bool, int, float)):
      if not isinstance(self._vset, np.ndarray) or self._vset.ndim < 1:
        not_sort = not isinstance(self._vset, set)
        self._vset = np.atleast_1d(self._vset) if not_sort \
                     else np.sort(np.atleast_1d(self._vset))
    elif isinstance(self._vset, range):
      self._vset = np.arange(self._vset.start, self._vset,stop, self._vset.step,
                             dtype=int)
    elif isinstance(self._vset, set): # Convert elements to float
      assert len(self._vset) == 2, \
          "Tuple vset must be of length 2, not {}".format(len(self._vset))
      vset = np.array(list(self._vset), dtype=float)
      self._vset = set([np.min(vset), np.max(vset)])
    else:
      raise TypeError("Unrecognised variable set type: {}".format(
                      type(self._vset)))

    # Detect vtype if not specified (None is permitted)
    self._vtype = vtype
    if self._vtype is None:
      if isinstance(self._vset, set):
        self._vtype = float
      elif isinstance(self._vset, np.ndarray):
        self._vtype = NUMPY_DTYPES.get(self._vset.dtype, None)
      else:
        self._vtype = type(self._vset)
    return self.ret_vtype()

#-------------------------------------------------------------------------------
  def set_vfun(self, vfun=None, *args, **kwds):
    self._vfun = vfun
    self._vfun_args = tuple(args)
    self._vfun_kwds = dict(kwds)

    if self._vfun is not None:
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
    vset = list(self._vset) if isinstance(self._vset, set) else self._vset
    lo = self.vfun_0(np.min(vset), use_vfun)
    hi = self.vfun_0(np.max(vset), use_vfun)
    if use_vfun and self._vfun is not None:
      lo, hi = float(lo), float(hi)
    return lo, hi

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None, use_vfun=True):

    # Default to complete sets
    if values is None:
      assert not isinstance(self._vset, set),\
          "Samples must be specified for variable set: {}".format(self._vset)
      self.values = np._vset

    # Sets may be used to sample from support sets
    elif isinstance(values, set):
      assert len(values) == 1, "Set values must contain one integer"
      val = int(list(values)[0])

      # Non-continuous
      if not isinstance(self._vset, set):
        divisor = len(self._vset)
        if val >= 0:
          indices = np.arange(val, dtype=int) % divisor
        else:
          indices = np.random.permutation(-val, dtype=int) % divisor
        values = self._vset[indices]
       
      # Continuous
      else:
        lo, hi = self.get_bounds(use_vfun=False)
        assert np.all(np.isfinite([lo, hi])), \
            "Cannot evaluate {} values for bounds: {}".format(values, vset)
        if val == 1:
          values = np.atleast_1d(0.5 * (lo+hi))
        elif val >= 0:
          values = np.linspace(lo, hi, val)
        else:
          values = np.sort(np.random.uniform(lo, hi, size=-val))

    return self.vfun_0(values, use_vfun)
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
