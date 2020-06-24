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
    if isinstance(self._vset, (set, list, np.ndarray, bool, int, float)):
      if not isinstance(self._vset, np.ndarray) or self._vset.ndim < 1:
        not_sort = not isinstance(self._vset, set)
        self._vset = np.atleast_1d(self._vset) if not_sort \
                     else np.sort(np.atleast_1d(self._vset))
    elif isinstance(self._vset, range):
      self._vset = np.arange(self._vset.start, self._vset,stop, self._vset.step,
                             dtype=int)
    elif isinstance(self._vset, tuple):
      assert len(self._vset) == 2,\
          "Tuple vset must be of length 2, not {}".format(len(self._vset))
      self._vset = tuple([float(self._vset[0]), float(self._vset[1])])
    else:
      raise TypeError("Unrecognised variable set type: {}".format(
                      type(self._vset)))

    # Detect vtype if not specified (None is permitted)
    self._vtype = vtype
    if self._vtype is None:
      if isinstance(self._vset, tuple):
        self._vtype = float
      elif isinstance(self._vset, np.ndarray):
        self._vtype = NUMPY_DTYPES.get(self._vset.dtype, None)
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
    lo = self.vfun_0(np.min(self._vset), use_vfun)
    hi = self.vfun_0(np.max(self._vset), use_vfun)
    if use_vfun and self._vfun is not None:
      lo, hi = float(lo), float(hi)
    return lo, hi

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):

    # Convert arrays
    if isinstance(values, (tuple, list, np.ndarray)):
      immutable = isinstance(values, tuple)
      values = np.atleast_1d(values) if not self._vtype else \
                np.atleast_1d(values).astype(self._vtype)
      values = self.vfun_0(values, not(immutable))
      
    # Integer values n values
    elif values is None or type(values) is int:
      if values is None:
        assert not isinstance(self._vset, tuple),\
            "Samples must be specified for variable set: {}".format(self._vset)
        values = len(self._vset)

      # Non-continuous support sets
      if not isinstance(self._vset, tuple):
        divisor = len(self._vset)
        if values >= 0:
          indices = np.arange(values, dtype=int) % divisor
        else:
          indices = np.random.permutation(-values, dtype=int) % divisor
        values = self._vset[indices]
      
      # Continuinous support sets
      else:
        vset = np.array(self._vset, dtype = float)
        assert np.all(np.isfinite(vset)), \
            "Cannot evaluate {} values for bounds: {}".format(values, vset)
        lo, hi = self.get_bounds()
        if values == 1:
          values = np.atleast_1d(0.5 * (lo+hi))
        elif values >= 0:
          values = np.linspace(lo, hi, values)
        else:
          values = np.sort(np.random.uniform(lo, hi, size=-values))
      values = self.vfun_0(values)

    else:
      raise TypeError("Ambiguous values type: ".format(type(values)))
    return values
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
