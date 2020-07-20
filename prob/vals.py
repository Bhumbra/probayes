"""
An abstract values class that defines a variable support set along and supports
invertible transformations.
"""

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import numpy as np
from prob.vtypes import eval_vtype, isunitsetint, uniform, VTYPES
from prob.func import Func

#-------------------------------------------------------------------------------
DEFAULT_VSET = {False, True}

#-------------------------------------------------------------------------------
class _Vals (ABC):

  # Protected
  _vset = None      # Variable set (array or 2-length tuple range)
  _vtype = None     # Variable type
  _vfun = None      # 2-length tuple of mutually inverting functions
  _lims = None      # Limits for floating point variable types
  _inside = None    # Lambda function for defining inside vset

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
    self._lims = None
    if vset is None: 
      vset = list(DEFAULT_VSET)
    elif isinstance(vset, (set, range)):
      vset = sorted(vset)
    elif np.iscalar(self._vset):
      vset = [self._vset]
    elif isinstance(vset, tuple):
      assert len(vset) == 2, \
          "Tuple vsets contain pairs of values, not {}".format(vset)
      vset = sorted(vset)
      vset = [(vset[0]), (vset[1])]
    elif isinstance(vset, np.ndarray):
      vset = np.sort(vset).tolist()
    else:
      assert isinstance(vset, list), \
          "Unrecognised vset specification: {}".format(vset)

    # At this point, self._vset can only be a sorted list
    if vtype is None:
      vset = np.array(vset)
      vtype = eval_vtype(vset)
    else:
      if any([isinstance(_vset, tuple) for _vset in vset]):
        for i in range(len(vset)):
          if isinstance(vset[i], tuple):
            vset[i] = vtype(vset[i])
      else:
        vset = np.array(vset, dtype=vtype).tolist()
    self._vset = vset
    self._vtype = eval_vtype(vtype)
    self._inside = lambda x: np.isin(x, self._vset, assume_unique=True)
    if self._vtype is not float:
      self._lims = np.array([min(self._vset), max(self._vset)])
      return self._vtype

    # Set up limits and inside function if float
    
    if any([isinstance(_vset, tuple) for _vset in self._vset]):
      lims = np.concatenate([np.array(_vtype).reshape([1]) \
                             for _vtype in vtype])
    else:
      lims = np.array(self._vset)
    assert len(lims) == 2, \
        "Floating point vset must be two elements, not {}".format(self._vset)
    if lims[1] < lims[0]:
      self._vset = self._vset[::-1]
    self._lims = np.sort(lims)
    if not isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._lims[0],
                                              x <= self._lims[1])
    elif not isinstance(self._vset[0], tuple) and \
        isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._lims[0],
                                              x < self._lims[1])
    elif isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x > self._lims[0],
                                              x <= self._lims[1])
    else:
      self._inside = lambda x: np.logical_and(x > self._lims[0],
                                              x < self._lims[1])
    return self._vtype

#-------------------------------------------------------------------------------
  def set_vfun(self, vfun=None, *args, **kwds):
    self._vfun = vfun
    if self._vfun is None:
      return

    assert self._vtype in VTYPES[float], \
        "Values transformation function only supported for floating point"
    message = "Input vfun be a two-sized tuple of callable functions"
    assert isinstance(self._vfun, tuple), message
    assert len(self._vfun) == 2, message
    assert callable(self._vfun[0]), message
    assert callable(self._vfun[1]), message
    self._vfun = Func(self._vfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def ret_vtype(self):
    return self._vtype

#-------------------------------------------------------------------------------
  def ret_vfun(self, index=None):
    if self._vfun is None:
      return lambda x:x
    if index is None:
      return self._vfun
    return self._vfun[index]

#-------------------------------------------------------------------------------
  def get_bounds(self, use_vfun=True):
    if self._lims is None:
      return None
    if use_vfun:
      use_vfun = self._vfun is not None
    if not use_vfun:
      return self._lims[0], self._lims[1]
    lims = self.ret_vfun(0)(self._lims)
    lo, hi = min(lims), max(lims)
    return lo, hi

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):

    # Default to arrays of complete sets
    if values is None:
      values = np.array(list(self._vset), dtype=self._vtype)

    # Sets may be used to sample from support sets
    elif isunitsetint(values):
      number = list(values)[0]

      # Non-continuous
      if self._vtype not in VTYPES[float]:
        values = np.array(list(self._vset), dtype=self._vtype)
        if not number:
          values = values[np.random.randint(0, len(values))]
        else:
          divisor = len(self._vset)
          if number > 0:
            indices = np.arange(number, dtype=int) % divisor
          else:
            indices = np.random.permutation(-number, dtype=int) % divisor
          values = values[indices]
       
      # Continuous
      else:
        lo, hi = self.get_bounds(use_vfun=True)
        lohi = np.array([lo, hi])
        assert np.all(np.isfinite(lohi)), \
            "Cannot evaluate {} values for bounds: {}".format(values, lohi)
        values = uniform(lo, hi, number)
        if self._vfun:
          return self.ret_vfun(1)(values)
        return values
    return values
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
