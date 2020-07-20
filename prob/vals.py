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
  _lims = None      # Numpy array of bounds of vset
  _limits = None    # Transformed self._lims
  _length = None    # Difference in self._limits
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
    if vset is None: 
      vset = list(DEFAULT_VSET)
    elif isinstance(vset, (set, range)):
      vset = sorted(vset)
    elif np.isscalar(self._vset):
      vset = [self._vset]
    elif isinstance(vset, tuple):
      assert len(vset) == 2, \
          "Tuple vsets contain pairs of values, not {}".format(vset)
      vset = sorted(vset)
      vset = [(vset[0],), (vset[1],)]
    elif isinstance(vset, np.ndarray):
      vset = np.sort(vset).tolist()
    else:
      assert isinstance(vset, list), \
          "Unrecognised vset specification: {}".format(vset)

    # At this point, vset can only be a list, but may contain tuples
    aretuples = ([isinstance(_vset, tuple) for _vset in vset])
    if not any(aretuples):
      if vtype is None:
        vset = np.array(vset)
        vtype = eval_vtype(vset)
      else:
        vset = np.array(vset, dtype=vtype)
    else:
      if vtype is not None:
        assert vtype in VTYPES[float], \
            "Bounded variables supported only for float vtypes, not {}".\
            format(vtype)
      vtype = float
      assert len(vset) == 2, \
          "Unrecognised set specification: {}".vset
      for i in range(len(vset)):
        if isinstance(vset[i], tuple):
          assert len(vset[i]) == 1, \
              "Unrecognised set specification: {}".vset[i]
          vset[i] = vtype(vset[i][0])
    self._vset = vset
    for i, istuple in enumerate(aretuples):
      if istuple:
        self._vset[i] = self._vset[i],
    self._vtype = eval_vtype(vtype)
    self._eval_lims()
    return self._vtype

#-------------------------------------------------------------------------------
  def _eval_lims(self):
    self._lims = None
    self._limits = None
    self._length = None
    self._inside = None
    
    if self._vset is None:
      return self._length

    if self._vtype not in VTYPES[float]:
      self._lims = np.array([min(self._vset), max(self._vset)])
      self._limits = self._lims
      self._length = len(self._vset)
      self._inside = lambda x: np.isin(x, self._vset, assume_unique=True)
      return self._length

    """ Evaluates the limits from vset assuming vtype is set """
    # Set up limits and inside function if float
    if any([isinstance(_vset, tuple) for _vset in self._vset]):
      lims = np.concatenate([np.array(_vset).reshape([1]) 
                             for _vset in self._vset])
    else:
      lims = np.array(self._vset)
    assert len(lims) == 2, \
        "Floating point vset must be two elements, not {}".format(self._vset)
    if lims[1] < lims[0]:
      vset = vset[::-1]
      self._vset = vset
    self._lims = np.sort(lims)
    self._limits = self._lims if self._vfun is None \
                   else self.ret_vfun(0)(self._lims)
    self._length = max(self._limits) - min(self._limits)

    # Now set inside function
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
    self._eval_lims()

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
  def ret_lims(self):
    return self._lims

#-------------------------------------------------------------------------------
  def ret_limits(self):
    return self._limits

#-------------------------------------------------------------------------------
  def ret_length(self):
    return self._length

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):

    # Default to arrays of complete sets
    if values is None:
      assert self._vtype not in VTYPES[float], \
          "Cannot default evaluation for vtype: {}".format(self._vtype)
      return np.array(list(self._vset), dtype=self._vtype)

    # Sets may be used to sample from support sets
    if isunitsetint(values):
      number = list(values)[0]

      # Non-continuous
      if self._vtype not in VTYPES[float]:
        values = np.array(list(self._vset), dtype=self._vtype)
        if not number:
          values = values[np.random.randint(0, len(values))]
        else:
          if number > 0:
            indices = np.arange(number, dtype=int) % self._length
          else:
            indices = np.random.permutation(-number, dtype=int) % self._length
          values = values[indices]
        return values
       
      # Continuous
      else:
        assert np.all(np.isfinite(self._limits)), \
            "Cannot evaluate {} values for bounds: {}".format(
                values, self._limits)
        values = uniform(self._limits[0], self._limits[1], number, 
                           isinstance(self._vset[0], tuple), 
                           isinstance(self._vset[1], tuple)
                        )

      # Only use vfun when isunitsetint(values)
      if self._vfun:
        return self.ret_vfun(1)(values)
    return values
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
