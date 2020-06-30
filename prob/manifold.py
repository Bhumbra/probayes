""" A module to provide manifold functionality to probability distributions """

#-------------------------------------------------------------------------------
import numpy as np
import collections
import warnings
from prob.vtypes import isscalar

#-------------------------------------------------------------------------------
class Manifold:

  # Public
  vals = None
  ndim = None

  # Private:
  _keys = None       # Keys of vals
  _dimension = None  # Dimension of vals
  _arescalars = None # Whether vals are scalars
  _isscalar = None   # all(_arescalars)

#-------------------------------------------------------------------------------
  def __init__(self, vals=None):
    self.set_vals(vals)
  
#-------------------------------------------------------------------------------
  def set_vals(self, vals=None):
    self.vals = vals
    self.ndim = None
    self._dimension = collections.OrderedDict()
    self._keys = list(self.vals.keys())
    self._arescalars = None
    self._isscalar = None
    if self.vals is None:
      return self._dimension
    else:
      assert isinstance(self.vals, dict), \
          "Dist vals must be a dictionary but given: {}".format(self.vals)
      if not isinstance(self.vals, collections.OrderedDict):
        warnings.warn("Ordered dictionary of values expected, not {}".\
                       format(type(self.vals)))
    self._keys = list(self.vals.keys())
    self._arescalars = [None] * len(self._keys)
    ndim = 0
    for i, key in enumerate(self._keys):
      values = self.vals[key]
      self._arescalars[i] = isscalar(values)
      if self._arescalars[i]:
        self._dimension.update({key:None})
      else:
        assert isinstance(values, np.ndarray), \
            "Dictionary of numpy arrays expected for nonscalars but found" + \
            "type {} for key {}".format(type(values), key)
        val_size = values.size
        val_shape = values.shape
        assert val_size == np.max(val_shape), \
            "Values must have one non-singleton dimension but found" + \
            "shape {} for key {}".format(val_shape, key)
        if val_size == 1:
          val_dim = max(0, ndim - 1)
        else:
          val_dim = np.argmax(val_shape)
          ndim = max(val_dim+1, ndim)
        self._dimension.update({key: val_dim})
    self.ndim = ndim
    self._iscalar = all(self._arescalars)
    return self._dimension

#-------------------------------------------------------------------------------
  def ret_dimension(self, key=None):
    if key is None:
      return self._dimension
    if type(key) is int:
      key = self._keys[key]
    return self._dimension[key]

#-------------------------------------------------------------------------------
  def ret_arescalars(self):
    return self._arescalar
         
#-------------------------------------------------------------------------------
  def ret_isscalar(self, key=None):
    if key is None:
      return self._isscalar
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._isscalar[key]

#-------------------------------------------------------------------------------
  def __getitem__(self, key=None):
    if key is None:
      return self.vals
    if type(key) is int:
      key = self._keys[key]
    if key not in self._keys:
      return None
    return self.vals[key]

#-------------------------------------------------------------------------------
  def __len__(self):
    return len(self.vals)
   
#-------------------------------------------------------------------------------
