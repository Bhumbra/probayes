""" A module to provide manifold functionality to probability distributions """

#-------------------------------------------------------------------------------
import numpy as np
import collections
import warnings
from probayes.vtypes import issingleton

#-------------------------------------------------------------------------------
class Manifold:

  # Public
  vals = None          # Ordered dictionary with values
  dims = None          # Ordered dictionary specifying dimension index of vals
  ndim = None          # Number of dimensions
  sizes = None         # Size of dimensions including shared
  shape = None         # Size of dimension shape excluding shared
  size = None          # prod(sizes)

  # Protected
  _keys = None         # Keys of vals as list
  _dimension = None    # Dimension of vals
  _aresingleton = None # Whether vals are scalars
  _issingleton = None  # all(_aresingleton)

#-------------------------------------------------------------------------------
  def __init__(self, vals=None, dims=None):
    self.set_vals(vals, dims)
  
#-------------------------------------------------------------------------------
  def set_vals(self, vals=None, dims=None):
    self.vals = vals
    self.dims = dims
    self.sizes = []
    self.size = None
    self.shape = []
    self._keys = []
    self._aresingleton = []
    self._issingleton = None
    eval_dims = self.dims is None
    if eval_dims:
      self.dims = collections.OrderedDict()
    if self.vals is None:
      return self.dims
    assert isinstance(self.vals, dict), \
        "Dist vals must be a dictionary but given: {}".format(type(self.vals))
    self._keys = list(self.vals.keys())

    # Tranform {None} to {0} to play nicely with isunitsetint
    for key in self._keys:
      if isinstance(self.vals[key], set):
        if len(self.vals[key]) == 1:
          element = list(self.vals[key])[0]
          if element is None:
            self.vals.update({key: {0}})

    # Count number of non-singleton dimensions
    self._aresingleton = [issingleton(val) for val in self.vals.values()]
    self._issingleton = np.all(self._aresingleton)
    self.ndim = 0 
    for dim in self.dims.values():
      if dim is not None:
        self.ndim = max(self.ndim, dim+1)

    # If evaluating non-singleton dimensions, warn if using dict
    if eval_dims and len(self.vals) > 1 and not self._issingleton:
      if not isinstance(self.vals, collections.OrderedDict):
        warnings.warn(\
            "Determining dimensions from multi-key {} rather than OrderedDict".\
            format(type(self.vals)))

    # Corroborate vals and dims
    ones_ndim = np.ones(self.ndim, dtype=int)
    self.shape = [None] * self.ndim
    nonsingleton_count = -1
    for i, key in enumerate(self._keys):
      values = self.vals[key]

      # Scalars are dimensionless and therefore shapeless
      if self._aresingleton[i]:
        if eval_dims:
          self.dims.update({key:None})
        elif key in self.dims:
          assert self.dims[key] == None,\
            "Dimension index for scalar value {} must be None, not {}".\
            format(key, self.dims[key])
        else:
          self.dims.update({key: None})

      # Non-scalars require correct dimensionality
      else:
        nonsingleton_count += 1
        assert isinstance(values, np.ndarray), \
            "Dictionary of numpy arrays expected for nonsingletons but found" + \
            "type {} for key {}".format(type(values), key)
        val_size = values.size
        assert val_size == np.max(values.shape), \
            "Values must have one non-singleton dimension but found" + \
            "shape {} for key {}".format(values.shape, key)
        if eval_dims:
          self.dims.update({key: nonsingleton_count})
          if len(self.shape) == nonsingleton_count:
            self.shape.append(None)
            self.ndim = len(self.shape)
            ones_ndim = np.ones(self.ndim, dtype=int)
        else:
          assert key in self.dims, "Missing key {} in dims specification {}".\
              format(key, self.dims)
        self.sizes.append(val_size)
        self.shape[self.dims[key]] = val_size
        vals_shape = np.copy(ones_ndim)
        vals_shape[self.dims[key]] = val_size
        re_shape = self.ndim != values.ndim or \
                   any(np.array(values.shape) != vals_shape)
        if re_shape:
          self.vals[key] = values.reshape(vals_shape)
    self.size = int(np.prod(self.shape))

    return self.dims

#-------------------------------------------------------------------------------
  def ret_vals(self, keys):
    for key in keys:
      assert key in self._keys, "Key {} not found among {}".\
          format(key, self._keys)
    keys = set(keys)
    vals = collections.OrderedDict()
    for key in self._keys:
      if key in keys:
        vals.update({key: self.vals[key]})
    return vals
    
#-------------------------------------------------------------------------------
  def redim(self, dims):
    """ 
    Returns a manifold according to redimensionised values in dims, index-
    ordered by the order in dims
    """
    for key in self._keys:
      if self.dims[key] is not None:
        assert key in dims, \
            "Missing key for nonsingleton {} in dim".format(key, dims)
      elif key in dims:
        assert dims[key] is None, \
            "Dimension {} requested for singleton with key {}".\
            format(dims[key], key)
    vals = {key: self.vals[key] for key in dims.keys()}
    return Manifold(vals, dims)

#-------------------------------------------------------------------------------
  def rekey(self, keymap):
    assert isinstance(keymap, dict), \
        "Input keymap must a dictionary in the form {old_key: new_key}"
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    for key in self._keys:
      if key not in keymap.keys():
        vals.update({key: self.vals[key]})
        dims.update({key: self.dims[key]})
      else:
        map_key = keymap[key]
        assert map_key not in vals, \
            "Key map results in duplicate for key: {}".format(map_key)
        vals.update({map_key: self.vals[key]})
        dims.update({map_key: self.dims[key]})
    return Manifold(vals, dims)
    
#-------------------------------------------------------------------------------
  def ret_aresingleton(self):
    return self._aresingleton
         
#-------------------------------------------------------------------------------
  def ret_issingleton(self, key=None):
    if key is None:
      return self._issingleton
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._issingleton[key]

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
