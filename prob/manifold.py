""" A module to provide manifold functionality to probability distributions """

#-------------------------------------------------------------------------------
import numpy as np
import collections
import warnings
from prob.vtypes import isscalar

#-------------------------------------------------------------------------------
class Manifold:

  # Public
  vals = None        # Ordered dictionary with values
  dims = None        # Ordered dictionary specifying dimension index of vals
  ndim = None        # Number of dimensions
  shape = None       # Dimension shape

  # Protected
  _keys = None       # Keys of vals as list
  _dimension = None  # Dimension of vals
  _arescalars = None # Whether vals are scalars
  _isscalar = None   # all(_arescalars)

#-------------------------------------------------------------------------------
  def __init__(self, vals=None, dims=None):
    self.set_vals(vals, dims)
  
#-------------------------------------------------------------------------------
  def set_vals(self, vals=None, dims=None):
    self.vals = vals
    self.dims = dims
    self.shape = []
    self._keys = []
    self._shape = []
    self._arescalars = []
    self._isscalar = None
    eval_dims = self.dims is None
    if eval_dims:
      self.dims = collections.OrderedDict()
    if self.vals is None:
      return self.dims
    assert isinstance(self.vals, dict), \
        "Dist vals must be a dictionary but given: {}".format(self.vals)
    if eval_dims:
      if not isinstance(self.vals, collections.OrderedDict):
        warnings.warn("Determining dimensions {} rather than OrderedDict".\
                       format(type(self.vals)))
    self._keys = list(self.vals.keys())

    # Tranform {None} to {0} to play nicely with isunitsetint
    for key in self._keys:
      if isinstance(self.vals[key], set):
        if len(self.vals[key]) == 1:
          element = list(self.vals[key])[0]
          import pdb; pdb.set_trace()
          if element is None:
            self.vals.update({key: {0}})

    # Count number of non-scalar dimensions
    print(self.vals)
    self._arescalars = [isscalar(val) for val in self.vals.values()]
    print(self._arescalars)
    self.ndim = sum(np.logical_not(self._arescalars))
    self._isscalar = self.ndim == 0

    # Corroborate vals and dims
    ones_ndim = np.ones(self.ndim, dtype=int)
    run_dim = -1
    for i, key in enumerate(self._keys):
      values = self.vals[key]


      # Scalars are dimensionless and therefore shapeless
      if self._arescalars[i]:
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
        run_dim += 1
        assert isinstance(values, np.ndarray), \
            "Dictionary of numpy arrays expected for nonscalars but found" + \
            "type {} for key {}".format(type(values), key)
        val_size = values.size
        assert val_size == np.max(values.shape), \
            "Values must have one non-singleton dimension but found" + \
            "shape {} for key {}".format(values.shape, key)
        self.shape.append(val_size)
        if eval_dims:
          self.dims.update({key: run_dim})
        else:
          assert key in self.dims, "Missing key {} in dims specification {}".\
              format(key, self.dims)
        vals_shape = np.copy(ones_ndim)
        vals_shape[self.dims[key]] = val_size
        re_shape = self.ndim != values.ndim or \
                   any(np.array(values.shape) != vals_shape)
        if re_shape:
          self.vals[key] = values.reshape(vals_shape)

    return self.dims

#-------------------------------------------------------------------------------
  def redim(self, dims):
    """ 
    Returns a manifold according to redimensionised values in dims, index-
    ordered by the order in dims
    """
    for key in self._keys:
      if self.dims[key] is not None:
        assert key in dims, \
            "Missing key for nonscalar {} in dim".format(key, dims)
      elif key in dims:
        assert dims[key] is None, \
            "Dimension {} requested for scalar with key {}".\
            format(dims[key], key)
    vals = {key: self.vals[key] for key in dims.keys()}
    return Manifold(vals, dims)

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
