""" Provides distribution functionality based on a named dictionary. """

#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.vtypes import issingleton
from probayes.named_dict import NamedDict

#-------------------------------------------------------------------------------
class Distribution (NamedDict):
  """ A distribution is a named topological space defined over specific values 
  for one or more variables. Those values are defined by the dictionary
  {variable_name: values} [Distribution.values()] with dimensions 
  {variable_name: dim} [Distribution.dims]. The dimensionality defines a finite 
  space of sizes with it permissable for variables to share dimensions and in 
  such cases reducing the shape dimensions.

  :example:
  >>> import numpy as np
  >>> import probayes as pb
  >>> xy_dict = {'x': np.arange(3), 'y': np.arange(2)}
  >>> xy = pb.Distribution('x,y', xy_dict, dims={'x': 0, 'y': 1})
  >>> print(xy.shape)
  [3, 2]

  Scalars can be included among vals, but must possess dimensionality None and
  do not contribute to the size or shape of the distribution.
  """

  # Protected
  _dims = None         # Ordered dictionary specifying dimension index of vals
  _ndim = None         # Number of dimensions
  _sizes = None        # Size of dimensions including shared
  _shape = None        # Size of dimension shape excluding shared
  _size = None         # prod(sizes)
  _keylist = None      # Keys of vals as list
  _keyset = None       # Keys of vals a set
  _aresingleton = None # Whether vals are scalars
  _issingleton = None  # all(_aresingleton)
  _disallowed = {'attrs', 'prob', 'pscale'} # disallowed keys

#-------------------------------------------------------------------------------
  def __init__(self, name, *args, **kwds):
    """ Initialises the Distribution with a name, args, and kwds in the same
    way as NamedDict(), except with 'dims' as a reserved keyword to set the
    dimensionality. """
    args = tuple(args)
    kwds = dict(kwds)
    dims = None if 'dims' not in kwds else kwds.pop('dims')
    super().__init__(name, *args, **kwds)
    self.dims = dims
  
#-------------------------------------------------------------------------------
  @property
  def dims(self):
    return self._dims

  @property
  def ndim(self):
    return self._ndim

  @property
  def sizes(self):
    return self._sizes

  @property
  def shape(self):
    return self._shape

  @property
  def size(self):
    return self._size

  @property
  def keylist(self):
    return self._keylist

  @property
  def short_name(self):
    return ','.join(self._keylist)

  @property
  def keyset(self):
    return self._keyset

  @property
  def aresingleton(self):
    return self._aresingleton

  @property
  def issingleton(self):
    return self._issingleton

  @dims.setter
  def dims(self, dims=None):
    """ Sets the dimensions for each of the variables.

    :param dims: a dictionary of {variable_name: variable_dim}

    The keys should correspond to that of a dictionary. If dims
    is None, then the dimensionality is set according to the
    order in values.

    :example:
    >>> import numpy as np
    >>> from collections import OrderedDict()
    >>> import probayes as pb
    >>> xy_dict = {'x': np.arange(3), 'y': np.arange(2)}
    >>> xy = pb.Distribution('x,y', xy_dict)
    >>> print(xy.dims))
    OrderedDict([('x', 0), ('y', 1)])
    """
    self._dims = dims
    self._sizes = []
    self._shape = []
    self._size = None
    self._keylist = list(self.keys())
    self._keyset = frozenset(self._keylist)
    self._aresingleton = []
    self._issingleton = None
    eval_dims = self._dims is None
    if eval_dims:
      self._dims = collections.OrderedDict()
    else:
      self._dims = collections.OrderedDict(self._dims)
    if not self.__len__():
      return

    # Tranform {None} to {0} to play nicely with isunitsetint
    for key in self._keylist:
      if key in self._disallowed:
        raise ValueError(f"Disallowed key: {key}")
      if isinstance(self[key], set):
        if len(self[key]) == 1:
          element = list(self[key])[0]
          if element is None:
            self.update({key: {0}})

    # Count number of non-singleton dimensions
    self._aresingleton = [issingleton(val) for val in self.values()]
    self._issingleton = np.all(self._aresingleton)
    self._ndim = 0 
    for dim in self._dims.values():
      if dim is not None:
        self._ndim = max(self._ndim, dim+1)

    # Corroborate vals and dims
    ones_ndim = np.ones(self._ndim, dtype=int)
    self._shape = [None] * self.ndim
    nonsingleton_count = -1
    for i, key in enumerate(self._keylist):
      values = self[key]

      # Scalars are dimensionless and therefore shapeless
      if self._aresingleton[i]:
        if eval_dims:
          self._dims.update({key:None})
        elif key in self._dims:
          assert self._dims[key] == None,\
            "Dimension index for scalar value {} must be None, not {}".\
            format(key, self._dims[key])
        else:
          self._dims.update({key: None})

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
          self._dims.update({key: nonsingleton_count})
          if len(self._shape) == nonsingleton_count:
            self._shape.append(None)
            self._ndim = len(self._shape)
            ones_ndim = np.ones(self._ndim, dtype=int)
        else:
          assert key in self._dims, "Missing key {} in dims specification {}".\
              format(key, self._dims)
        self._sizes.append(val_size)
        self._shape[self._dims[key]] = val_size
        vals_shape = np.copy(ones_ndim)
        vals_shape[self._dims[key]] = val_size
        re_shape = self._ndim != values.ndim or \
                   any(np.array(values.shape) != vals_shape)
        if re_shape:
          self[key] = values.reshape(vals_shape)
    self._size = int(np.prod(self._shape))

#-------------------------------------------------------------------------------
  def singleton(self, key=None):
    """ Returns whether Distribution defines a singleton according key:

    if key is None (default): returns self.issingleton
    if key is an int: returns self.aresingleton[key]
    if key is an key: returns key-corresponded element of self.aresingleton
    """
    if key is None:
      return self._issingleton
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._issingleton[key]

#-------------------------------------------------------------------------------
  def lookup(self, keys, as_dist=False):
    """ Returns the values of Distribution filtering by keys """ 
    keys = keys or self._keyset
    keys = set(keys)
    vals = collections.OrderedDict()
    if as_dist:
      dims = collections.OrderedDict()
      keylist = []
      for key in self._keylist:
        if key in keys:
          keylist.append(key)
          vals.update({key: self[key]})
          dims.update({key: self._dims[key]})
      return Distribution(','.join(keylist), vals, dims=dims)
    seen_keys = set()
    for i, key in enumerate(self._keylist):
      if key in keys and key not in seen_keys:
        if self._aresingleton[i]:
          seen_keys.add(key)
          vals.update({key: self[key]})
        else:
          shared_keys = [key]
          for j, cand_key in enumerate(self._keylist):
            if j > i and cand_key in keys and not self._aresingleton[j]:
              if self.dims[key] == self.dims[cand_key]:
                shared_keys.append(cand_key)
          if len(shared_keys) == 1:
            vals.update({key: np.ravel(self[key])})
            seen_keys.add(key)
          else:
            val = [None] * len(shared_keys)
            for j, shared_key in enumerate(shared_keys):
              val[j] = np.ravel(self[shared_key])
              seen_keys.add(shared_key)
            vals.update({','.join(shared_keys): tuple(val)})
    return vals

#-------------------------------------------------------------------------------
  def redim(self, dims):
    """  Returns a Distribution according to redimensionised values in dims, 
    index-ordered by the order in dims.
    """
    for key in self._keylist:
      if self._dims[key] is not None:
        assert key in dims, \
            "Missing key for nonsingleton {} in dim {}".format(key, dims)
      elif key in dims:
        assert dims[key] is None, \
            "Dimension {} requested for singleton with key {}".\
            format(dims[key], key)
    vals = {key: self[key] for key in dims.keys()}
    return Distribution(self._name, vals, dims=dims)

#-------------------------------------------------------------------------------
  def rekey(self, keymap):
    """ Returns a Manifold rekeying the values and dimensions according the
    dictionary mappings given in keymap. 
    """
    assert isinstance(keymap, dict), \
        "Input keymap must a dictionary in the form {old_key: new_key}"
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    for key in self._keylist:
      if key not in keymap.keys():
        vals.update({key: self[key]})
        dims.update({key: self.dims[key]})
      else:
        map_key = keymap[key]
        assert map_key not in vals, \
            "Key map results in duplicate for key: {}".format(map_key)
        vals.update({map_key: self[key]})
        dims.update({map_key: self.dims[key]})
    return Distribution(self._name, vals, dims=dims)
    
#-------------------------------------------------------------------------------
  def serialise(self):
    short_name = self.short_name
    serialised = {short_name: {key:val for key, val in self.items()}}
    serialised[short_name].update({'attrs': self.dims})
    return serialised

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
