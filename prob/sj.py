"""
A stochastic junction comprises a collection of a random variables that 
participate in a joint probability distribution function.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
from prob.rv import RV, io_use_vfun
from prob.dist import Dist, marg_prod
from prob.ptypes import prod_ptype, prod_prob

#-------------------------------------------------------------------------------
class SJ:

  # Protected
  _name = None      # Cannot be set externally
  _rvs = None       # Dict of random variables
  _nrvs = None
  _keys = None
  _keyset = None
  _ptype = None
  _use_vfun = None
  _arg_order = None
  _as_scalar = None # dictionary of bools
  _prob = None
  _prob_args = None
  _prob_kwds = None

  # Private
  __callable = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)
    self.set_prob()
    self.set_use_vfun()

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    if len(args) == 1 and isinstance(args[0], (SJ, dict, set, tuple, list)):
      args = args[0]
    else:
      args = tuple(args)
    self.add_rv(args)
    return self.ret_rvs()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    assert self._prob is None, \
      "Cannot assign new randon variables after specifying joint/condition prob"
    if self._rvs is None:
      self._rvs = collections.OrderedDict()
    if isinstance(rv, (SJ, dict, set, tuple, list)):
      rvs = rv
      if isinstance(rvs, SJ):
        rvs = rvs.ret_rvs()
      if isinstance(rvs, dict):
        rvs = rvs.values()
      [self.add_rv(rv) for rv in rvs]
    else:
      key = rv.ret_name()
      assert isinstance(rv, RV), \
          "Input not a RV instance but of type: {}".format(type(rv))
      assert key not in self._rvs.keys(), \
          "Existing RV name {} already present in collection".format(rv_name)
      self._rvs.update({key: rv})
    self._nrvs = len(self._rvs)
    self._keys = list(self._rvs.keys())
    self._keyset = set(self._keys)
    self._name = ','.join(self._keys)
    return self._nrvs
  
#-------------------------------------------------------------------------------
  def ret_rvs(self, aslist=True):
    # Defaulting aslist=True plays more nicely with inheriting classes
    rvs = self._rvs
    if aslist:
      if isinstance(rvs, dict):
        rvs = list(rvs.values())
      assert isinstance(rvs, list), "RVs not a recognised variable type: {}".\
                                    format(type(rvs))
    return rvs

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def ret_nrvs(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def set_ptype(self, ptype=None):
    if ptype is not None or not self._nvrs:
      self._ptype = eval_ptype(ptype)
      return self._ptype
    rvs = self.ret_rvs(aslist=True)
    ptypes = [rv.ret_ptype() for rv in rvs]
    self._ptype = prod_ptypes(ptypes)
    return self._ptype

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)
    self._arg_order = None
    if 'order' not in self._prob_kwds:
      return 
    assert self._prob is not None, "No order without specifying prob"
    self._arg_order = self._prob_kwds.pop(self._prob_kwds)
    if self._arg_order is None:
      return
    assert isinstance(self._arg_order, dict), "Keyword order must be a dict"

    # Sanity check the order dictionary
    key_list = list(self._arg_order.keys())
    ind_list = list(self._arg_order.values())
    self.check_keys(key_list, ind_list)

    assert isinstance(ind_list, list), "Input ind_list must be a list"
    keys = []
    inds = []
    for key, ind in zip(key_list, ind_list):
      keys.append(key)
      if type(ind) is int:
        inds.append(ind)
      else:
        raise TypeError("Cannot interpret order value: {}".ind)
    keyset = set(keys)
    assert keyset == self._keyset, \
        "RV name {} mismatch with order keys {}".format(keyset, self._keyset)
    indset = set(inds)
    assert indset == set(range(self._nvrs)), \
        "Index specification insuffient: {}".format(indset)
    return keyset, indset

#-------------------------------------------------------------------------------
  def fuse_dict(self, val_dict=None, def_val=None):
    if not val_dict:
      return {key: def_val for key in self._keys}
    fused = dict(val_dict)
    keys = []
    for key in fused.keys():
      if ',' in key:
        keys.extend(key_split)
      else:
        keys.append(key)
    for key in keys:
      assert key in self._keys, "Unknown key: {}".format(key)
    for key in self._keys:
      if key not in keys:
        fused.update({key: def_val})
    return fused

#-------------------------------------------------------------------------------
  def eval_vals(self, *args, **kwds):
    """ This ignores self._prob and self._arg_order """
    values = None
    if not len(args):
      if len(kwds):
        values = self.fuse_dict(kwds)
    else:
      assert not len(kwds), "Please input args or kwds but no both"
      if len(args) == 1 and isinstance(args[0], dict):
        values = self.fuse_dict(args[0])
      else:
        assert len(args) == self._nrvs, \
            "Number of positional arguments must match number of RVs"
        values = {key: arg for key, arg in zip(self._keys, args)}
    
    # Don't reshape if all scalars (and therefore by definition no joint keys)
    if all([np.isscalar(value) for value in values.values()]):
      return values

    # Reduce dimensionality based on joint variables and scalars
    dimensionality = {key: i for i, key in enumerate(self._keys)}
    values_ref = {key: [key, None] for key in self._keys}
    seen_keys = []
    for i, key in enumerate(self._keys):
      rem_keys = self._keys[(i+1):]
      if key in values.keys():
        seen_keys.append(key)
        if np.isscalar(values[key]):
          for rem_key in rem_keys:
            dimensionality[rem_keys] -= 1
      elif key not in seen_keys:
        seen_keys.append(key)
        for val_key in value.keys():
          subkeys = val_key.split(',')
          matches = 0
          for j, subkey in enumerate(subkeys):
            for rem_key in rem_keys:
              if rem_key == val_key:
                values_ref[rem_key] = [val_key, j] 
              else:
                if rem_key in subkeys:
                  seen_keys.append(key)
                  values_ref[rem_key] = [val_key, j] 
                  matches += 1
                  dimensionality[rem_key] = dimensionality[key]
                else:
                  dimensionality[rem_key] -= matches

    # Reshape
    ndims = max(dimensionality.values()) + 1
    ones_ndims = np.ones(ndims, dtype=int)
    vals = {}
    rvs = self.ret_rvs(aslist=True)
    for i, rv in enumerate(rvs):
      key = rv.ret_name()
      reshape = True
      if key in values.keys():
        vals.update({key: values[key]})
        reshape = not np.isscalar(vals[key])
        if vals[key] is None or isinstance(vals[key], set):
          vals[key] = rv.eval_vals(vals[key])
      else:
        val_ref = values_ref[key]
        vals.update({key: values[val_ref[0]][val_ref[1]]})
        dist_dict.update({key: key + "={}"})
      if reshape:
        re_shape = np.copy(ones_ndims)
        re_dim = dimensionality[key]
        re_shape[re_dim] = vals[key].size
        vals[key] = vals[key].reshape(re_shape)
    return vals

#-------------------------------------------------------------------------------
  def eval_prob(self, values):
    assert isinstance(values, dict), "Input to eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())
    if self._prob is None:
      dists = tuple([rv(values[rv.ret_name()]) for rv in self.ret_rvs(aslist=True)])
      return marg_prod(*dists, check=False).prob
    if self.__callable:
      args = list(self._prob_args)
      kwds = dict(self._prob_kwds)
      if self._arg_order:
        vals = [None] * self._nrvs
        for key, val in self._arg_order.items():
          vals[val] = values[key]
        args = vals + args
        return self._prob(*tuple(args), **kwds)
      return self._prob(values, *tuple(args), **kwds)
    return self._prob

#-------------------------------------------------------------------------------
  def dist_dict(self, values=None):
    dist_dict = collections.OrderedDict()
    for key in self._keys:
      dist_str = None
      if values is None or not isinstance(values, dict):
        dist_str = key
      elif key not in values:
        dist_str = key
      else:
        if np.isscalar(values[key]):
          dics_str = "{}={}".format(key, values[key])
      if dist_str is None:
        dist_str = key + "=[]"
      dist_dict.update({key: dist_str})
    return dist_dict

#-------------------------------------------------------------------------------
  def set_use_vfun(self, use_vfun=True):
    self._use_vfun = io_use_vfun(use_vfun)
    return self._use_vfun

#-------------------------------------------------------------------------------
  def ret_use_vfun(self):
    return self._use_vfun

#-------------------------------------------------------------------------------
  def vfun_0(self, values, use_vfun=True):
    if not use_vfun:
      return values
    rvs = self.ret_rvs(aslist=True)
    for key, rv in zip(self._self._keys, rvs):
      if key in values:
        values[key] = rv.vfun_0(values[key], use_vfun)
    return values

#-------------------------------------------------------------------------------
  def vfun_1(self, values, use_vfun=True):
    if not use_vfun:
      return values
    rvs = self.ret_rvs(aslist=True)
    for key, rv in zip(self._keys, rvs):
      if key in values:
        values[key] = rv.vfun_1(values[key], use_vfun)
    return values

#-------------------------------------------------------------------------------
  def __call__(self, values=None, **kwds):  # Let's make this args ands kwds
    ''' 
    Returns a namedtuple of the rvs.
    '''
    if self._rvs is None:
      return None
    if values is None and len(kwds):
      values = dict(kwds)
    dist_dict = self.dist_dict(values)
    if not isinstance(values, dict):
      values = {key: values for key in self._keys}
    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    vals = self.vfun_1(vals, self._use_vfun[1])
    dist_name = ','.join(dist_dict.values())
    return Dist(dist_name, vals, prob, self._ptype)

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = self._keys[key]
    if isinstance(key, str):
      return self._rvs[key]
    raise TypeError("Unexpected key type: {}".format(key))

#-------------------------------------------------------------------------------
