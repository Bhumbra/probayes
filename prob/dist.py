# A module for realised probability distributions, a triple comprising 
# variable names, their values (vals), and respective probabilities (prob).

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.dist_ops import str_margcond, margcond_str, prod_dist
from prob.vtypes import isscalar
from prob.ptypes import eval_ptype, rescale, prod_ptype, prod_rule, iscomplex
from prob.manifold import Manifold

#-------------------------------------------------------------------------------
DIVISOR_MINIMUM = 1e-300

#-------------------------------------------------------------------------------
class Dist (Manifold):

  # Public
  prob = None   # Numpy array
  name = None   # Name of distribution
  marg = None   # Ordered dictionary of marginals: {key: name}
  cond = None   # Ordered dictionary of conditionals: key: name}

  # Protected
  _keyset = None        # Keys as set according to name
  _marg_scalarset = None # Set of marginal scalar keys
  _ptype = None         # Same convention as _Prob

#-------------------------------------------------------------------------------
  def __init__(self, name=None, vals=None, dims=None, prob=None, ptype=None):
    self.set_name(name)
    self.set_vals(vals, dims)
    self.set_prob(prob, ptype)

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    # Only the name is sensitive to what are marginal and conditional variables
    self.name = name
    self.marg, self.cond = str_margcond(self.name)
    self._keyset = set(self.marg).union(set(self.cond))
    return self._keyset

#-------------------------------------------------------------------------------
  def set_vals(self, vals=None, dims=None):
    argout = super().set_vals(vals, dims)
    self._marg_scalarset = set()
    if not self._keys:
      return argout
    for i, key in enumerate(self._keys):
      assert key in self._keyset, \
          "Value key {} not found among name keys {}".format(key, self._keyset)
      change_name = False
      if self._arescalars[i]:
        if key in self.marg.keys():
          self._marg_scalarset.add(key)
          if '=' not in self.marg[key]:
            change_name = True
            self.marg[key] = "{}={}".format(key, self.vals[key])
        elif key in self.cond.keys():
          if '=' not in self.cond[key]:
            change_name = True
            self.cond[key] = "{}={}".format(key, self.vals[key])
        else:
          raise ValueError("Variable {} not accounted for in name {}".format(
                            key, self.name))
    if change_name:
      self.name = margcond_str(self.marg, self.cond)
    return argout

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = eval_ptype(ptype)
    if self.prob is None:
      return self._ptype
    if self._isscalar:
      assert isscalar(self.prob), "Scalar vals with non-scalar prob"
    else:
      assert not isscalar(self.prob), "Non scalar values with scalar prob"
      assert self.ndim == self.prob.ndim, \
        "Mismatch in dimensionality between values {} and probabilities {}".\
        format(self.ndim, self.prob.ndim)
      assert np.all(np.array(self.shape) == np.array(self.prob.shape)), \
        "Mismatch in dimensions between values {} and probabilities {}".\
        format(self.shape, self.prob.shape)
    return self._ptype

#-------------------------------------------------------------------------------
  def marginalise(self, keys):
    # from p(A, key | B), returns P(A | B)
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    keys  = set(keys)
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    dim_delta = 0
    sum_axes = []
    for i, key in enumerate(self._keys):
      new_dim = None
      if key in keys:
        assert not self._arescalars[i], \
            "Cannot marginalise along scalar for key {}".format(key)
        sum_axes.append(self.dims[key])
        marg.pop(key)
        dim_delta += 1
      else:
        if not self._arescalars[i]:
          dims.update({key: self.dims[key] - dim_delta})
        vals.update({key:self.vals[key]})
    name = margcond_str(marg, cond)
    prob = rescale(self.prob, self._ptype, 1.)
    sum_prob = np.sum(prob, axis=tuple(sum_axes), keepdims=False)
    prob = rescale(sum_prob, 1., self._ptype)
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                ptype=self._ptype)

#-------------------------------------------------------------------------------
  def marginal(self, keys):
    # from p(A, key | B), returns P(key | B)
    if isinstance(keys, str):
      keys = [keys]
    keys = set(keys)
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    marginalise_keys = set()
    arescalars = []
    for i, key in enumerate(self._keys):
      isscalar = self._arescalars[i]
      marginal = key in keys
      if key in self.marg.keys():
        arescalars.append(isscalar)
        if not isscalar and not marginal:
          marginalise_keys.add(key)

    # If including any marginal scalars, must include all scalars
    if any(arescalars):
      assert self._marg_scalarset.issubset(keys), \
        "If evaluating marginal for key {}".format(key) + "," + \
        "must include all marginal scalars in {}".format(self._marg_scalarset)

    return self.marginalise(marginalise_keys)
        
#-------------------------------------------------------------------------------
  def conditionalise(self, keys):
    # from P(A, key | B), returns P(A | B, key).
    # if vals[key] is a scalar, this effectively normalises prob
    if isinstance(keys, str):
      keys = [keys]
    keys = set(keys)
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    swap = [None] * self.ndim
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    dims = collections.OrderedDict()
    cond_dims = collections.OrderedDict()
    cond_dim0 = self.ndim - len(keys)
    sum_axes = []
    dim_delta = 0
    arescalars = []
    for i, key in enumerate(self._keys):
      new_dim = None
      if key in keys:
        cond.update({key:marg.pop(key)})
        arescalars.append(self._arescalars[i])
        if not arescalars[-1]:
          old_dim = self.dims[key]
          new_dim = cond_dim0 + dim_delta
          cond_dims.update({key: new_dim})
          dim_delta += 1
      elif not self._arescalars[i]:
        old_dim = self.dims[key]
        new_dim = old_dim - dim_delta
        dims.update({key: new_dim})
        # Axes to return key's marginal distribution not its marginalisation
        if key in self.marg.keys():
          sum_axes.append(new_dim)
      if not self._arescalars[i]:
        swap[old_dim] = new_dim

    # If including any marginal scalars, normalising must include all scalars
    normalise = any(arescalars)
    if normalise:
      assert self._marg_scalarset.issubset(set(keys)), \
        "If conditionalising for key {}".format(key) + "," + \
        "must include all marginal scalars in {}".format(self._marg_scalarset)

    # Setup vals dimensions and evaluate probabilities
    dims.update(cond_dims)
    name = margcond_str(marg, cond)
    vals = self.redim(dims).vals
    prob = rescale(self.prob, self._ptype, 1.)
    prob = np.moveaxis(prob, [*range(self.ndim)], swap)
    if normalise:
      prob = prob / np.maximum(DIVISOR_MINIMUM, np.sum(prob))
    if len(sum_axes):
      prob = prob / np.maximum(DIVISOR_MINIMUM,
                        np.sum(prob, axis=tuple(sum_axes), keepdims=True))
    prob = rescale(prob, 1., self._ptype)
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                ptype=self._ptype)

#-------------------------------------------------------------------------------
  def prod(self, keys):
    # from P(A, key | B), returns P(A, {} | B)
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    keys  = set(keys)
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    dim_delta = 0
    prod_axes = []
    for i, key in enumerate(self._keys):
      new_dim = None
      if key in keys:
        assert not self._arescalars[i], \
            "Cannot apply product along scalar for key {}".format(key)
        prod_axes.append(self.dims[key])
        marg.update({key: key+"={}"})
        vals.update({key: {self.vals[key].size}})
        dim_delta += 1
      else:
        if not self._arescalars[i]:
          dims.update({key: self.dims[key] - dim_delta})
        vals.update({key:self.vals[key]})
    name = margcond_str(marg, cond)
    ptype = self._ptype
    ptype_product = ptype
    if ptype_product not in [0., 1.]:
      ptype_scaling = np.prod(np.array(self.shape)[prod_axes])
      if iscomplex(ptype):
        ptype_product += ptype*ptype_scaling 
      else:
        ptype_product *= ptype**ptype_scaling 
    prob = np.sum(self.prob, axis=tuple(prod_axes)) if iscomplex(ptype) \
           else np.prod(self.prob, axis=tuple(prod_axes))
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                ptype=ptype_product)

#-------------------------------------------------------------------------------
  def ret_keyset(self):
    return self._keyset

#-------------------------------------------------------------------------------
  def ret_marg_names(self):
    return list(self.marg.keys())

#-------------------------------------------------------------------------------
  def ret_cond_names(self):
    return list(self.cond.keys())

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def rescale(self, ptype=None):
    self.set_prob(rescale(self.prob, self._ptype, ptype), ptype)
    return self.prob

#-------------------------------------------------------------------------------
  def __call__(self, values):
    # Slices distribution according to scalar values given as a dictionary

    assert isinstance(values, dict),\
        "Values must be dict type, not {}".format(type(values))
    keys = values.keys()
    keyset = set(values.keys())
    assert len(keyset.union(self._keyset)) == len(self._keyset),\
        "Unrecognised key among values keys: {}".format(keys())
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    dims = collections.OrderedDict(self.dims)
    inds = collections.OrderedDict()
    vals = collections.OrderedDict(self.vals)
    slices = [None] * self.ndim
    dim_delta = 0
    for i, key in enumerate(self._keys):
      isscalar = self._arescalars[i]
      dimension = self.dims[key]
      if key in keyset:
        inds.update({key: None})
        assert np.isscalar(values[key]), \
            "Values must contain scalars but found {} for {}".\
            format(values[key], key)
        vals[key] = values[key]
        if isscalar:
          if self.vals[key] == values[key]:
            inds[key] = 0
        else:
          dim_delta += 1
          dims[key] = None
          index = np.nonzero(np.ravel(self.vals[key]) == values[key])[0]
          if len(index):
            inds[key] = index[0]
            slices[dimension] = index[0]
        if key in marg.keys():
          marg[key] = "{}={}".format(key, values[key])
        elif key in cond.keys():
          cond[key] = "{}={}".format(key, values[key])
      elif not isscalar:
        dims[key] = dims[key] - dim_delta
        slices[dimension] = slice(self.shape[dimension])
    name = margcond_str(marg, cond)
    prob = None
    if not any(idx is None for idx in inds.values()):
      prob = self.prob[tuple(slices)]
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                ptype=self._ptype)

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return prod_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __add__(self, other):
    return sum_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._ptype) else 'p'
    suffix = '' if not self._isscalar else '={}'.format(self.prob)
    return super().__repr__() + ": " + prefix + "(" + self.name + ")" + suffix

#-------------------------------------------------------------------------------
