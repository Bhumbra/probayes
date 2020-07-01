# A distribution is triple comprising variable names, their values (vals), and respective probabilities.

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.dist_ops import str_margcond, margcond_str, prod_dist
from prob.vtypes import isscalar
from prob.ptypes import eval_ptype, rescale, prod_ptype, prod_rule, iscomplex
from prob.ptypes import NEARLY_POSITIVE_ZERO
from prob.manifold import Manifold

#-------------------------------------------------------------------------------
class Dist (Manifold):

  # Public
  prob = None   # Numpy array
  name = None   # Name of distribution
  marg = None   # Ordered dictionary of marginals: {key: name}
  cond = None   # Ordered dictionary of conditionals: key: name}

  # Protected
  _keyset = None # Keys as set according to name
  _ptype = None  # Same convention as _Prob

#-------------------------------------------------------------------------------
  def __init__(self, name=None, vals=None, dims=None, prob=None, ptype=None):
    self.set_name(name)
    self.set_vals(vals, dims)
    self.set_prob(prob, ptype)

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    self.name = name
    self.marg, self.cond = str_margcond(self.name)
    self._keyset = set(self.marg).union(set(self.cond))
    return self._keyset

#-------------------------------------------------------------------------------
  def set_vals(self, vals=None, dims=None):
    argout = super().set_vals(vals, dims)
    if not self._keys:
      return argout
    for key in self._keys:
      assert key in self._keyset, \
          "Value key {} not found among name keys {}".format(key, self._keyset)
    return argout

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = eval_ptype(ptype)
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
    marg_name = margcond_str(marg, cond)
    prob = rescale(self.prob, self._ptype, 1.)
    sum_prob = np.sum(prob, axis=tuple(sum_axes), keepdims=False)
    prob = rescale(sum_prob, 1., self._ptype)
    return Dist(name=marg_name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                ptype=self._ptype)

#-------------------------------------------------------------------------------
  def conditionalise(self, keys):
    # from P(A, key | B), returns P(A | B, key)
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    keys  = set(keys)
    swap = [None] * self.ndim
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    dims = collections.OrderedDict()
    cond_dims = collections.OrderedDict()
    cond_dim0 = self.ndim - len(keys)
    dim_delta = 0
    for i, key in enumerate(self._keys):
      new_dim = None
      if key in keys:
        assert not self._arescalars[i], \
            "Cannot conditionalised along scalar for key {}".format(key)
        cond.update({key:marg.pop(key)})
        old_dim = self.dims[key]
        new_dim = cond_dim0 + dim_delta
        cond_dims.update({key: new_dim})
        dim_delta += 1
      elif not self._arescalars[i]:
        old_dim = self.dims[key]
        new_dim = old_dim - dim_delta
        dims.update({key: new_dim})
      swap[old_dim] = new_dim
    dims.update(cond_dims)
    cond_name = margcond_str(marg, cond)
    redim = self.redim(dims)
    prob = rescale(self.prob, self._ptype, 1.)
    prob = np.moveaxis(prob, [*range(self.ndim)], swap)
    sum_axes = tuple(cond_dims.values())
    sum_prob = np.maximum(NEARLY_POSITIVE_ZERO, 
                      np.sum(prob, axis=sum_axes, keepdims=True))
    prob = rescale(prob / sum_prob, 1., self._ptype)
    return Dist(name=cond_name, 
                vals=redim.vals, 
                dims=dims, 
                prob=prob, 
                ptype=self._ptype)

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
    # Slices distribution according to dictionary in values
    pass

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return prod_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __add__(self, other):
    return sum_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._ptype) else 'p'
    return super().__repr__() + ": " + prefix + "(" + self.name + ") [vals,prob]"

#-------------------------------------------------------------------------------
