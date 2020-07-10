# A module for realised probability distributions, a triple comprising 
# variable names, their values (vals), and respective probabilities (prob).

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.dist_ops import str_margcond, margcond_str, product
from prob.vtypes import isscalar
from prob.pscales import eval_pscale, rescale, iscomplex
from prob.pscales import prod_pscale, prod_rule, div_prob
from prob.manifold import Manifold

#-------------------------------------------------------------------------------
class Dist (Manifold):

  # Public
  prob = None   # Numpy array
  name = None   # Name of distribution
  marg = None   # Ordered dictionary of marginals: {key: name}
  cond = None   # Ordered dictionary of conditionals: key: name}

  # Protected
  _keyset = None         # Keys as set according to name
  _marg_scalarset = None # Set of marginal scalar keys
  _cond_scalarset = None # Set of condition scalar keys
  _pscale = None         # Same convention as _Prob

#-------------------------------------------------------------------------------
  def __init__(self, name=None, vals=None, dims=None, prob=None, pscale=None):
    self.set_name(name)
    self.set_vals(vals, dims)
    self.set_prob(prob, pscale)

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
    self._cond_scalarset = set()
    if not self._keys or not any(self._arescalars):
      return argout

    # Override name entries for scalar values
    for i, key in enumerate(self._keys):
      assert key in self._keyset, \
          "Value key {} not found among name keys {}".format(key, self._keyset)
      if self._arescalars[i]:
        if key in self.marg.keys():
          self._marg_scalarset.add(key)
          self.marg[key] = "{}={}".format(key, self.vals[key])
        elif key in self.cond.keys():
          self._cond_scalarset.add(key)
          self.cond[key] = "{}={}".format(key, self.vals[key])
        else:
          raise ValueError("Variable {} not accounted for in name {}".format(
                            key, self.name))
    self.name = margcond_str(self.marg, self.cond)
    return argout

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, pscale=None):
    self.prob = prob
    self._pscale = eval_pscale(pscale)
    if self.prob is None:
      return self._pscale
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
    return self._pscale

#-------------------------------------------------------------------------------
  def ret_vals(self, keys=None):
    keys = keys or self._keys
    keys = set(keys)
    vals = collections.OrderedDict()
    seen_keys = set()
    for i, key in enumerate(self._keys):
      if key in keys and key not in seen_keys:
        if self._arescalars[i]:
          seen_keys.add(keys)
          vals.update({key: vals})
        else:
          shared_keys = [key]
          for j, cand_key in enumerate(self._keys):
            if j > i and cand_key in keys and not self._arescalars[j]:
              if self.dims[key] == self.dims[cand_key]:
                shared_keys.append(cand_key)
          if len(shared_keys) == 1:
            vals.update({key: np.ravel(self.vals[key])})
            seen_keys.add(key)
          else:
            val = [None] * len(shared_keys)
            for j, shared_key in enumerate(shared_keys):
              val[j] = np.ravel(self.vals[shared_key])
              seen_keys.add(shared_key)
            vals.update({','.join(shared_keys): tuple(val)})
    return vals

#-------------------------------------------------------------------------------
  def ret_marg_vals(self):
    return self.ret_vals(self.marg.keys())

#-------------------------------------------------------------------------------
  def ret_cond_vals(self):
    assert self.cond, "No conditioning variables"
    return self.ret_vals(self.cond.keys())

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
    sum_axes = set()
    for i, key in enumerate(self._keys):
      new_dim = None
      if key in keys:
        assert not self._arescalars[i], \
            "Cannot marginalise along scalar for key {}".format(key)
        sum_axes.add(self.dims[key])
        marg.pop(key)
        dim_delta += 1
      else:
        if not self._arescalars[i]:
          dims.update({key: self.dims[key] - dim_delta})
        vals.update({key:self.vals[key]})
    name = margcond_str(marg, cond)
    prob = rescale(self.prob, self._pscale, 1.)
    sum_prob = np.sum(prob, axis=tuple(sum_axes), keepdims=False)
    prob = rescale(sum_prob, 1., self._pscale)
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                pscale=self._pscale)

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
    dims = collections.OrderedDict()
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    normalise = False
    delta = 0
    for i, key in enumerate(self._keys):
      if key in keys:
        cond.update({key: marg.pop(key)})
      if self._arescalars[i]:
        dims.update({key: None})
        if key in keys:
          normalise = True
      elif key in self.marg.keys():
        if key in keys:
          delta += 1 # Don't add to dim just yet
        else:
          dim = self.dims[key]
          dims.update({key: dim})
      else:
        dim = self.dims[key] - delta
        dims.update({key: dim})

    # Reduce remaining marginals to lowest dimension
    dim_val = [val for val in dims.values() if val is not None]
    dim_max = 0
    if len(dim_val):
      dim_min = min(dim_val)
      for key in dims.keys():
        if dims[key] is not None:
          dim = dims[key]-dim_min
          dims.update({key: dim})
          dim_max = max(dim_max, dim)
    dim_min = self.ndim
    for key in keys:
      dim = self.dims[key]
      if dim is not None:
        dim_min = min(dim_min, dim)
    for key in keys:
      dim = self.dims[key]
      if dim is not None:
        dims.update({key: dim-dim_min+dim_max+1})
    if normalise:
      assert self._marg_scalarset.issubset(set(keys)), \
        "If conditionalising for key {}".format(key) + "," + \
        "must include all marginal scalars in {}".format(self._marg_scalarset)
    # Setup vals dimensions and evaluate probabilities
    name = margcond_str(marg, cond)
    vals = self.redim(dims).vals
    old_dims = []
    new_dims = []
    sum_axes = set()
    for key in self._keys:
      old_dim = self.dims[key]
      if old_dim is not None and old_dim not in old_dims:
        old_dims.append(old_dim)
        new_dims.append(dims[key])
        if key not in keys and key in self.marg.keys():
          sum_axes.add(dims[key])
    prob = np.moveaxis(self.prob, old_dims, new_dims)
    prob = rescale(prob, self._pscale, 1.)
    if normalise:
      prob = div_prob(prob, np.sum(prob))
    if len(sum_axes):
      prob = div_prob(prob, \
                         np.sum(prob, axis=tuple(sum_axes), keepdims=True))
    prob = rescale(prob, 1., self._pscale)
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                pscale=self._pscale)

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
    pscale = self._pscale
    pscale_product = pscale
    if pscale_product not in [0., 1.]:
      pscale_scaling = np.prod(np.array(self.shape)[prod_axes])
      if iscomplex(pscale):
        pscale_product += pscale*pscale_scaling 
      else:
        pscale_product *= pscale**pscale_scaling 
    prob = np.sum(self.prob, axis=tuple(prod_axes)) if iscomplex(pscale) \
           else np.prod(self.prob, axis=tuple(prod_axes))
    return Dist(name=name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                pscale=pscale_product)

#-------------------------------------------------------------------------------
  def expectation(self, keys=None, exponent=None):
    keys = keys or self.marg.keys()
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    keys = set(keys)
    sum_axes = []
    dims = collections.OrderedDict(self.dims)
    for i, key in enumerate(self._keys):
      if key in keys:
        sum_axes.append(self.dims[key])
        dims[key] = None
    prob = rescale(self.prob, self._pscale, 1.)
    sum_prob = np.sum(prob, axis=tuple(set(sum_axes)), keepdims=False)
    vals = collections.OrderedDict()
    for i, key in enumerate(self._keys):
      if key in keys:
        val = self.vals[key] if not exponent else self.vals[key]**exponent
        if self._arescalars[i]:
          vals.update({key: val})
        else:
          expt_numerator = np.sum(prob*val, 
                                  axis=tuple(set(sum_axes)), keepdims=False)
          vals.update({key: div_prob(expt_numerator, sum_prob)})
      elif key in self.cond.keys():
        vals.update({key: self.vals[key]})
    return vals

#-------------------------------------------------------------------------------
  def ret_keyset(self):
    return self._keyset

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    return self._pscale

#-------------------------------------------------------------------------------
  def ret_marg_scalarset(self):
    return self._marg_scalarset

#-------------------------------------------------------------------------------
  def ret_cond_scalarset(self):
    return self._cond_scalarset

#-------------------------------------------------------------------------------
  def rescaled(self, pscale=None):
    prob = rescale(np.copy(self.prob), self._pscale, pscale)
    return Dist(name=self.name, 
                vals=self.vals, 
                dims=self.dims,
                prob=prob, 
                pscale=pscale)

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
                pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return product(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __add__(self, other):
    return sum_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    """ If self is P(A, B | C, D), and other is P(A | C, D), this function
    returns P(B | C, D, A) subject to the following conditions:
    The divisor must be a scalar.
    The conditionals must match.
    The scalar marginals must match.
    """
    # Assert scalar division and operands compatible
    assert set(list(self.cond.keys())) == other.ret_cond_scalarset(), \
      "Conditionals must match"

    divs = other.ret_isscalar()
    if divs:
      assert self._marg_scalarset == other.ret_marg_scalarset(), \
        "For divisor scalars, scalar marginals must match"

    # Prepare quotient marg and cond keys
    keys = other.marg.keys()
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    vals = collections.OrderedDict(self.cond)
    re_shape = np.ones(self.ndim, dtype=int)
    for i, key in enumerate(self._keys):
      if key in keys:
        cond.update({key:marg.pop(key)})
        if not self._arescalars[i] and not divs:
          re_shape[self.dims[key]] = other.vals[key].size
      else:
        vals.update({key:self.vals[key]})

    # Append the marginalised variables and end of vals
    for i, key in enumerate(self._keys):
      if key in keys:
        vals.update({key:self.vals[key]})

    # Evaluate probabilities
    name = margcond_str(marg, cond)
    divp = other.prob if divs else other.prob.reshape(re_shape)
    prob = div_prob(self.prob, divp, self._pscale, other.ret_pscale())
    return Dist(name=name, 
                vals=vals, 
                dims=self.dims, 
                prob=prob, 
                pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._pscale) else 'p'
    suffix = '' if not self._isscalar else '={}'.format(self.prob)
    return super().__repr__() + ": " + prefix + "(" + self.name + ")" + suffix

#-------------------------------------------------------------------------------
