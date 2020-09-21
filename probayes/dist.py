# A module for realised probability distributions, a triple comprising 
# variable names, their values (vals), and respective probabilities (prob).

#-------------------------------------------------------------------------------
import collections
import numpy as np
from probayes.dist_utils import str_margcond, margcond_str, product, summate, \
                                rekey_dict, ismonotonic
from probayes.vtypes import issingleton, isscalar
from probayes.pscales import eval_pscale, rescale, iscomplex
from probayes.pscales import prod_pscale, prod_rule, div_prob
from probayes.manifold import Manifold

#-------------------------------------------------------------------------------
class Dist (Manifold):

  # Public
  prob = None   # Numpy array
  name = None   # Name of distribution
  marg = None   # Ordered dictionary of marginals: {key: name}
  cond = None   # Ordered dictionary of conditionals: key: name}

  # Protected
  _keyset = None         # Keys as set according to name
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
    if not self._keys or not any(self._aresingleton):
      return argout

    # Override name entries for scalar values
    for i, key in enumerate(self._keys):
      assert key in self._keyset, \
          "Value key {} not found among name keys {}".format(key, self._keyset)
      if self._aresingleton[i]:
        if key in self.marg.keys():
          self.marg[key] = "{}={}".format(key, self.vals[key])
        elif key in self.cond.keys():
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
    if self._issingleton:
      assert isscalar(self.prob), "Singleton vals with non-scalar prob"
    else:
      assert not isscalar(self.prob), "Non singleton values with scalar prob"
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
        if self._aresingleton[i]:
          seen_keys.add(key)
          vals.update({key: self.vals[key]})
        else:
          shared_keys = [key]
          for j, cand_key in enumerate(self._keys):
            if j > i and cand_key in keys and not self._aresingleton[j]:
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
        assert not self._aresingleton[i], \
            "Cannot marginalise along scalar for key {}".format(key)
        sum_axes.add(self.dims[key])
        marg.pop(key)
        dim_delta += 1
      else:
        if not self._aresingleton[i]:
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

    # Check keys arg marginal
    keys = set(keys)
    dims = set()
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
      dim = self.dims[key]
      if dim is not None:
        dims.add(dim)

    # Check consistency of marginal dims
    for key in self._keys:
      dim = self.dims[key]
      if dim in dims:
        assert key in keys, \
            "Dimensionality precludes marginalising {} without: {}".\
            format(keys, key)

    # Determine keys to marginalise by exclusion
    marginalise_keys = set()
    aresingletons = []
    marg_scalars = set()
    for i, key in enumerate(self._keys):
      singleton = self._aresingleton[i]
      marginal = key in keys
      if key in self.marg.keys():
        aresingletons.append(singleton)
        if singleton:
          marg_scalars.add(key)
        if not singleton and not marginal:
          marginalise_keys.add(key)

    # If including any marginal scalars, must include all scalars
    if any(aresingletons):
      assert marg_scalars.issubset(keys), \
        "If evaluating marginal for key {}".format(key) + ", " + \
        "must include all marginal scalars in {}".format(self.marg.keys())

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
    marg_scalars = set()
    for i, key in enumerate(self._keys):
      if key in keys:
        cond.update({key: marg.pop(key)})
      if self._aresingleton[i]:
        dims.update({key: None})
        if key in keys:
          normalise = True
      elif key in self.marg.keys():
        if self._aresingleton[i]:
          marg_scalars.add(key)
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
      assert marg_scalars.issubset(set(keys)), \
        "If conditionalising for key {}".format(key) + "," + \
        "must include all marginal scalars in {}".format(self.marg.keys())

    # Setup vals dimensions and evaluate probabilities
    name = margcond_str(marg, cond)
    vals = super().redim(dims).vals
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
    if normalise and iscomplex(self._pscale): 
      prob = prob - prob.max()
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
  def redim(self, dims):
    """ 
    Returns a distribution according to redimensionised values in dims, index-
    ordered by the order in dims
    """
    manifold = super().redim(dims)
    vals, dims = manifold.vals, manifold.dims
    prob = self.prob

    # Need to realign prob axes to new dimensions
    if not self._issingleton:
      old_dims = []
      new_dims = []
      for i, key in enumerate(self._keys):
        if not self._aresingletons[i]:
          old_dims.append(self._dims[key])
          new_dims.append(dims[key])
      prob = np.moveaxis(prob, old_dims, new_dims)

    return Dist(name=self._name, 
                vals=vals, 
                dims=dims, 
                prob=prob, 
                pscale=self._pscale)

#-------------------------------------------------------------------------------
  def rekey(self, keymap):
    """
    Returns a distribution with modified key names without axes changes.
    """
    manifold = super().rekey(keymap)
    marg = rekey_dict(self.marg, keymap) 
    cond = rekey_dict(self.cond, keymap)
    name = margcond_str(marg, cond)
    return Dist(name=name, 
                vals=manifold.vals, 
                dims=manifold.dims, 
                prob=self.prob, 
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
        assert not self._aresingleton[i], \
            "Cannot apply product along scalar for key {}".format(key)
        if self.dims[key] not in prod_axes:
          prod_axes.append(self.dims[key])
          dim_delta += 1
        marg.update({key: key+"={}"})
        vals.update({key: {self.vals[key].size}})
      else:
        if not self._aresingleton[i]:
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
        if self.dims[key] is not None:
          sum_axes.append(self.dims[key])
        dims[key] = None
    prob = rescale(self.prob, self._pscale, 1.)
    if sum_axes:
      sum_prob = np.sum(prob, axis=tuple(set(sum_axes)), keepdims=False)
    else:
      sum_prob = np.sum(prob)
    vals = collections.OrderedDict()
    for i, key in enumerate(self._keys):
      if key in keys:
        val = self.vals[key] if not exponent else self.vals[key]**exponent
        if self._aresingleton[i]:
          vals.update({key: val})
        else:
          expt_numerator = np.sum(prob*val, 
                                  axis=tuple(set(sum_axes)), keepdims=False)
          vals.update({key: div_prob(expt_numerator, sum_prob)})
      elif key in self.cond.keys():
        vals.update({key: self.vals[key]})
    return vals

#-------------------------------------------------------------------------------
  def quantile(self, q=0.5):
    """ Returns probability quantiles in distribution for sorted values """
    quants = [q] if isscalar(q) else q

    # Deal with trivial scalar case
    if self._issingleton:
      quantiles = [self.vals] * len(quants)
      if isscalar(q):
        return quantiles[0]
      return quantiles

    # Check for monotonicity
    unsorted = set()
    for i, key in enumerate(self._keys):
      if not self._aresingleton[i]:
        if not ismonotonic(self.vals[key]):
          unsorted.add(key)

    # Evaluate quantiles from cumulative probability
    ravprob = rescale(np.ravel(self.prob), self._pscale, 1.)
    cumprob = np.cumsum(ravprob)
    cumprob = div_prob(cumprob, cumprob[-1])
    cum_idx = np.maximum(0, np.digitize(np.array(quants), cumprob)-1).tolist()

    # Interpolate in last axis
    quantiles = [None] * len(cum_idx)
    for j, _cum_idx in enumerate(cum_idx):
      rav_idx = int(_cum_idx)
      unr_idx = np.unravel_index(rav_idx, self.shape)
      quantiles[j] = collections.OrderedDict()
      for i, key in enumerate(self._keys):
        if self._aresingleton[i]:
          quantiles[j].update({key: self.vals[key]})
        elif key in unsorted:
          quantiles[j].update({key: {self.vals[key].size}})
        else:
          dim = self.dims[key]
          val = np.ravel(self.vals[key])
          idx = np.minimum(unr_idx[dim], len(val) - 1)
          if dim < self.ndim - 1 or idx == len(val) - 1:
            quantiles[j].update({key: val[idx]})
          else:
            vals = val[idx:idx+2]
            ravp = ravprob[rav_idx:rav_idx+2]
            if np.abs(np.diff(ravp)) < min(quants[j], 1. - quants[j]):
              cump = cumprob[rav_idx:rav_idx+2]
              interp_val = np.interp(quants[j], cump, vals)
              quantiles[j].update({key: interp_val})
            else:
              weighted_val = np.sum(ravp * vals) / np.sum(ravp)
              quantiles[j].update({key: weighted_val})
    if isscalar(q):
      return quantiles[0]
    return quantiles

#-------------------------------------------------------------------------------
  def sorted(self, key):
    """ Returns a distribution ordered by key """
    dim = self.dims[key]

    # Handle trivial case
    if dim is None:
      return Dist(self.name, self.vals, self.dims, self.prob, self._pscale)

    # Argsort needed to sort probabilities
    ravals = np.ravel(self.vals[key])
    keyidx = np.argsort(ravals)
    keyval = ravals[keyidx]

    # Reorder probabilities in affected dimension
    slices = [slice(i) for i in self.shape]
    slices[dim] = keyidx
    prob = self.prob[tuple(slices)]

    # Evaluate values arrays
    vals = collections.OrderedDict()
    for _key, _val in self.vals.items():
      if self.dims[_key] != dim:
        vals.update({_key: _val})
      elif _key == key:
        vals.update({_key: keyval})
      else:
        vals.update({_key: np.ravel(_val)[keyidx]})
    return Dist(self.name, vals, self.dims, prob, self._pscale)
    
#-------------------------------------------------------------------------------
  def rescaled(self, pscale=None):
    prob = rescale(np.copy(self.prob), self._pscale, pscale)
    return Dist(name=self.name, 
                vals=self.vals, 
                dims=self.dims,
                prob=prob, 
                pscale=pscale)

#-------------------------------------------------------------------------------
  def ret_keyset(self):
    return self._keyset

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    return self._pscale

 
#-------------------------------------------------------------------------------
  def __call__(self, values, keepdims=False):
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
    vals = collections.OrderedDict(self.vals)
    slices = [slice(None) for _ in range(self.ndim)]
    dim_delta = 0
    for i, key in enumerate(self._keys):
      check_dims = False
      if not self._aresingleton[i]:
        dim = self.dims[key]
        if key in keyset:
          assert np.isscalar(values[key]), \
              "Values must contain scalars but found {} for {}".\
              format(values[key], key)
          match = np.ravel(self.vals[key]) == values[key]
          n_matches = match.sum()
          post_eq = '{}'.format(values[key])
          if n_matches == 0:
            slices[dim] = slice(0, 0)
            vals[key] = np.array([])
          elif n_matches == 1 and not keepdims:
            dim_delta += 1
            slices[dim] = int(np.nonzero(match)[0])
            vals[key] = values[key]
            dims[key] = None
          else:
            post_eq = '[]'
            slices[dim] = np.nonzero(match)[0]
            vals[key] = self.vals[key][slices[dim]]
            check_dims = True
          update_keys = [key]
          if check_dims:
            for k, v in self.vals.items():
              if dim == self.dims[k] and k not in keyset:
                vals[k] = self.vals[k][slices[dim]]
                update_keys.append(k)
          for update_key in update_keys:
            if key in marg.keys():
              marg[key] = "{}={}".format(key, post_eq)
            elif key in cond.keys():
              cond[key] = "{}={}".format(key, post_eq)
        elif dim_delta:
          dims[key] = dims[key] - dim_delta
    name = margcond_str(marg, cond)
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
    return summate(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    """ If self is P(A, B | C, D), and other is P(A | C, D), this function
    returns P(B | C, D, A) subject to the following conditions:
    The divisor must be a scalar.
    The conditionals must match.
    The scalar marginals must match.
    """
    # Assert scalar division and operands compatible
    assert set(self.cond.keys())== set(other.cond.keys()),  \
      "Conditionals must match"

    divs = other.ret_issingleton()
    if divs:
      marg_scalars = set()
      for i, key in enumerate(self._keys):
        if key in self.marg.keys() and self._aresingleton[i]:
          marg_scalars.add(key)
      assert marg_scalars == set(other.marg.keys()), \
        "For divisor singletons, scalar marginals must match"

    # Prepare quotient marg and cond keys
    keys = other.marg.keys()
    marg = collections.OrderedDict(self.marg)
    cond = collections.OrderedDict(self.cond)
    vals = collections.OrderedDict(self.cond)
    re_shape = np.ones(self.ndim, dtype=int)
    for i, key in enumerate(self._keys):
      if key in keys:
        cond.update({key:marg.pop(key)})
        if not self._aresingleton[i] and not divs:
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
  def remarginalise(self, manifold, *args, **kwds):
    """ Redistributes probability distribution within manifold according to a
    dictionary (in args[0]) or keywords, for which the keys represent the
    marginal keys and corresponding values contain their corresponding values
    according to the shape of self.prob. Conditional variables are unchanged.
    """

    # Simple assertions
    assert not self._issingleton,\
        "Function dist.remarginalise() not operative for scalar probabilities"
    assert isinstance(manifold, Manifold),\
        "First argument must be Manifold type not {}".format(type(manifold))

    # Read and check dictionary
    vals = None
    if args:
      assert len(args) == 1 and not kwds, \
          "Use a single dictionary argument or keywords, not both"
      vals = args[0]
    elif kwds:
      vals = dict(kwds)
    assert isinstance(vals, dict), \
        "Dictionary argument expected, not {}".format(type(vals))
    marg_keys = list(manifold.vals.keys())
    dims = manifold.dims
    for key, val in vals.items():
      if dims[key] is not None:
        assert np.all(np.array(val.shape) == np.array(self.shape)),\
            "Values for {} incommensurate".format
        assert key in marg_keys, \
            "Key {} not present in inputted manifold"

    # Evaluate output indices for each space in probability
    shape = manifold.shape
    indices = [np.empty(_size, dtype=int) for _size in shape]
    for key, dim in dims.items():
      if dim is not None:
        edges = np.array(np.ravel(manifold.vals[key]), dtype=float)
        rav_vals = np.array(np.ravel(vals[key]), dtype=float)
        indices[dim] = np.maximum(0, np.minimum(len(edges)-1, \
                                     np.digitize(rav_vals, edges)-1))

    # Iterate through every self.prob value
    rav_prob = np.ravel(rescale(self.prob, self._pscale, 1.))
    rem_prob = np.zeros(manifold.shape, dtype=float)
    for i in range(self.size):
      rem_idx = tuple([idx[i] for idx in indices])
      rem_prob[rem_idx] += rav_prob[i]

    # Replace marginal keys and keep conditional keys
    rem_vals = collections.OrderedDict(manifold.vals)
    rem_dims = collections.OrderedDict(manifold.dims)
    marg_str = []
    for key, val in manifold.vals.items():
      marg_str.append(key)
      if val is not None:
        if np.isscalar(val):
          marg_str[-1] = "{}={}".format(key, val)
        else:
          marg_str[-1] = key + "=[]"
    rem_name = ','.join(marg_str)
    if '|' in self.name:
      rem_name += self.name.split('|')[1]
    cond_keys = list(self.cond.keys())
    if cond_keys:
      for key in cond_keys:
        rem_vals.update({key: self.vals[key]})
        rem_dims.update({key: self.dims[key]})

    return Dist(rem_name,
                rem_vals,
                rem_dims,
                rescale(rem_prob, 1., self._pscale),
                self._pscale)

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._pscale) else 'p'
    suffix = '' if not self._issingleton else '={}'.format(self.prob)
    return super().__repr__() + ": " + prefix + "(" + self.name + ")" + suffix

#-------------------------------------------------------------------------------
