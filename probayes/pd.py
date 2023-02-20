""" Provides probability distribution funciontlaity based on Distribution """

#-------------------------------------------------------------------------------
import collections
import numpy as np
from probayes.named_dict import NamedDict
from probayes.pd_utils import str_margcond, margcond_str, product, summate, \
                              rekey_dict, ismonotonic
from probayes.vtypes import isscalar
from probayes.pscales import eval_pscale, rescale, iscomplex
from probayes.pscales import div_prob
from probayes.distribution import Distribution

#-------------------------------------------------------------------------------
class PD (Distribution):
  """ A probability distribution is a distribution with corresponding 
  probabilities. The dimensions of the probability scalar or array must be
  commensurate with the values of the distribution according to their assigned
  dimensions. 

  While it is intended for PD instances to come from RV, RF, SD, or SP calls,
  PDs can be instantiated directly.
  """
  # Protected
  _prob = None    # Probability
  _pscale = None  # Same convention as Prob()
  _marg = None    # Ordered dictionary of marginals: {key: name}
  _cond = None    # Ordered dictionary of conditionals: key: name}

#-------------------------------------------------------------------------------
  def __init__(self, name, *args, **kwds):
    """ Initialises the PD with a name, args, and kwds in the same way as 
    Distribution(), except with the following reserved keywords:
      
      'dims': sets the dimensionality. 
      'prob': sets the probability scalar or array. 
      'pscale': sets the pscale
    """
    args = tuple(args)
    kwds = dict(kwds)
    prob = None if 'prob' not in kwds else kwds.pop('prob')
    pscale = None if 'pscale' not in kwds else kwds.pop('pscale')
    super().__init__(name, *args, **kwds)
    self.pscale = pscale
    self.prob = prob

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @property
  def marg(self):
    return self._marg

  @property
  def cond(self):
    return self._cond

  @name.setter
  def name(self, name):
    # Only the name is sensitive to what are marginal and conditional variables
    NamedDict.name.fset(self, name)
    self._marg, self._cond = str_margcond(self.name)
    
  @property
  def short_name(self):
    marg_keys = list(self._marg.keys())
    cond_keys = list(self._cond.keys())
    marg_str = ','.join(marg_keys)
    cond_str = ','.join(cond_keys)
    return '|'.join([marg_str, cond_str]) if cond_str else marg_str

#-------------------------------------------------------------------------------
  @property
  def pscale(self):
    return self._pscale

  @pscale.setter
  def pscale(self, pscale=None):
    self._pscale = eval_pscale(pscale)

#-------------------------------------------------------------------------------
  @property
  def prob(self):
    return self._prob

  @prob.setter
  def prob(self, prob=None):
    self._prob = prob
    if self._prob is None:
      return
    if self._issingleton:
      assert isscalar(self._prob), "Singleton vals with non-scalar prob"
    else:
      assert not isscalar(self._prob), "Non singleton values with scalar prob"
      assert self._ndim == self._prob.ndim, \
        "Mismatch in dimensionality between values {} and probabilities {}".\
        format(self.ndim, self._prob.ndim)
      assert np.all(np.array(self._shape) == np.array(self._prob.shape)), \
        "Mismatch in dimensions between values {} and probabilities {}".\
        format(self._shape, self._prob.shape)

#-------------------------------------------------------------------------------
  @property
  def dims(self):
    return self._dims

  @dims.setter
  def dims(self, dims=None):
    """ Sets the dimensions for each of the variables.

    :param dims: a dictionary of {variable_name: variable_dim}

    The keys should correspond to that of a dictionary. If dims
    is None, then the dimensionality is set according to the
    order in values.
    """
    Distribution.dims.fset(self, dims)

    # Override name entries for scalar values
    for i, key in enumerate(self._keylist):
      assert key in self._keyset, \
          "Value key {} not found among name keys {}".format(key, self._keyset)
      if self._aresingleton[i]:
        if key in self.marg.keys():
          self.marg[key] = "{}={}".format(key, self[key])
        elif key in self.cond.keys():
          self.cond[key] = "{}={}".format(key, self[key])
        else:
          raise ValueError("Variable {} not accounted for in name {}".format(
                            key, self.name))
    self._name = margcond_str(self.marg, self.cond)

#-------------------------------------------------------------------------------
  def marginalise(self, keys):
    # from p(A, key | B), returns P(A | B)
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self._marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self._name)
    keys  = set(keys)
    marg = collections.OrderedDict(self._marg)
    cond = collections.OrderedDict(self._cond)
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    dim_delta = 0
    sum_axes = set()
    for i, key in enumerate(self._keylist):
      if key in keys:
        assert not self._aresingleton[i], \
            "Cannot marginalise along scalar for key {}".format(key)
        sum_axes.add(self._dims[key])
        marg.pop(key)
        dim_delta += 1
      else:
        if not self._aresingleton[i]:
          dims.update({key: self._dims[key] - dim_delta})
        vals.update({key:self[key]})
    name = margcond_str(marg, cond)
    prob = rescale(self._prob, self._pscale, 1.)
    sum_prob = np.sum(prob, axis=tuple(sum_axes), keepdims=False)
    prob = rescale(sum_prob, 1., self._pscale)
    return PD(name, vals, dims=dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def marginal(self, keys):
    # from p(A, key | B), returns P(key | B)
    if isinstance(keys, str):
      keys = [keys]

    # Check keys arg marginal
    keys = set(keys)
    dims = set()
    for key in keys:
      assert key in self._marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self._name)
      dim = self._dims[key]
      if dim is not None:
        dims.add(dim)

    # Check consistency of marginal dims
    for key in self._keylist:
      dim = self._dims[key]
      if dim in dims:
        assert key in keys, \
            "Dimensionality precludes marginalising {} without: {}".\
            format(keys, key)

    # Determine keys to marginalise by exclusion
    marginalise_keys = set()
    aresingletons = []
    marg_scalars = set()
    for i, key in enumerate(self._keylist):
      singleton = self._aresingleton[i]
      marginal = key in keys
      if key in self._marg.keys():
        aresingletons.append(singleton)
        if singleton:
          marg_scalars.add(key)
        if not singleton and not marginal:
          marginalise_keys.add(key)

    # If including any marginal scalars, must include all scalars
    if any(aresingletons):
      assert marg_scalars.issubset(keys), \
        "If evaluating marginal for key {}".format(key) + ", " + \
        "must include all marginal scalars in {}".format(self._marg.keys())

    return self.marginalise(marginalise_keys)
        
#-------------------------------------------------------------------------------
  def conditionalise(self, keys):
    # from P(A, key | B), returns P(A | B, key).
    # if vals[key] is a scalar, this effectively normalises prob
    if isinstance(keys, str):
      keys = [keys]
    keys = set(keys)
    for key in keys:
      assert key in self._marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    dims = collections.OrderedDict()
    marg = collections.OrderedDict(self._marg)
    cond = collections.OrderedDict(self._cond)
    normalise = False
    delta = 0
    marg_scalars = set()
    for i, key in enumerate(self._keylist):
      if key in keys:
        cond.update({key: marg.pop(key)})
      if self._aresingleton[i]:
        dims.update({key: None})
        if key in keys:
          normalise = True
      elif key in self._marg.keys():
        if self._aresingleton[i]:
          marg_scalars.add(key)
        if key in keys:
          delta += 1 # Don't add to dim just yet
        else:
          dim = self._dims[key]
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
      dim = self._dims[key]
      if dim is not None:
        dims.update({key: dim-dim_min+dim_max+1})
    if normalise:
      assert marg_scalars.issubset(set(keys)), \
        "If conditionalising for key {}".format(key) + "," + \
        "must include all marginal scalars in {}".format(self._marg.keys())

    # Setup vals dimensions and evaluate probabilities
    name = margcond_str(marg, cond)
    vals = collections.OrderedDict(super().redim(dims))
    old_dims = []
    new_dims = []
    sum_axes = set()
    for key in self._keylist:
      old_dim = self._dims[key]
      if old_dim is not None and old_dim not in old_dims:
        old_dims.append(old_dim)
        new_dims.append(dims[key])
        if key not in keys and key in self._marg.keys():
          sum_axes.add(dims[key])
    prob = np.moveaxis(self._prob, old_dims, new_dims)
    if normalise and iscomplex(self._pscale): 
      prob = prob - prob.max()
    prob = rescale(prob, self._pscale, 1.)
    if normalise:
      prob = div_prob(prob, np.sum(prob))
    if len(sum_axes):
      prob = div_prob(prob, \
                         np.sum(prob, axis=tuple(sum_axes), keepdims=True))
    prob = rescale(prob, 1., self._pscale)
    return PD(name, vals, dims=dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def redim(self, dims):
    """ 
    Returns a distribution according to redimensionised values in dims, index-
    ordered by the order in dims
    """
    dist = super().redim(dims)
    vals, dims = dict(dist), dist.dims
    prob = self._prob

    # Need to realign prob axes to new dimensions
    if not self._issingleton:
      old_dims = []
      new_dims = []
      for i, key in enumerate(self._keylist):
        if not self._aresingletons[i]:
          old_dims.append(self._dims[key])
          new_dims.append(dims[key])
      prob = np.moveaxis(prob, old_dims, new_dims)

    return PD(self._name, vals, dims=dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def rekey(self, keymap):
    """
    Returns a distribution with modified key names without axes changes.
    """
    dist = super().rekey(keymap)
    marg = rekey_dict(self._marg, keymap) 
    cond = rekey_dict(self._cond, keymap)
    name = margcond_str(marg, cond)
    return PD(name, dict(dist), dims=dist.dims, 
              prob=self.prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def prod(self, keys):
    # from P(A, key | B), returns P(A, {} | B)
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self.marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self.name)
    keys  = set(keys)
    marg = collections.OrderedDict(self._marg)
    cond = collections.OrderedDict(self._cond)
    vals = collections.OrderedDict()
    dims = collections.OrderedDict()
    dim_delta = 0
    prod_axes = []
    for i, key in enumerate(self._keylist):
      if key in keys:
        assert not self._aresingleton[i], \
            "Cannot apply product along scalar for key {}".format(key)
        if self._dims[key] not in prod_axes:
          prod_axes.append(self.dims[key])
          dim_delta += 1
        marg.update({key: key+"={}"})
        vals.update({key: {self[key].size}})
      else:
        if not self._aresingleton[i]:
          dims.update({key: self._dims[key] - dim_delta})
        vals.update({key:self[key]})
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
    return PD(name, vals, dims=dims, prob=prob, pscale=pscale_product)

#-------------------------------------------------------------------------------
  def expectation(self, keys=None, exponent=None):
    keys = keys or self.marg.keys()
    if isinstance(keys, str):
      keys = [keys]
    for key in keys:
      assert key in self._marg.keys(), \
        "Key {} not marginal in distribution {}".format(key, self._name)
    keys = set(keys)
    sum_axes = []
    dims = collections.OrderedDict(self._dims)
    for i, key in enumerate(self._keylist):
      if key in keys:
        if self._dims[key] is not None:
          sum_axes.append(self._dims[key])
        dims[key] = None
    prob = rescale(self._prob, self._pscale, 1.)
    if sum_axes:
      sum_prob = np.sum(prob, axis=tuple(set(sum_axes)), keepdims=False)
    else:
      sum_prob = np.sum(prob)
    vals = collections.OrderedDict()
    for i, key in enumerate(self._keylist):
      if key in keys:
        val = self[key] if not exponent else self[key]**exponent
        if self._aresingleton[i]:
          vals.update({key: val})
        else:
          expt_numerator = np.sum(prob*np.array(val, dtype=float), 
                                  axis=tuple(set(sum_axes)), keepdims=False)
          vals.update({key: div_prob(expt_numerator, sum_prob)})
      elif key in self._cond.keys():
        vals.update({key: self[key]})
    return vals

#-------------------------------------------------------------------------------
  def quantile(self, q=0.5):
    """ Returns probability quantiles in distribution for sorted values """
    quants = [q] if isscalar(q) else q

    # Deal with trivial scalar case
    if self._issingleton:
      quantiles = [collections.OrderedDict(self)] * len(quants)
      if isscalar(q):
        return quantiles[0]
      return quantiles

    # Check for monotonicity
    unsorted = set()
    for i, key in enumerate(self._keylist):
      if not self._aresingleton[i]:
        if not ismonotonic(self[key]):
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
      unr_idx = np.unravel_index(rav_idx, self._shape)
      quantiles[j] = collections.OrderedDict()
      for i, key in enumerate(self._keylist):
        if self._aresingleton[i]:
          quantiles[j].update({key: self[key]})
        elif key in unsorted:
          quantiles[j].update({key: {self[key].size}})
        else:
          dim = self._dims[key]
          val = np.ravel(self[key])
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
    dim = self._dims[key]

    # Handle trivial case
    if dim is None:
      return PD(self._name, dict(self), dims=self._dims, 
                prob=self._prob, pscale=self._pscale)

    # Argsort needed to sort probabilities
    ravals = np.ravel(self[key])
    keyidx = np.argsort(ravals)
    keyval = ravals[keyidx]

    # Reorder probabilities in affected dimension
    slices = [slice(i) for i in self._shape]
    slices[dim] = keyidx
    prob = self._prob[tuple(slices)]

    # Evaluate values arrays
    vals = collections.OrderedDict()
    for _key, _val in self.items():
      if self._dims[_key] != dim:
        vals.update({_key: _val})
      elif _key == key:
        vals.update({_key: keyval})
      else:
        vals.update({_key: np.ravel(_val)[keyidx]})
    return PD(self.name, vals, dims=self._dims, 
              prob=prob, pscale=self._pscale)
    
#-------------------------------------------------------------------------------
  def rescaled(self, pscale=None):
    prob = rescale(np.copy(self.prob), self._pscale, pscale)
    return PD(self.name, dict(self), dims=self.dims,
              prob=prob, pscale=pscale)

#-------------------------------------------------------------------------------
  def __call__(self, values, keepdims=False):
    # Slices distribution according to scalar values given as a dictionary

    if isinstance(values, str):
      assert values in ['marg', 'cond'], \
          "String call must be either 'marg' or 'cond'"
      dictionary = self.marg if values == 'marg' else self.cond
      return self.lookup(list(dictionary.keys()))
    assert isinstance(values, dict),\
        "Values must be dict type, not {}".format(type(values))
    keys = values.keys()
    keyset = set(values.keys())
    assert len(keyset.union(self._keyset)) == len(self._keyset),\
        "Unrecognised key among values keys: {}".format(keys())
    marg = collections.OrderedDict(self._marg)
    cond = collections.OrderedDict(self._cond)
    dims = collections.OrderedDict(self._dims)
    vals = collections.OrderedDict(self)
    slices = [slice(None) for _ in range(self.ndim)]
    dim_delta = 0
    for i, key in enumerate(self._keylist):
      check_dims = False
      if not self._aresingleton[i]:
        dim = self.dims[key]
        if key in keyset:
          assert np.isscalar(values[key]), \
              "Values must contain scalars but found {} for {}".\
              format(values[key], key)
          match = np.ravel(self[key]) == values[key]
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
            vals[key] = self[key][slices[dim]]
            check_dims = True
          update_keys = [key]
          if check_dims:
            for k, v in self.items():
              if dim == self._dims[k] and k not in keyset:
                vals[k] = self[k][slices[dim]]
                update_keys.append(k)
          for update_key in update_keys:
            if key in marg.keys():
              marg[key] = "{}={}".format(key, post_eq)
            elif key in cond.keys():
              cond[key] = "{}={}".format(key, post_eq)
        elif dim_delta:
          dims[key] = dims[key] - dim_delta
    name = margcond_str(marg, cond)
    prob = self._prob[tuple(slices)]
    return PD(name, vals, dims=dims, prob=prob, pscale=self._pscale)

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

    divs = other.issingleton
    if divs:
      marg_scalars = set()
      for i, key in enumerate(self._keylist):
        if key in self._marg.keys() and self._aresingleton[i]:
          marg_scalars.add(key)
      assert marg_scalars == set(other.marg.keys()), \
        "For divisor singletons, scalar marginals must match"

    # Prepare quotient marg and cond keys
    keys = other.marg.keys()
    marg = collections.OrderedDict(self._marg)
    cond = collections.OrderedDict(self._cond)
    vals = collections.OrderedDict(self._cond)
    re_shape = np.ones(self._ndim, dtype=int)
    for i, key in enumerate(self._keylist):
      if key in keys:
        cond.update({key:marg.pop(key)})
        if not self._aresingleton[i] and not divs:
          re_shape[self.dims[key]] = other[key].size
      else:
        vals.update({key:self[key]})

    # Append the marginalised variables and end of vals
    for i, key in enumerate(self._keylist):
      if key in keys:
        vals.update({key:self[key]})

    # Evaluate probabilities
    name = margcond_str(marg, cond)
    divp = other.prob if divs else other.prob.reshape(re_shape)
    prob = div_prob(self._prob, divp, self._pscale, other.pscale)
    return PD(name, vals, dims=self._dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def remarginalise(self, distribution, *args, **kwds):
    """ Redistributes probability distribution within distribution according to a
    dictionary (in args[0]) or keywords, for which the keys represent the
    marginal keys and corresponding values contain their corresponding values
    according to the shape of self.prob. Conditional variables are unchanged.
    """

    # Simple assertions
    assert not self._issingleton,\
        "Function dist.remarginalise() not operative for scalar probabilities"
    assert isinstance(distribution, Distribution),\
        "First argument must be Distribution type not {}".format(type(distribution))

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
    marg_keys = list(distribution.keys())
    dims = distribution.dims
    for key, val in vals.items():
      if dims[key] is not None:
        assert np.all(np.array(val.shape) == np.array(self.shape)),\
            "Values for {} incommensurate".format
        assert key in marg_keys, \
            "Key {} not present in inputted distribution"

    # Evaluate output indices for each space in probability
    shape = distribution.shape
    indices = [np.empty(_size, dtype=int) for _size in shape]
    for key, dim in dims.items():
      if dim is not None:
        edges = np.array(np.ravel(distribution[key]), dtype=float)
        rav_vals = np.array(np.ravel(vals[key]), dtype=float)
        indices[dim] = np.maximum(0, np.minimum(len(edges)-1, \
                                     np.digitize(rav_vals, edges)-1))

    # Iterate through every self.prob value
    rav_prob = np.ravel(rescale(self._prob, self._pscale, 1.))
    rem_prob = np.zeros(distribution.shape, dtype=float)
    for i in range(self.size):
      rem_idx = tuple([idx[i] for idx in indices])
      rem_prob[rem_idx] += rav_prob[i]

    # Replace marginal keys and keep conditional keys
    rem_vals = collections.OrderedDict(distribution)
    rem_dims = collections.OrderedDict(distribution.dims)
    marg_str = []
    for key, val in distribution.items():
      marg_str.append(key)
      if val is not None:
        if np.isscalar(val):
          marg_str[-1] = "{}={}".format(key, val)
        else:
          marg_str[-1] = key + "=[]"
    rem_name = ','.join(marg_str)
    if '|' in self._name:
      rem_name += self._name.split('|')[1]
    cond_keys = list(self._cond.keys())
    if cond_keys:
      for key in cond_keys:
        rem_vals.update({key: self[key]})
        rem_dims.update({key: self._dims[key]})

    return PD(rem_name, rem_vals, dims=rem_dims,
              prob=rescale(rem_prob, 1., self._pscale),
              pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._pscale) else 'p'
    suffix = '' if not self._issingleton else '={}'.format(self.prob)
    return prefix + "(" + self._name + ")" + suffix

#-------------------------------------------------------------------------------
  def serialise(self):
    serialised = super().serialise()
    short_name = self.short_name
    serialised[short_name].update({'prob': self._prob})
    serialised[short_name]['attrs'].update({'pscale': self._pscale})
    return serialised

#-------------------------------------------------------------------------------
