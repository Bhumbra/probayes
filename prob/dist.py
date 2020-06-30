# A distribution is triple comprising variable names, their values (vals), and respective probabilities.

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.vtypes import isscalar
from prob.ptypes import eval_ptype, rescale, prod_ptype, prod_prob, iscomplex

#-------------------------------------------------------------------------------
class Dist:

  # Public
  vals = None          # Dictionary of values
  prob = None          # Numpy array

  # Protected
  _name = None          # Name of distribution
  _marg_names = None   # List of marginal RV names
  _cond_names = None   # List of conditional RV names 
  _ptype = None        # Same convention as _Prob
  _isscalar = None     # if all scalars

#-------------------------------------------------------------------------------
  def __init__(self, name=None, vals=None, prob=None, ptype=None):
    self.set_name(name)
    self.set_vals(vals)
    self.set_prob(prob, ptype)

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    self._name = name

    # Parse name
    marg_str = self._name
    cond_str = ''
    if '|' in marg_str:
      split_str = self._name.split('|')
      assert len(split_str) == 2, "Ambiguous name: {}".format(self._name)
      marg_str, cond_str = split_str
    marg = []
    cond = []
    if len(marg_str):
      marg = marg_str.split(',') if ',' in marg_str else [marg_str]
    if len(cond_str):
      cond = cond_str.split(',') if ',' in cond_str else [cond_str]
    self._marg_names = marg
    self._cond_names = cond

#-------------------------------------------------------------------------------
  def set_vals(self, vals=None):
    self.vals = vals
    self._isscalar = None
    if self.vals is not None:
      assert isinstance(self.vals, dict), \
          "Dist vals must be variable-name keyed dictionary but given: {}". \
          format(self.vals)
      self._isscalar = True
      for val in self.vals.values():
        if not isscalar(val):
          self._isscalar = False
        break
    return self._isscalar

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = eval_ptype(ptype)
    if not isscalar(self.prob) and self._isscalar:
      self.prob = float(self.prob)
    return self._ptype

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self.name

#-------------------------------------------------------------------------------
  def ret_marg_names(self):
    return self._marg_names

#-------------------------------------------------------------------------------
  def ret_cond_names(self):
    return self._cond_names

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    return self._isscalar

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def rescale(self, ptype=None):
    self.set_prob(rescale(self.prob, self._ptype, ptype), ptype)
    return self.prob

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return prod_dist(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if iscomplex(self._ptype) else 'p'
    return super().__repr__() + ": " + prefix + "(" + self._name + ") [vals,prob]"

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def str2key(string):
  if isinstance(string, str):
    k = string.find('=')
    if k > 0:
      return string[:k]
    return string
  return [str2key(element) for element in string]

#-------------------------------------------------------------------------------
def prod_dist(*args, **kwds):
  """ Multiplies two or more distributions subject to the following:
  1. They must not share the same marginal variables. 
  2. Conditional variables must be identical unless contained as marginal from
     another distribution.
  """
  # Check ptypes, scalars, possible fasttrack
  kwds = dict(kwds)
  ptypes = [arg.ret_ptype() for arg in args]
  ptype = kwds.get('ptype', None) or prod_ptype(ptypes)
  arescalars = [arg.ret_isscalar() for arg in args]
  maybe_fasttrack = all(arescalars) and \
                    np.all(ptype == np.array(ptypes)) and \
                    ptype in [0, 1.]
  vals = [arg.vals for arg in args]
  probs = [arg.prob for arg in args]

  # Extract marginal and conditional names
  marg_names = [arg.ret_marg_names() for arg in args]
  cond_names = [arg.ret_cond_names() for arg in args]
  prod_marg = [name for dist_marg_names in marg_names \
                          for name in dist_marg_names]
  assert len(prod_marg) == len(set(prod_marg)), \
      "Marginal RV random variable names not unique across distributions: {}".\
      format(prod_marg)
  prod_marg_name = ','.join(prod_marg)

  # Maybe fast-track identical conditionals
  if maybe_fasttrack:
    if not any(cond_names) or len(set(cond_names)) == 1:
      cond_names = cond_names[0]
      if not cond_names or \
          len(set(cond_names).union(prod_marg)) == len(cond_names) + len(prod_marg):
        prod_cond_name = ','.join(cond_names)
        prod_name = '|'.join([prod_marg_name, prod_cond_name])
        prod_vals = collections.OrdereDict()
        [prod_vals.update(val) for val in vals]
        prob = float(sum(probs)) if iscomplex(ptype) else float(np.prod(probs))
        return dist(prod_name, prod_vals, prob, ptype)

   # Check cond->marg accounts for all differences between conditionals
  flat_cond_names = [name for dist_cond_names in cond_names \
                          for name in dist_cond_names]
  cond2marg = [cond_name for cond_name in flat_cond_names \
                         if cond_name in prod_marg]
  prod_cond = [cond_name for cond_name in flat_cond_names \
                         if cond_name not in cond2marg]
  cond2marg_set = set(cond2marg)

  # Check conditionals compatible
  prod_cond_set = set(prod_cond)
  cond2marg_dict = {name: None for name in prod_cond}
  for i, arg in enumerate(args):
    cond_set = set(cond_names[i]) - cond2marg_set
    assert prod_cond_set == cond_set, \
        "Incompatible conditionals {} vs {}: ".format(prod_cond_set, cond_set)
    for name in cond2marg:
      if name in arg.vals:
        values = arg.vals[name]
        if cond2marg_dict[name] is None:
          cond2marg_dict[name] = values
        elif not np.allclose(cond2marg_dict[name], values):
          raise ValueError("Mismatch in values for condition {}".format(name))

  # Establish product name, values, and dimensions
  prod_keys = str2key(prod_marg + prod_cond)
  prod_nkeys = len(prod_keys)
  prod_arescalars = np.zeros(prod_nkeys, dtype=bool)
  prod_cond_name = ','.join(prod_cond)
  prod_name = '|'.join([prod_marg_name, prod_cond_name])
  prod_vals = collections.OrderedDict()
  for i, key in enumerate(prod_keys):
    values = None
    for val in vals:
      if key in val.keys():
        values = val[key]
        break
    assert values is not None, "Values for key {} not found".format(key)
    prod_arescalars[i] = isscalar(values)
    prod_vals.update({key: values})
  prod_cdims = np.cumsum(np.logical_not(prod_arescalars))
  prod_ndims = prod_cdims[-1]

  # Fast-track scalar products
  if maybe_fasttrack and prod_ndims == 0:
     prob = float(sum(probs)) if iscomplex(ptype) else float(np.prod(probs))
     return dist(prod_name, prod_vals, prob, ptype)

  # Reshape values - they require no axes swapping
  ones_ndims = np.ones(prod_ndims, dtype=int)
  prod_dims = np.ones(prod_ndims, dtype=int)
  scalarset = set()
  dimension = collections.OrderedDict()
  for i, key in enumerate(prod_keys):
    if prod_arescalars[i]:
      scalarset += {key}
    else:
      values = prod_vals[key]
      re_shape = np.copy(ones_ndims)
      dim = prod_cdims[i]-1
      dimension.update({key: dim})
      re_shape[dim] = values.size
      prod_dims[dim] = values.size
      prod_vals.update({key: values.reshape(re_shape)})
  
  # Match probability axes and shapes with axes swapping then reshaping
  prod_probs = [None] * len(args)
  for i, prob in enumerate(probs):
    if not isscalar(prob):
      val_names = str2key(marg_names[i] + cond_names[i])
      nonscalars = [val_name for val_name in val_names \
                             if val_name not in scalarset]
      dims = np.array([dimension[name] for name in nonscalars])
      if dims.size > 1 and np.min(np.diff(dims)) < 0:
        swap = np.argsort(dims)
        probs[i] = np.swapaxes(prob, list(range(dims)), swap)
        dims = dims[swap]
      re_shape = np.copy(ones_ndims)
      for dim in dims:
        re_shape[dim] = prod_dims[dim]
      probs[i] = probs[i].reshape(re_shape)

  # Multiply the probabilities and output the result as a distribution instance
  prob, ptype = prod_prob(*tuple(probs), ptypes=ptypes, ptype=ptype)

  return Dist(prod_name, prod_vals, prob, ptype)

#-------------------------------------------------------------------------------
