# A distribution is triple comprising variable names, their values (vals), and respective probabilities.

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.ptypes import eval_ptype, rescale, prod_prob


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
        if not np.isscalar(val):
          self._isscalar = False
        break
    return self._isscalar

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = eval_ptype(ptype)
    if not np.isscalar(self.prob) and self._isscalar:
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
    return dist_prod(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if np.iscomplex(self._ptype) else 'p'
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
def marg_prod(*args, **kwds):
  """ 
  Returns the marginal product of single-variable distributions with identical 
  ptypes. Assert checks cans be bypassed if set to zero
  """
  kwds = dict(kwds)
  check = kwds.get('check', True)
  ptype = kwds.get('ptype', None)
  ptypes = [arg.ret_ptype() for arg in args]
  if ptype is None:
    ptype = args[0].ret_ptype()
  track_ptype = True
  if ptype == 0.j:
    track_ptype = False
    for _ptype in ptypes[1:]:
      if _ptype != 0.j:
        track_ptype = True
        break
  elif ptype == 1.:
    track_ptype = False
    for _ptype in ptypes[1:]:
      if _ptype != 1:
        track_ptype = True
        break
  cond_names = [arg.ret_cond_names() for arg in args]
  if check:
    if any(cond_names):
      assert len(cond_names) == len(set(cond_names)),\
        "Not all conditional variables are identical"
  cond_names = cond_names[0]
  marg_names = []
  for arg in args:
    marg_names.extend(arg.ret_marg_names())
  if check:
    assert len(marg_names) == len(set(marg_names)),\
      "Non-unique marginal variable name found in {}".format(marg_names)
    if len(cond_names):
       for marg_name in marg_names:
         assert marg_name not in cond_names,\
           "Overlap between variable {} found within condition {}".format(
               marg_name, cond_names)
    assert_msg = "Variable ptypes incompatible with ptype {}".format(ptype)
    if np.iscomplex(ptype):
      assert all([np.iscomplex(_ptype) for _ptype in ptypes]), assert_msg
    else:
      assert all([not np.iscomplex(_ptype) for _ptype in ptypes]), assert_msg
  prod_vals = collections.OrderedDict()
  [prod_vals.update(arg.vals) for arg in args]
  prod_marg_name = ','.join(marg_names)
  prod_cond_name = ','.join(cond_names)
  prod_name = '|'.join([prod_marg_name, prod_cond_name])
  prod_keys = str2key(marg_names) + str2key(cond_names)
  arg_isscalars = [arg.ret_isscalar() for arg in args]

  # Bypass comprehensive approach for scalars
  if not check and not track_ptype and all(arg_isscalars):
    probs = [arg.prob for arg in args]
    prob = float(sum(probs)) if np.iscomplex(ptype) \
           else float(np.prod(probs))
    return Dist(prod_name, prod_vals, prob, ptype)

  # Reshape values and marginal probabilities
  reshape_refs = np.cumsum(np.logical_not(arg_isscalars))
  maybe_reshape = np.max(reshape_refs) > 0
  nres_ones = None if not maybe_reshape else np.ones(reshape_refs[-1], dtype=int)
  probs = [None] * len(args)
  for i, arg in enumerate(args):
    vals, prob, marg_names, ptype = arg.vals, arg.prob, marg_names[i], ptypes[i]
    if maybe_reshape and not(arg_isscalars[i]):
      reshape_prob = np.copy(nres_ones)
      for key in vals.keys():
        reshape_vals = np.copy(nres_ones)
        index = prod_keys.index(key)
        reshape_vals[reshape_refs[index]-1] = vals[key].size
        reshape_prob[reshape_refs[index]-1] = vals[key].size
        prod_vals.update({key: vals[key].reshape(reshape_vals)})
      if isinstance(prob, np.ndarray):
        prob = prob.reshape(reshape_prob)
    probs[i] = prob

  # Multiply the probabilities and output the result as a distribution
  prob, ptype = prod_prob(*tuple(probs), ptypes=ptypes)
  return Dist(prod_name, prod_vals, prob, ptype)
 
#-------------------------------------------------------------------------------
def dist_prod(*args, **kwds):
  """ Multiplies two or more distributions subject to the following:
  They must not share the same marginal variables. If ptype is
  specified as a keyword, the resulting product distribution will
  conform to that ptype.
  """
  return marg_prod(*args, **kwds)

#-------------------------------------------------------------------------------
