# A distribution is triple comprising variable names, their values (vals), and respective probabilities.

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.dist_ops import prod_dist
from prob.vtypes import isscalar
from prob.ptypes import eval_ptype, rescale, prod_ptype, prod_prob, iscomplex
from prob.manifold import Manifold

#-------------------------------------------------------------------------------
class Dist (Manifold):

  # Public
  prob = None          # Numpy array

  # Protected
  _name = None         # Name of distribution
  _marg_names = None   # List of marginal RV names
  _cond_names = None   # List of conditional RV names 
  _ptype = None        # Same convention as _Prob

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
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = eval_ptype(ptype)
    if self._iscalar:
      if not isscalar(self.prob):
        self.prob = float(self.prob)
    else:
      assert self.ndim == self.prob.ndim, \
        "Dimensionality mismatch between values {} and probabilities {}".\
        format(self.ndim, self.prob.ndim)
    return self._ptype

#-------------------------------------------------------------------------------
  def conditionalise(self, keys):
    # if P(A, key | B), returns P(A | B, key)
    if isinstance(keys, str):
      keys = [keys]
    keys = set(keys)
    for key in keys:
      assert key in self._keys, \
          "Key {} not found among {}".format(key, self._keys)
      assert not self.ret_isscalar(key),\
          format("Conditionalising along scalar value {}".format(key))
    cond_inds = []
    cond_dims = []
    for i, key in enumerate(self._keys):
      if key in keys:
        cond_inds.append(i)
        cond_dims.append(self.dimension[key])




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
    return super().__repr__() + ": " + prefix + "(" + self._name + ") [vals,prob]"

#-------------------------------------------------------------------------------
