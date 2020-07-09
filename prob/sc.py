"""
A stocastic condition is a stochastic junction (here called marg) conditioned 
by a another stochastic junction (here called cond) according to a conditional
probability distribution function.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
from prob.sj import SJ
from prob.func import Func

#-------------------------------------------------------------------------------
class SC (SJ):

  # Protected
  _marg = None
  _cond = None
  _prop = None # Proposition function

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_prob()
    assert len(args) < 3, "Maximum of two initialisation arguments"
    arg0 = None if len(args) < 1 else args[0]
    arg1 = None if len(args) < 2 else args[1]
    self.add_marg(arg0)
    self.add_cond(arg1)
    self.set_prop()

#-------------------------------------------------------------------------------
  def add_marg(self, *args):
    if self._marg is None: self._marg = SJ()
    self._marg.add_rv(*args)
    self._refresh()

#-------------------------------------------------------------------------------
  def add_cond(self, *args):
    if self._cond is None: self._cond = SJ()
    self._cond.add_rv(*args)
    self._refresh()

#-------------------------------------------------------------------------------
  def _refresh(self):
    marg_name, cond_name = None, None
    self._rvs = []
    self._keys = []
    if self._marg:
      marg_name = self._marg.ret_name()
      marg_rvs = [rv for rv in self._marg.ret_rvs()]
      self._rvs.extend([rv for rv in self._marg.ret_rvs()])
    if self._cond:
      cond_name = self._cond.ret_name()
      cond_rvs = [rv for rv in self._cond.ret_rvs()]
      self._rvs.extend([rv for rv in self._cond.ret_rvs()])
    if self._marg is None or self._cond is None:
      return
    self._nrvs = len(self._rvs)
    self._keys = [rv.ret_name() for rv in self._rvs]
    self._keyset = set(self._keys)
    self._defiid = self._marg.ret_keyset()
    names = [name for name in [marg_name, cond_name] if name]
    self._name = '|'.join(names)

#-------------------------------------------------------------------------------
  def set_prop(self, prop=None, *args, **kwds):
    self._prop = prop
    if self._prop is None:
      return
    self._prop = Func(self._prop, *args, **kwds)

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values=None):
    keys = self._keys 
    vals = collections.OrderedDict()
    if not isinstance(vals, dict):
      vals.update({key: vals for key in keys})
    else:
      for key, val in values.items():
        if ',' in key:
          subkeys = key.split(',')
          for i, subkey in enumerate(subkeys):
            vals.update({subkey: val[i]})
        else:
          vals.update({key: val})
      for key in self._keys:
        if key not in vals.keys():
          vals.update({key: None})
    marg_vals = collections.OrderedDict()
    for key in self._marg.ret_keys():
      if key in keys:
        marg_vals.update({key: vals[key]})
    cond_vals = collections.OrderedDict()
    for key in self._cond.ret_keys():
      if key in keys:
        cond_vals.update({key: vals[key]})
    marg_dist_name = self._marg.eval_dist_name(marg_vals)
    cond_dist_name = self._cond.eval_dist_name(cond_vals)
    dist_name = marg_dist_name
    if len(cond_dist_name):
      dist_name += "|{}".format(cond_dist_name)
    return dist_name

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_marg_prod(self, samples):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_vals(self, *args, _skip_parsing=False, **kwds):
    assert self._marg, "No marginal stochastic random variables defined"
    return super().eval_vals(*args, _skip_parsing=_skip_parsing, **kwds)

#-------------------------------------------------------------------------------
  def eval_prop(self, values):
    assert self._prop is not None, "Proposal function not set"
    assert isinstance(values, dict), "Input to eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())
    if not self._prop.ret_callable():
      return self._prop()
    return self._prop(values)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    # Similar to __call__ except evaluates prob from proposal if available
    if self._prop is None:
      return self.__call__(*args, **kwds)
    if self._rvs is None:
      return None
    iid = False if 'iid' not in kwds else kwds.pop('iid')
    if type(iid) is bool and iid:
      iid = self._defiid
    values = self._parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values)
    vals, dims = self.eval_vals(values, _skip_parsing=True)
    prop = self.eval_prop(vals)
    if not iid: 
      return Dist(dist_name, vals, dims, prop, self._pscale)
    return Dist(dist_name, vals, dims, prop, self._pscale).prod(iid)

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._rvs[key]

#-------------------------------------------------------------------------------
