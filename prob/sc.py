"""
A stocastic condition is a stochastic junction (here called marg) conditioned 
by a another stochastic junction (here called cond) according to a conditional
probability distribution function.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
from prob.sj import SJ

#-------------------------------------------------------------------------------
class SC (SJ):

  # Protected
  _marg = None
  _cond = None
  _iid = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    assert len(args) < 3, "Maximum of two initialisation arguments"
    arg0 = None if len(args) < 1 else args[0]
    arg1 = None if len(args) < 2 else args[1]
    self.add_marg(arg0)
    self.add_cond(arg1)
    self.set_iid()

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
  def set_iid(self, iid=False):
    self._iid = iid

#-------------------------------------------------------------------------------
  def _refresh(self):
    marg_name, cond_name = None, None
    self._rvs = []
    self._keys = []
    if self._marg:
      marg_name, marg_id = self._marg.ret_name(), self._marg.ret_id()
      marg_rvs = [rv for rv in self._marg.ret_rvs()]
      self._rvs.extend([rv for rv in self._marg.ret_rvs()])
    if self._cond:
      cond_name, cond_id = self._cond.ret_name(), self._cond.ret_id()
      cond_rvs = [rv for rv in self._cond.ret_rvs()]
      self._rvs.extend([rv for rv in self._cond.ret_rvs()])
    if self._marg is None or self._cond is None:
      return
    self._nrvs = len(self._rvs)
    self._keys = [rv.name for rv in self._rvs]
    self._keyset = set(self._keys)
    names = [name for name in [marg_name, cond_name] if name]
    rvids = [rvid for rvid in [marg_rvid, cond_rvid] if rvid]
    self._name = '|'.join(names)

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
  def eval_vals(self, values):
    assert self._marg, "No marginal stochastic random variables defined"
    return super().eval_values(values, self._marg.ret_nrvs()-1)

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._rvs[key]

#-------------------------------------------------------------------------------
