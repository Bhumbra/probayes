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
from prob.dist import Dist
from prob.dist_ops import product

#-------------------------------------------------------------------------------
class SC (SJ):

  # Protected
  _marg = None
  _cond = None

  # Private
  __sym_tran = None

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    self.set_prob()
    assert len(args) < 3, "Maximum of two initialisation arguments"
    arg0 = None if len(args) < 1 else args[0]
    arg1 = None if len(args) < 2 else args[1]
    if arg0 is not None: self.add_marg(arg0)
    if arg1 is not None: self.add_cond(arg1)

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
    if self._marg is None and self._cond is None:
      return
    self._nrvs = len(self._rvs)
    self._keys = [rv.ret_name() for rv in self._rvs]
    self._keyset = set(self._keys)
    self._defiid = self._marg.ret_keyset()
    names = [name for name in [marg_name, cond_name] if name]
    self._name = '|'.join(names)
    self.eval_length()
    tran = 'cond' if self._cond else 'marg'
    self.set_tran(tran)
    tran_obj = self._marg if tran=='marg' else self._cond
    self.delta = tran_obj.delta
    self.Delta = tran_obj.Delta

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    self._tran = tran
    self.__sym_tran = False
    if isinstance(self._tran, str):
      assert self._tran in ['marg', 'cond'],\
          "If string, tran must be 'marg' or 'cond', not {}".format(self._tran)
    else:
      return super().set_tran(tran, *args, **kwds)


#-------------------------------------------------------------------------------
  def eval_dist_name(self, values, suffix=None):
    if suffix is not None:
      return super().eval_dist_name(values, suffix)
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
    if self._marg:
      for key in self._marg.ret_keys():
        if key in keys:
          marg_vals.update({key: vals[key]})
    cond_vals = collections.OrderedDict()
    if self._cond:
      for key in self._cond.ret_keys():
        if key in keys:
          cond_vals.update({key: vals[key]})
    marg_dist_name = self._marg.eval_dist_name(marg_vals)
    cond_dist_name = '' if not self._cond else \
                     self._cond.eval_dist_name(cond_vals)
    dist_name = marg_dist_name
    if len(cond_dist_name):
      dist_name += "|{}".format(cond_dist_name)
    return dist_name

#-------------------------------------------------------------------------------
  def ret_marg(self):
    return self._marg

#-------------------------------------------------------------------------------
  def ret_cond(self):
    return self._cond

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
  def __call__(self, *args, **kwds):
    """  Returns p[args] join distribution instance. 
    Optionally takes 'joint' keyword 
    """
    if self._rvs is None:
      return None
    joint = False if 'joint' not in kwds else kwds.pop('joint')
    dist = super().__call__(*args, **kwds)
    if not joint:
      return dist
    vals = dist.ret_cond_vals()
    cond_dist = self._cond(vals)
    return product(cond_dist, dist)

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    obj = None
    if self._tran == 'marg': obj = self._marg
    if self._tran == 'cond': obj = self._cond
    if obj is None:
      return super().step(*args, **kwds)
    return obj.step(*args, **kwds)

#-------------------------------------------------------------------------------
  def parse_pred_args(self, arg):
    obj = None
    if self._tran == 'marg': obj = self._marg
    if self._tran == 'cond': obj = self._cond
    if obj is None:
      return self.parse_args(args)
    if not isinstance(arg, dict):
      return obj.parse_args(args)
    keyset = obj.ret_keyset()
    pred = collections.OrderedDict({key: val for key, val in arg.items() 
                                             if key in keyset})
    return obj.parse_args(pred)

#-------------------------------------------------------------------------------
  def sample(self, *args, **kwds):
    """ If len(args) == 1, returns just a call.
    If len(args) == 2, returns a two-distribution tuple (step, call) based on 
    the step. This function is implemented here and not for SJ to ensure
    support for the keyword 'joint' which is to be encouraged.
    """
    assert 0 < len(args) < 3, "Accepts either one or two arguments"

    # Single call is the trivial case
    if len(args) == 1:
      return self.__call__(*args, **kwds)

    # Step
    pred = self.parse_pred_args(args[0])
    cond = self.step(pred, args[1], **kwds)

    # Call
    values = self.parse_args(args[0])
    for key, val in cond.vals.items():
      if key[-1] == "'":
        values.update({key[:-1]: val})
    return cond, self.__call__(values, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._rvs[key]

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    from prob.rv import RV
    from prob.sj import SJ
    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, SC):
      marg = marg + other.ret_marg().ret_rvs()
      cond = cond + other.ret_cond().ret_rvs()
      return SC(marg, cond)

    if isinstance(other, SJ):
      marg = marg + other.ret_rvs()
      return SC(marg, cond)

    if isinstance(other, RV):
      marg = marg + [other]
      return SC(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    from prob.rv import RV
    from prob.sj import SJ
    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, SC):
      marg = marg + other.ret_cond().ret_rvs()
      cond = cond + other.ret_marg().ret_rvs()
      return SC(marg, cond)

    if isinstance(other, SJ):
      cond = cond + other.ret_rvs()
      return SC(marg, cond)

    if isinstance(other, RV):
      cond = cond + [self]
      return SC(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
