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
  _marg_cond = None    # {'marg': marg, 'cond': cond}
  _prop_obj = None

  # Private

  __sym_tran = None

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    self.set_prob()
    assert len(args) < 3, "Maximum of two initialisation arguments"
    arg0 = None if len(args) < 1 else args[0]
    arg1 = None if len(args) < 2 else args[1]
    if arg0 is not None: self.set_marg(arg0)
    if arg1 is not None: self.set_cond(arg1)

#-------------------------------------------------------------------------------
  def set_marg(self, arg):
    if isinstance(arg, SJ):
      assert not isinstance(arg, SC), "Marginal must be SJ class type"
      self._marg = arg
      self._refresh()
    else:
      self.add_marg(arg)

#-------------------------------------------------------------------------------
  def set_cond(self, arg):
    if isinstance(arg, SJ):
      assert not isinstance(arg, SC), "Conditional must be SJ class type"
      self._cond = arg
      self._refresh()
    else:
      self.add_cond(arg)

#-------------------------------------------------------------------------------
  def add_marg(self, *args):
    if self._marg is None: 
      self._marg = SJ()
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
    self._marg_cond = {'marg': self._marg, 'cond': self._cond}
    self._nrvs = len(self._rvs)
    self._keys = [rv.ret_name() for rv in self._rvs]
    self._keyset = set(self._keys)
    self._defiid = self._marg.ret_keyset()
    names = [name for name in [marg_name, cond_name] if name]
    self._name = '|'.join(names)
    self.eval_length()
    prop_obj = self._cond if self._cond is not None else self._marg
    self.set_prop_obj(prop_obj)

#-------------------------------------------------------------------------------
  def set_prop_obj(self, prop_obj=None):
    """ Sets the object used for assigning proposal distributions """
    self._prop_obj = prop_obj
    if self._prop_obj is None:
      return
    self.delta = self._prop_obj.delta
    self.Delta = self._prop_obj.Delta

#-------------------------------------------------------------------------------
  def set_prop(self, prop=None, *args, **kwds):
    if not isinstance(prop, str) and prop not in self._marg_cond.values():
      return super().set_prop(prop, *args, **kwds)
    if isinstance(prop, str):
      prop = self._marg_cond[prop]
    self.set_prop_obj(prop)
    self._prop = prop._prop
    return self._prop

#-------------------------------------------------------------------------------
  def set_step(self, step=None, *args, **kwds):
    if not isinstance(step, str) and step not in self._marg_cond.values(): 
      return super().set_step(step, *args, **kwds)
    if isinstance(step, str):
      step = self._marg_cond[tran]
    self.set_prop_obj(step)
    self._step = step._step
    return self._step

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    if not isinstance(tran, str) and tran not in self._marg_cond.values(): 
      return super().set_tran(tran, *args, **kwds)
    if isinstance(tran, str):
      tran = self._marg_cond[tran]
    self.set_prop_obj(tran)
    self._tran = tran._tran
    return self._tran

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    if not isinstance(tfun, str) and tfun not in self._marg_cond.values(): 
      return super().set_tfun(tfun, *args, **kwds)
    if isinstance(tfun, str):
      tfun = self._marg_cond[tfun]
    self.set_prop_obj(tfun)
    self._tfun = tfun._tfun
    return self._tfun

#-------------------------------------------------------------------------------
  def set_cfun(self, cfun=None, *args, **kwds):
    if not isinstance(cfun, str) and cfun not in self._marg_cond.values(): 
      return super().set_cfun(cfun, *args, **kwds)
    if isinstance(cfun, str):
      cfun = self._marg_cond[cfun]
    self.set_prop_obj(cfun)
    self._cfun = cfun._cfun
    return self._cfun

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
    """ Like SJ.__call__ but optionally takes 'joint' keyword """

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
    if self._prop_obj is None:
      return super().step(*args, **kwds)
    return self._prop_obj.step(*args, **kwds)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    if self._prop_obj is None:
      return super().propose(*args, **kwds)
    return self._prop_obj.propose(*args, **kwds)

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
