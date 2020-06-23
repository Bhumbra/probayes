"""
A stocastic condition is the condition of a conditioned stochastic junction
(called marg here) and conditioning stochastic junction (called cond here).
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

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    arg0, arg1 = None, None
    assert len(args) < 3, "Maximum of two initialisation arguments"
    if len(args) > 1:
      arg0 = args[0]
    if len(args) > 1:
      arg1 = args[1]
    self.add_marg(arg0)
    self.add_cond(arg1)

#-------------------------------------------------------------------------------
  def add_marg(self, *args):
    if self._marg is None: self._marg = SJ()
    self._marg.add_rv(*args)
    self._set_name()

#-------------------------------------------------------------------------------
  def add_cond(self, *args):
    if self._cond is None: self._cond = SJ()
    self._cond.add_rv(*args)
    self._set_name()

#-------------------------------------------------------------------------------
  def _set_name(self):
    if self._marg is None or self._cond is None:
      return
    marg_name, marg_id = self._marg.ret_name(), self._marg.ret_id()
    cond_name, cond_id = self._cond.ret_name(), self._cond.ret_id()
    self._name = '|'.join([marg_name, cond_name])
    self._id = '_with_'.join([marg_id, cond_id])

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def ret_rvs(self):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def get_rvs(self):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_samp(self, samples):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_marg_prod(self, samples):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_prob(self, samples):
    raise NotImplementedError()
   
#-------------------------------------------------------------------------------
  def __call__(self, samples=None, **kwds): 
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def __len__(self):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
