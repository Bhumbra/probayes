"""
A random field is a random junction (here called marg) that may be conditioned 
by a another random junction (here called cond) according to a conditional
probability distribution function.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.rj import RJ
from probayes.func import Func
from probayes.dist import Dist
from probayes.dist_utils import product
from probayes.rf_utils import desuffix, get_suffixed

#-------------------------------------------------------------------------------
class RF (RJ):
  # Public
  opqr = None          # (p(pred), p(succ), q(succ|pred), q(pred|succ))

  # Protected
  _marg = None
  _cond = None
  _marg_cond = None    # {'marg': marg, 'cond': cond}
  _def_prop_obj = None
  _prop_obj = None
  _unit_prob = None # Flag for single RV probability
  _unit_tran = None # Flag for single RV transitional

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
    if isinstance(arg, RJ):
      assert not isinstance(arg, RF), "Marginal must be RJ class type"
      self._marg = arg
      self._refresh()
    else:
      self.add_marg(arg)

#-------------------------------------------------------------------------------
  def set_cond(self, arg):
    if isinstance(arg, RJ):
      assert not isinstance(arg, RF), "Conditional must be RJ class type"
      self._cond = arg
      self._refresh()
    else:
      self.add_cond(arg)

#-------------------------------------------------------------------------------
  def add_marg(self, *args):
    if self._marg is None: 
      self._marg = RJ()
    self._marg.add_rv(*args)
    self._refresh()

#-------------------------------------------------------------------------------
  def add_cond(self, *args):
    if self._cond is None: self._cond = RJ()
    self._cond.add_rv(*args)
    self._refresh()

#-------------------------------------------------------------------------------
  def _refresh(self):
    marg_name, cond_name = None, None
    marg_id, cond_id = None, None
    self._rvs = []
    self._keys = []
    if self._marg:
      marg_name = self._marg.ret_name()
      marg_id = self._marg.ret_id()
      marg_rvs = [rv for rv in self._marg.ret_rvs()]
      self._rvs.extend([rv for rv in self._marg.ret_rvs()])
    if self._cond:
      cond_name = self._cond.ret_name()
      cond_id = self._cond.ret_id()
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
    ids = [_id for _id in [marg_id, cond_id] if _id]
    self._id = '_with_'.join(ids)
    self.set_pscale()
    self.eval_length()
    self.opqr = collections.namedtuple(self._id, ['o', 'p', 'q', 'r'])

    # Set the default proposal object and default the delta accordingly
    self._def_prop_obj = self._cond if self._cond is not None else self._marg
    self.delta = self._def_prop_obj.delta
    self._delta_type = self._def_prop_obj._delta_type
    self.set_prop_obj(None)

    # Determine unit RVRF
    self._unit_prob = False
    self._unit_tran = False
    if self._nrvs == 1:
      rv = self._rvs[0]
      self._unit_prob = self._prob is None and rv.ret_prob() is not None
      self._unit_tran = self._tran is None and rv.ret_tran() is not None

#-------------------------------------------------------------------------------
  def set_prop_obj(self, prop_obj=None):
    """ Sets the object used for assigning proposal distributions """
    self._prop_obj = prop_obj
    if self._prop_obj is None:
      return
    self.delta = self._prop_obj.delta
    self._delta_type = self._prop_obj._delta_type

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
  def set_delta(self, delta=None, *args, **kwds):
    if not isinstance(delta, str) and delta not in self._marg_cond.values(): 
      return super().set_delta(delta, *args, **kwds)
    if isinstance(delta, str):
      delta = self._marg_cond[delta]
    self.set_prop_obj(delta)
    self._delta = delta._delta
    self._delta_args = delta._delta_args
    self._delta_kwds = delta._delta_kwds
    self._delta_type = delta._delta_type
    self._spherise = delta._spherise
    return self._delta

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
    self._cfun_lud = cfun._cfun_lud
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
    """ Like RJ.__call__ but optionally takes 'joint' keyword """

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
    prop_obj = self._prop_obj
    if prop_obj is None and (self._tran is not None or self._prop is not None):
      return super().step(*args, **kwds)
    prop_obj = prop_obj or self._def_prop_obj
    return prop_obj.step(*args, **kwds)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    prop_obj = self._prop_obj
    if prop_obj is None and (self._tran is not None or self._prop is not None):
      return super().propose(*args, **kwds)
    prop_obj = prop_obj or self._def_prop_obj
    return prop_obj.propose(*args, **kwds)

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
    """ A function for unconditional and conditional sampling. For conditional
    sampling, use RF.set_delta() to set the delta specification. if neither
    set_prob() nor set_tran() are set, then opqr inputs are disallowed and this
    function outputs a normal __call__(). Otherwise this function returns a 
    namedtuple-generated opqr object that can be accessed using opqr.p or 
    opqr[1] for the probability distribution and opqr.q or opqr[2] for the 
    proposal. Unavailable values are set to None. 
    
    If using set_prop() the output opqr comprises:

    opqr.o: None
    opqr.p: Probability distribution 
    opqr.q: Proposition distribution
    opqr.r: None

    If using set_tran() the output opqr comprises:

    opqr.o: Probability distribution for predecessor
    opqr.p: Probability distribution for successor
    opqr.q: Proposition distribution (successor | predecessor)
    opqr.r: None [for now, reserved for proposition (predecessor | successor)]

    If inputting and opqr object using set_prop(), the values for performing any
    delta operations are taken from the entered proposition distribution. If using
    set_prop(), optional keyword flag suffix=False may be used to remove prime
    notation in keys.

    An optional argument args[1] can included in order to input a dictionary
    of values beyond outside the proposition distribution required to evaluate
    the probability distribution.
    """
    if not args:
      args = {0},
    assert len(args) < 3, "Maximum of two positional arguments"
    if self._tran is None and not self._unit_tran:
      if self._prop is None:
        assert not isinstance(args[0], self.opqr),\
            "Cannot input opqr object with neither set_prob() nor set_tran() set"
        return self.__call__(*args, **kwds)
      return self._sample_prop(*args, **kwds)
    return self._sample_tran(*args, **kwds)

#-------------------------------------------------------------------------------
  def _sample_prop(self, *args, **kwds):

    # Extract suffix status; it is latter popped by propose()
    suffix = "'" if 'suffix' not in kwds else kwds['suffix'] 

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self.opqr):
      prop = self.propose(args[0], **kwds)

    # Otherwise parse:
    else:
      assert args[0].q is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = desuffix(args[0].q.vals)
      prop = self.propose(vals, **kwds)

    # Evaluation of probability
    vals = desuffix(prop.vals)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    call = self.__call__(vals, **kwds)

    return self.opqr(None, call, prop, None)

#-------------------------------------------------------------------------------
  def _sample_tran(self, *args, **kwds):
    assert 'suffix' not in kwds, \
        "Disallowed keyword 'suffix' when using set_tran()"

    # Original probability distribution, proposal, and revp defaults to None
    orig = None
    prop = None
    revp = None

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self.opqr):
      prop = self.step(args[0], **kwds)

    # Otherwise parse successor:
    else:
      dist = args[0].q
      orig = args[0].p
      assert dist is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = get_suffixed(dist.vals)
      prop = self.step(vals, **kwds)

    # Evaluate reverse proposal if transition function not symmetric
    if not self._sym_tran and not self._unit_tran:
      revp = self.reval_tran(prop)

    # Extract values evaluating probability
    vals = get_suffixed(prop.vals)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    prob = self.__call__(vals, **kwds)

    return self.opqr(orig, prob, prop, revp)

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if isinstance(key, str):
      if key not in self._keys:
        return None
      key = self._keys.index(key)
    return self._rvs[key]

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    from probayes.rv import RV
    from probayes.rj import RJ

    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, RF):
      marg = marg + other.ret_marg().ret_rvs()
      cond = cond + other.ret_cond().ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RJ):
      marg = marg + other.ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RV):
      marg = marg + [other]
      return RF(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    from probayes.rv import RV
    from probayes.rj import RJ

    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, RF):
      marg = marg + other.ret_cond().ret_rvs()
      cond = cond + other.ret_marg().ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RJ):
      cond = cond + other.ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RV):
      cond = cond + [self]
      return RF(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
