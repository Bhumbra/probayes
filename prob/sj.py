"""
A stochastic junction comprises a collection of a random variables that 
participate in a joint probability distribution function.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
from prob.prob import log_prob, exp_logs
from prob.rv import RV, io_use_vfun
from prob.dist import Dist, marg_prod

#-------------------------------------------------------------------------------
class SJ:

  # Protected
  _name = None    # Cannot be set externally
  _rvs = None     # Dict of random variables
  _nrvs = None
  _keys = None
  _keyset = None
  _ptype = None
  _use_vfun = None

  # Private
  __callable = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)
    self.set_prob()
    self.set_use_vfun()

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    if len(args) == 1 and isinstance(args[0], (SJ, dict, set, tuple, list)):
      args = args[0]
    else:
      args = tuple(args)
    self.add_rv(args)
    return self.ret_rvs()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    if self._rvs is None:
      self._rvs = collections.OrderedDict()
    if isinstance(rv, (SJ, dict, set, tuple, list)):
      rvs = rv
      if isinstance(rvs, SJ):
        rvs = rvs.ret_rvs()
      if isinstance(rvs, dict):
        rvs = rvs.values()
      [self.add_rv(rv) for rv in rvs]
    else:
      rv_name = rv.ret_name()
      assert isinstance(rv, RV), \
          "Input not a RV instance but of type: {}".format(type(rv))
      assert rv_name not in self._rvs.keys(), \
          "Existing RV name {} already present in collection".format(rv_name)
      self._rvs.update({rv_name: rv})
    self._nrvs = len(self._rvs)
    self._keys = list(self._rvs.keys())
    self._keyset = set(self._keys)
    self._name = ','.join(self._keys)
    return self._nrvs
  
#-------------------------------------------------------------------------------
  def ret_rvs(self, aslist=True):
    # Defaulting aslist=True plays more nicely with inheriting classes
    rvs = self._rvs
    if aslist:
      if isinstance(rvs, dict):
        rvs = list(rvs.values())
      assert isinstance(rvs, list), "RVs not a recognised variable type: {}".\
                                    format(type(rvs))
    return rvs

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def ret_nrvs(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def set_ptype(self, ptype=None):
    self._ptype = ptype
    if self._ptype is not None:
      if self._ptype in ['log', 'ln']:
        self._ptype = str(float(0))
      return self._ptype

    rvs = self.ret_rvs(aslist=True)
    all_none = all([rv.ret_ptype() is None for rv in rvs])
    if all_none:
      return self._ptype
    use_logs = any([isinstance(rv.ret_ptype(), str) for rv in rvs])
    ptype = 0. if use_logs else 1.
    for rv in rvs:
      rv_ptype = rv.ret_ptype()
      if rv_ptype is None:
        rv_ptype = 1.
      rv_log = isinstance(rv_ptype, str)
      if use_logs:
        rtype = float(rv_ptype) if rv_log else np.log(rv_ptype)
        ptype += rtype
      else:
        rtype = np.exp(float(rv_ptype)) if rv_log else rv_ptype
        ptype *= ptype
    self._ptype = str(ptype) if use_logs else ptype
    return self._ptype

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def eval_vals(self, values, min_rdim=0):
    if isinstance(values, dict):
      no_check = True
      for val in values.values(): # bypass checks if possible
        if val is None or type(val) is int:
          no_check = False
          break
        elif type(val) is not float and isinstance(val, np.ndarray):
          if val.size != 1:
            no_check = False
            break
      if no_check:
        return values
    else:
      values = {key: values for key in self._keys}
    rvs = self.ret_rvs(aslist=True)
    nrvs_1s = np.ones(self._nrvs, dtype=int)
    for i, rv in enumerate(rvs):
      rv_name = rv.ret_name()
      vals = values[rv_name]
      re_shape = False
      if vals is None or type(vals) is int:
        vals = rv.eval_vals(vals)
        re_shape = True
      elif isinstance(vals, np.ndarray):
        if vals.size != 1:
          re_shape = vals.ndim != self._nrvs - min_rdim
      if re_shape:
        re_shape = np.copy(nrvs_1s[min_rdim:])
        re_dim = max(0, i - min_rdim)
        re_shape[re_dim] = vals.size
        vals = vals.reshape(re_shape)
      values[rv_name] = vals
    return values

#-------------------------------------------------------------------------------
  def eval_prob(self, values):
    assert isinstance(values, dict), "Input to eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())
    if self._prob is None:
      dists = tuple([rv(values[rv.ret_name()]) for rv in self.ret_rvs(aslist=True)])
      return marg_prod(*dists, check=False).prob
    if self.__callable:
      probs = probs(values, *self._prob_args, **self._prob_kwds)
    else:
      probs = np.atleast_1d(probs).astype(float)
    if probs.ndim != self._nrvs:
      warnings.warn(
          "Evaluated probability dimensionality {}".format(probs.ndim) + \
          "incommensurate with number of RVs {}".format(self._nrvs)
      )
    return probs

#-------------------------------------------------------------------------------
  def set_use_vfun(self, use_vfun=True):
    self._use_vfun = io_use_vfun(use_vfun)
    return self._use_vfun

#-------------------------------------------------------------------------------
  def ret_use_vfun(self):
    return self._use_vfun

#-------------------------------------------------------------------------------
  def vfun_0(self, values, use_vfun=True):
    if not use_vfun:
      return values
    rvs = self.ret_rvs(aslist=True)
    for key, rv in zip(self._self._keys, rvs):
      if key in values:
        values[key] = rv.vfun_0(values[key], use_vfun)
    return values

#-------------------------------------------------------------------------------
  def vfun_1(self, values, use_vfun=True):
    if not use_vfun:
      return values
    rvs = self.ret_rvs(aslist=True)
    for key, rv in zip(self._keys, rvs):
      if key in values:
        values[key] = rv.vfun_1(values[key], use_vfun)
    return values

#-------------------------------------------------------------------------------
  def __call__(self, values=None, **kwds):  # Let's make this args ands kwds
    ''' 
    Returns a namedtuple of the rvs.
    '''
    if self._rvs is None:
      return None
    if values is None and len(kwds):
      values = dict(kwds)
    elif not isinstance(values, dict):
      values = {key: values for key in self._keys}
    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    vals = self.vfun_1(vals, self._use_vfun[1])
    return Dist(self._name, vals, prob, self._ptype)

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = self._keys[key]
    if isinstance(key, str):
      return self._rvs[key]
    raise TypeError("Unexpected key type: {}".format(key))

#-------------------------------------------------------------------------------
