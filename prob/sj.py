"""
A stochastic junction comprises a collection of a random variables that 
participate in a joint probability distribution function.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
from prob.prob import log_prob, exp_logs
from prob.rv import RV

#-------------------------------------------------------------------------------
class SJ:

  # Protected
  _name = None
  _id = None
  _get = None
  _rvs = None
  _nrvs = None
  _keys = None
  _keyset = None

  # Private
  __nrvs_1s = None
  __callable = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)
    self.set_prob()

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
      assert isinstance(rv, RV), \
          "Input not a RV instance but of type: {}".format(type(rv))
      assert rv.name not in self._rvs.keys(), \
          "Existing RV name {} already present in collection".format(rv.name)
      self._rvs.update({rv.name: rv})
    self._nrvs = len(self._rvs)
    self.__nrvs_1s = np.ones(self._nrvs, dtype=int)
    self._keys = list(self._rvs.keys())
    self._keyset = set(self._keys)
    self._name = ','.join(self._keys)
    self._id = '_and_'.join(self._keys)
    return self._nrvs

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    self._prob = prob
    self._prob_args = tuple(args)
    self._prob_kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def ret_rvs(self):
    return self._rvs

#-------------------------------------------------------------------------------
  def ret_nrvs(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def ret_id(self):
    return self._id

#-------------------------------------------------------------------------------
  def get_rvs(self):
    if self._get is None:
      self._get = collections.namedtuple(self._id, self._keys, **kwds)
    rvs = self.ret_rvs()
    rvs = list(rvs.values()) if isinstance(rvs, dict) else list(rvs)
    return self._get(*tuple(self.ret_rvs().values()))

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

    rvs = self.ret_rvs()
    # This next line is there just to play nice with inheriting classes
    rvs = list(rvs.values()) if isinstance(rvs, dict) else list(rvs)
    for i, rv in enumerate(rvs):
      vals = values[rv.name]
      re_shape = False
      if vals is None or type(vals) is int:
        vals = rv.eval_vals(vals)
        re_shape = True
      elif isinstance(vals, np.ndarray):
        if vals.size != 1:
          re_shape = vals.ndim != self._nrvs - min_rdim
      if re_shape:
        re_shape = np.copy(self.__nrvs_1s[min_rdim:])
        re_dim = max(0, i - min_rdim)
        re_shape[re_dim] = vals.size
        vals = vals.reshape(re_shape)
      values[rv.name] = vals
    return values

#-------------------------------------------------------------------------------
  def eval_marg_prod(self, values):
    """ Evaluates the marginal product """
    assert isinstance(values, dict), "SJ.eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())
    probs = [None] * self._nrvs
    use_logs = any([isinstance(rv.ret_ptype(), str) for rv in self._rvs.values()])
    run_ptype = 0. if use_logs else 1.
    for i, rv in enumerate(self._rvs.values()):
      prob = rv.eval_prob(values[rv.name])
      re_shape = np.copy(self.__nrvs_1s)
      re_shape[i] = prob.size
      prob = prob.reshape(re_shape)
      ptype = rv.ret_ptype()
      if not use_logs:
        if ptype is not None and ptype != 1.:
          run_ptype *= ptype
        probs[i] = np.copy(prob)
      else:
        logprob = ptype is None
        if isinstance(ptype, str):
          ptype = float(ptype)
          logprob = False
        elif type(ptype) is float:
          ptype = np.log(ptype)
          logprob = True
        run_ptype += ptype
        probs[i] = log_prob(prob) if logprob else np.copy(prob)
    prob = None
    for i in range(self._nrvs):
      if prob is None:
        prob = probs[i]
      elif use_logs:
        prob = prob + probs[i]
      else:
        prob = prob * probs[i]
    if use_logs:
      if run_ptype != 0.:
        prob = prob - run_ptype
      probs = exp_logs(prob)
    else:
      if run_ptype != 1. and run_ptype != 0.:
        prob = prob / run_ptype
      probs = prob
    return probs

#-------------------------------------------------------------------------------
  def eval_prob(self, values):
    assert isinstance(values, dict), "Input to eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())
    if self._prob is None:
      return self.eval_marg_prod(values)
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
  def __call__(self, values=None, **kwds): 
    ''' 
    Returns a namedtuple of the rvs.
    '''
    if self._rvs is None:
      return None
    if self._get is None or len(kwds):
      self._get = collections.namedtuple(self._id, ['vals', 'prob'], **kwds)

    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    return self._get(vals, prob)

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
