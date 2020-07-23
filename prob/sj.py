"""
A stochastic junction comprises a collection of a random variables that 
participate in a joint probability distribution function.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
from prob.rv import RV
from prob.dist import Dist
from prob.vtypes import isscalar, isunitsetint, issingleton
from prob.pscales import iscomplex, real_sqrt, prod_rule, \
                         rescale, eval_pscale, prod_pscale
from prob.func import Func

#-------------------------------------------------------------------------------
def rv_prod_rule(*args, rvs, pscale=None):
  """ Returns the probability product treating all rvs as independent.
  Values (=args[0]) are keyed by RV name and rvs are a list of RVs.
  """
  values = args[0]
  pscales = [rv.ret_pscale() for rv in rvs]
  pscale = pscale or prod_pscale(pscales)
  use_logs = iscomplex(pscale)
  probs = [rv.eval_prob(values[rv.ret_name()]) for rv in rvs]
  prob, pscale = prod_rule(*tuple(probs),
                           pscales=pscales,
                           pscale=pscale)

  # This section below is there just to play nicely with conditionals
  if len(args) > 1:
    if use_logs:
      prob = rescale(prob, pscale, 0.j)
    else:
      prob = rescale(prob, pscale, 1.)
    for arg in args[1:]:
      if use_logs:
        offs, _ = rv_prod_rule(arg, rvs=rvs, pscale=0.j)
        prob = prob + offs
      else:
        coef, _ = rv_prod_rule(arg, rvs=rvs, pscale=1.)
        prob = prob * coef
    if use_logs:
      prob = prob / float(len(args))
      prob = rescale(prob, 0.j, pscale)
    else:
      prob = prob ** (1. / float(len(args)))
      prob = rescale(prob, 1., pscale)
  return prob, pscale

#-------------------------------------------------------------------------------
class SJ:
  # Public
  Delta = None
  delta = None

  # Protected
  _name = None      # Cannot be set externally
  _rvs = None       # Dict of random variables
  _nrvs = None
  _keys = None
  _keyset = None
  _defiid = None
  _pscale = None
  _prob = None
  _pscale = None
  _tran = None
  _tfun = None
  _cfun = None
  _length = None
  _lengths = None
  _cfun = None

  # Private
  __isscalar = None
  __callable = None
  __sym_tran = None
  __cfun_lud = None

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
    assert self._prob is None, \
      "Cannot assign new randon variables after specifying joint/condition prob"
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
      key = rv.ret_name()
      assert key not in self._rvs.keys(), \
          "Existing RV name {} already present in collection".format(rv_name)
      self._rvs.update({key: rv})
    self._nrvs = len(self._rvs)
    self._keys = list(self._rvs.keys())
    self._keyset = set(self._keys)
    self._defiid = self._keyset
    self._name = ','.join(self._keys)
    self._id = '_and_'.join(self._keys)
    if self._id:
      self.Delta = collections.namedtuple(self._id, 'ð')
      self.delta = collections.namedtuple('ð', self._keys)
    self.set_pscale()
    self.eval_length()
    return self._nrvs

#-------------------------------------------------------------------------------
  def set_cfun(self, cfun=None, *args, **kwds):
    self._cfun = cfun
    self.__cfun_lud = None
    if self._cfun is None:
      return
    self._cfun = Func(self._cfun, *args, **kwds)
    if not self._cfun.ret_callable():
      message = "Non callable cfun objects must be a square 2D Numpy array " + \
                "of size corresponding to number of variables {}".format(self._nrvs)
      assert isinstance(cfun, np.ndarray), message
      assert cfun.ndim == 2, message
      assert np.all(np.array(cfun.shape) == self._nrvs), message
      self.__cfun_lud = np.linalg.cholesky(cfun)

#-------------------------------------------------------------------------------
  def eval_length(self):
    rvs = self.ret_rvs(aslist=True)
    self._lengths = np.array([rv.ret_length() for rv in rvs], dtype=float)
    self._length = np.sqrt(np.sum(self._lengths))
    return self._length

#-------------------------------------------------------------------------------
  def ret_length(self):
    return self._length

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
  def ret_keys(self):
    return self._keys

#-------------------------------------------------------------------------------
  def ret_keyset(self):
    return self._keyset

#-------------------------------------------------------------------------------
  def set_pscale(self, pscale=None):
    if pscale is not None or not self._nrvs:
      self._pscale = eval_pscale(pscale)
      return self._pscale
    rvs = self.ret_rvs(aslist=True)
    pscales = [rv.ret_pscale() for rv in rvs]
    self._pscale = prod_pscale(pscales)
    return self._pscale

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    return self._pscale

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    kwds = dict(kwds)
    if 'pscale' in kwds:
      pscale = kwds.pop('pscale')
      self.set_pscale(pscale)
    self.__callable = None
    self.__isscalar = None
    self._prob = prob
    if self._prob is None:
      return self.__callable
    self._prob = Func(self._prob, *args, **kwds)
    self.__callable = self._prob.ret_callable()
    self.__isscalar = self._prob.ret_isscalar()
    return self.__callable

#-------------------------------------------------------------------------------
  def parse_args(self, *args, **kwds):
    """ Returns (values, iid) from *args and **kwds """
    args = tuple(args)
    kwds = dict(kwds)
    if not args and not kwds:
      args = (None,)
    if args:
      assert len(args) == 1 and not kwds, \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
      kwds = dict(args[0]) if isinstance(args[0], dict) else \
             ({key: args[0] for key in self._keys})

    elif kwds:
      assert not args, \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
    values = dict(kwds)
    seen_keys = []
    for key, val in values.items():
      count_comma = key.count(',')
      if count_comma:
        seen_keys.extend(key.split(','))
        if isinstance(val, (tuple, list)):
          assert len(val) == count_comma+1, \
              "Mismatch in key specification {} and number of values {}".\
              format(key, len(val))
        else:
          values.update({key: [val] * (count_comma+1)})
      else:
        seen_keys.append(key)
      assert seen_keys[-1] in self._keys, \
          "Unrecognised key {} among available RVs {}".format(
              seen_keys[-1], self._keys)
    for key in self._keys:
      if key not in seen_keys:
        values.update({key: None})

    return values

#-------------------------------------------------------------------------------
  def eval_vals(self, *args, _skip_parsing=False, min_dim=0, **kwds):
    """ 
    Keep args and kwds since could be called externally. This ignores self._prob.
    """
    values = self.parse_args(*args, **kwds) if not _skip_parsing else args[0]
    dims = {}
    
    # Don't reshape if all scalars (and therefore by definition no shared keys)
    if all([np.isscalar(value) for value in values.values()]): # use np.scalar
      return values, dims

    # Create reference mapping for shared keys across rvs
    values_ref = collections.OrderedDict({key: [key, None] for key in self._keys})
    for key in values.keys():
      if ',' in key:
        subkeys = key.split(',')
        for i, subkey in enumerate(subkeys):
          values_ref[subkey] = [key, i]

    # Share dimensions for joint variables and do not dimension scalars
    ndim = min_dim
    dims = collections.OrderedDict({key: None for key in self._keys})
    seen_keys = set()
    for i, key in enumerate(self._keys):
      new_dim = False
      if values_ref[key][1] is None: # i.e. not shared
        if not np.isscalar(values[key]): # use np.scalar here (to exclude unitsetint)
          dims[key] = ndim
          new_dim = True
        seen_keys.add(key)
      elif key not in seen_keys:
        val_ref = values_ref[key]
        subkeys = val_ref[0].split(',')
        for subkey in subkeys:
          dims[subkey] = ndim
          seen_keys.add(subkey)
        if not np.isscalar(values[val_ref[0]][val_ref[1]]): # and here
          new_dim = True
      if new_dim:
        ndim += 1

    # Reshape
    ndims = max([dim for dim in dims.values() if dim is not None]) + 1 or 0
    ones_ndims = np.ones(ndims, dtype=int)
    vals = collections.OrderedDict()
    rvs = self.ret_rvs(aslist=True)
    for i, rv in enumerate(rvs):
      key = rv.ret_name()
      reshape = True
      if key in values.keys():
        vals.update({key: values[key]})
        reshape = not np.isscalar(vals[key])
        if vals[key] is None or isinstance(vals[key], set):
          vals[key] = rv.eval_vals(vals[key])
      else:
        val_ref = values_ref[key]
        vals_val = values[val_ref[0]][val_ref[1]]
        if vals_val is None or isinstance(vals_val, set):
          vals_val = rv.eval_vals(vals_val)
        vals.update({key: vals_val})
      if reshape and not isscalar(vals[key]):
        re_shape = np.copy(ones_ndims)
        re_dim = dims[key]
        re_shape[re_dim] = vals[key].size
        vals[key] = vals[key].reshape(re_shape)
    
    # Remove dimensionality for singletons
    for key in self._keys:
      if issingleton(vals[key]):
        dims[key] = None
    return vals, dims

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    self._tran = tran
    self.__sym_tran = False
    if self._tran is None:
      return
    self._tran = Func(self._tran, *args, **kwds)
    self.__sym_tran = self._tran.ret_istuple()

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    # Provide cdf and inverse cdf for conditional sampling
    self._tfun = tfun if tfun is None else Func(tfun, *args, **kwds)
    if self._tfun is None:
      return
    raise NotImplemented(
        "Multidimensional transitional CDF sampling not yet implemented")
    assert self._tfun.ret_istuple(), "Tuple of two functions required"
    assert len(self._tfun) == 2, "Tuple of two functions required."

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    if values is None:
      values = {}
    assert isinstance(values, dict), "Input to eval_prob() requires values dict"
    assert set(values.keys()) == self._keyset, \
      "Sample dictionary keys {} mismatch with RV names {}".format(
        values.keys(), self._keys())

    # If not specified, treat as independent variables
    if self._prob is None:
      prob, pscale = rv_prod_rule(values, 
                                  rvs=self.ret_rvs(aslist=True),
                                  pscale=self._pscale)
      return prob

    # Otherwise distinguish between uncallable and callables
    if not self.__callable:
      return self._prob()
    return self._prob(values)

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values=None, suffix=None):
    vals = collections.OrderedDict()
    if isinstance(values, dict):
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
    else:
      vals.update({key: values for key in keys})
    rvs = self.ret_rvs()
    rv_dist_names = [rv.eval_dist_name(vals[rv.ret_name()], suffix) \
                     for rv in rvs]
    dist_name = ','.join(rv_dist_names)
    return dist_name

#-------------------------------------------------------------------------------
  def eval_delta(self, delta):

    if isinstance(delta, self.delta):
      return delta
    assert isinstance(delta, self.Delta),\
        "Unknown delta specification type: {}".format(delta)

    # Determine delta type, extract delta_scale and use random number generator
    delta_val = delta[0]
    delta_type = None
    if isinstance(delta, list):
      delta_type = list
      delta_val = delta_val[0]
    elif isinstance(delta_val, tuple):
      delta_type = tuple
      delta_val = delta_val[0]
    if isinstance(delta_val, set):
      assert np.isfinite(self._length), \
          "Length is infinite and therefore precludes scaling along length"
      delta_val = list(delta_val)[0]
      delta_lim = abs(delta_val) * float(self._length)
    else:
      delta_lim = abs(delta_val)
    deltas = np.random.uniform(-delta_lim, delta_lim, size=self._nrvs)

    # Either scale with positive delta_val denoting hyperspherical step...
    if self._cfun is None:
      if delta_val > 0.:
        root_sum_squares = real_sqrt(np.sum(deltas ** 2))
        deltas = deltas * delta_lim / root_sum_squares
    
    # ...or multiply the LUD with the deltas...
    elif self.__cfun_lud is not None:
      deltas = self.__cfun_lud.dot(deltas)

    # ...or call supplied covariance function...
    else:
      deltas = np._cfun(deltas)

    # Package RV deltas by delta type
    delta_args = [None] * self._nrvs
    for i, key in enumerate(self._keys):
      arg = deltas[i]
      if delta_type is list:
        arg = [arg]
      elif delta_type is tuple:
        arg = (arg,)
      delta_args[i] = arg

    return self.delta(*tuple(delta_args))

#-------------------------------------------------------------------------------
  def eval_succ(self, pred_vals, succ_vals, reverse=False):
    """ Returns adjusted succ_vals """
    rvs = self.ret_rvs(aslist=True)
    if isinstance(succ_vals, self.Delta):
      succ_vals = self.eval_delta(succ_vals)
    succ_delta = None
    if isinstance(succ_vals, self.delta):
      succ_values = collections.OrderedDict()
      for i, key in enumerate(self._keys):
        succ_values[key] = rvs[i].apply_delta(pred_vals[key], 
                                              getattr(succ_vals, key))
      succ_vals =  succ_values
    elif isunitsetint(succ_vals):
      assert self._tfun is not None and self._tfun_ret_callable(),\
          "Transitional CDF calling requires callable tfun"

    # TODO: to increase capability of this section to cope beyond scalars
    dims = {}
    kwargs = {'reverse': reverse}
    vals = collections.OrderedDict()
    for key in self._keys:
      vals.update({key: pred_vals[key]})
    for key in self._keys:
      mod_key = key+"'"
      succ_key = key if mod_key not in succ_vals else mod_key
      vals.update({key+"'": succ_vals[mod_key]})
    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def eval_tran(self, vals, **kwargs):
    reverse = False if 'reverse' not in kwargs else kwargs['reverse']
    if self._tran is None:
      rvs = self.ret_rvs(aslist=True)
      pred_vals = dict()
      succ_vals = dict()
      for key, val in vals.items():
        if key[-1] == "'":
          succ_vals.update({key[:-1]: val})
        else:
          pred_vals.update({key: val})
      cond, _ = rv_prod_rule(pred_vals, succ_vals, rvs=rvs, pscale=self._pscale)
    else:
      assert self._tran.ret_callable(), \
          "Only callable transitional functions supported for multidimensionals"
      prob = self._tran if not self._tran.ret_istuple() else \
             self._tran[int(reverse)]
      cond = prob(**vals)
    return cond

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Returns a joint distribution p(args) """
    if self._rvs is None:
      return None
    iid = False if 'iid' not in kwds else kwds.pop('iid')
    if type(iid) is bool and iid:
      iid = self._defiid
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values)
    vals, dims = self.eval_vals(values, _skip_parsing=True)
    prob = self.eval_prob(vals)
    if not iid: 
      return Dist(dist_name, vals, dims, prob, self._pscale)
    return Dist(dist_name, vals, dims, prob, self._pscale).prod(iid)

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    """ Returns a conditional distribution p(args[1] | args[0]) """
    reverse = False if 'reverse' not in kwds else kwds.pop('reverse')
    pred_vals, succ_vals = None, None 
    if len(args) == 1:
      if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        pred_vals, succ_vals = args[0][0], args[0][1]
      else:
        pred_vals = args[0]
    elif len(args) == 2:
      pred_vals, succ_vals = args[0], args[1]
    pred_vals = self.parse_args(pred_vals)
    dist_pred_name = self.eval_dist_name(pred_vals)
    pred_vals, pred_dims = self.eval_vals(pred_vals)
    vals, dims, kwargs = self.eval_succ(pred_vals, succ_vals, reverse=reverse)
    cond = self.eval_tran(vals, **kwargs)
    succ_vals = {key[:-1]: val for key, val in vals.items() if key[-1] == "'"}
    dist_succ_name = self.eval_dist_name(succ_vals, "'")
    dist_name = '|'.join([dist_succ_name, dist_pred_name])
    return Dist(dist_name, vals, dims, cond, self._pscale)

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = self._keys[key]
    if isinstance(key, str):
      if key not in self._keys:
        return None
    return self._rvs[key]

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    from prob.rv import RV
    from prob.sc import SC
    if isinstance(other, SC):
      marg = self.ret_rvs() + other.ret_marg().ret_rvs()
      cond = other.ret_cond().ret_rvs()
      return SC(marg, cond)

    if isinstance(other, SJ):
      rvs = self.ret_rvs() + other.ret_rvs()
      return SJ(*tuple(rvs))

    if isinstance(other, RV):
      rvs = self.ret_rvs() + [other]
      return SJ(*tuple(rvs))

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    from prob.rv import RV
    from prob.sc import SC
    if isinstance(other, SC):
      marg = self.ret_rvs() + other.ret_cond().ret_rvs()
      cond = other.ret_marg().ret_rvs()
      return SC(marg, cond)

    if isinstance(other, SJ):
      return SC(self, other)

    if isinstance(other, RV):
      return SC(self, other)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
