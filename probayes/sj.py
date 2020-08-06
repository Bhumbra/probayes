"""
A stochastic junction comprises a collection of a random variables that 
participate in a joint probability distribution function.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
import scipy.stats
from probayes.rv import RV
from probayes.dist import Dist
from probayes.dist_ops import margcond_str
from probayes.vtypes import isscalar, isunitsetint, issingleton, revtype
from probayes.pscales import iscomplex, real_sqrt, prod_rule, \
                         rescale, eval_pscale, prod_pscale
from probayes.sp_utils import sample_generator, \
                          metropolis_scores, hastings_scores, metropolis_update
from probayes.func import Func

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
  delta = None

  # Protected
  _name = None       # Cannot be set externally
  _rvs = None        # Dict of random variables
  _nrvs = None      
  _keys = None      
  _keyset = None    
  _defiid = None    
  _pscale = None    
  _prob = None      
  _pscale = None    
  _prop = None       # Non-transitional proposition function
  _delta = None      # Delta function (to replace step)
  _delta_args = None # Optional delta args (must be dictionaries)
  _delta_kwds = None # Optional delta kwds
  _delta_type = None # Same as delta
  _tran = None       # Transitional proposition function
  _tfun = None       # CDF/IDF of transition function
  _cfun = None       # Covariance function
  _length = None     # Length of junction
  _lengths = None    # Lengths of RVs
  _sym_tran = None
  _cfun_lud = None
  _spherise = None

  # Private
  __isscalar = None
  __callable = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)
    self.set_prob()
    self.set_prop()
    self.set_delta()
    self.set_tran()
    self.set_cfun()

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
      self.delta = collections.namedtuple('ð', self._keys)
      self._delta_type = self.delta
    self.set_pscale()
    self.eval_length()
    return self._nrvs

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    # Set joint probability distribution function
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
  def set_pscale(self, pscale=None):
    if pscale is not None or not self._nrvs:
      self._pscale = eval_pscale(pscale)
      return self._pscale
    rvs = self.ret_rvs(aslist=True)
    pscales = [rv.ret_pscale() for rv in rvs]
    self._pscale = prod_pscale(pscales)
    return self._pscale

#-------------------------------------------------------------------------------
  def set_prop(self, prop=None, *args, **kwds):
    # Set joint proposition function
    self._prop = prop
    if self._prop is None:
      return
    assert self._tran is None, \
        "Cannot assign both proposition and transition probabilities"
    self._prop = Func(self._prop, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    """ Input argument delta may be:

    1. A callable function (for which args and kwds are passed on as usual).
    2. An sj.delta instance (this defaults all RV deltas).
    3. A dictionary for RVs, this is converted to an sj.delta.
    4. A scalar that may contained in a list or tuple:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly and independently sampled across RVs.
      c) Tuple - delta is spherically sampled across RVs.

      For non-tuples, an optional argument (args[0]) can be included as a 
      dictionary to specify by RV-name deltas following the above conventions
      except their values are not subject to scaling even if 'scale' is given,
      but they are subject to bounding if 'bound' is specified.

    For setting types 2-4, optional keywords are (default False):
      'scale': Flag to denote scaling deltas to RV lengths
      'bound': Flag to constrain delta effects to RV bounds (None bounces)
      
    """
    self._delta = delta
    self._delta_args = args
    self._delta_kwds = dict(kwds)
    self._spherise = {}
    if self._delta is None:
      return
    elif callable(self._delta):
      self._delta = Func(self._delta, *args, **kwds)
      return

    # Default scale and bound
    if 'scale' not in self._delta_kwds:
      self._delta_kwds.update({'scale': False})
    if 'bound' not in self._delta_kwds:
      self._delta_kwds.update({'bound': False})

    # Handle deltas and dictionaries
    if isinstance(self._delta, dict):
      self._delta = self._delta_type(**self._delta)
    if isinstance(delta, self._delta_type):
      assert not args, \
        "Optional args prohibited for dict/delta instance inputs"
      rvs = self.ret_rvs(aslist=True)
      for i, rv in enumerate(rvs):
        rv.set_delta(self._delta[i], scale=scale, bound=bound)
      return

    # Default scale and bound and check args
    scale = self._delta_kwds['scale']
    bound = self._delta_kwds['bound']
    if self._delta_args:
      assert len(self._delta_args) == 1, \
          "Optional positional arguments must comprises a single dict"
      unscale = self._delta_args[0]
      assert isinstance(unscale, dict), \
          "Optional positional arguments must comprises a single dict"

    # Non tuples can be converted to deltas; can pre-scale here
    if not isinstance(self._delta, tuple):
      scaling = self._lengths
      delta = self._delta 
      urand = isinstance(delta, list)
      if urand:
        assert len(delta) == 1, "List delta requires a single element"
        delta = delta[0]
      deltas = {key: delta for key in self._keys}
      unscale = {} if not self._delta_args else self._delta_args[0]
      deltas.update(unscale)
      delta_dict = collections.OrderedDict(deltas)
      for i, (key, val) in enumerate(deltas.items()):
        delta = val
        if scale and key not in unscale:
          assert np.isfinite(self._lengths[i]), \
              "Cannot scale by infinite length for RV {}".format(key)
          delta = val * self._lengths[i]
        if urand:
          delta = [delta]
        delta_dict.update({key: delta})
      self._delta = self._delta_type(**delta_dict)
      rvs = self.ret_rvs(aslist=True)
      for i, rv in enumerate(rvs):
        rv.set_delta(self._delta[i], scale=False, bound=bound)

    # Tuple deltas must be evaluated on-the-fly and cannot be pre-scaled
    else:
      unscale = {} if not self._delta_args else self._delta_args[0]
      self._spherise = {}
      for i, key in enumerate(self._keys):
        if key not in unscale.keys():
          length = self._lengths[i]
          assert np.isfinite(length), \
              "Cannot spherise RV {} with infinite length".format(key)
          self._spherise.update({key: length})

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    # Set transition function
    self._tran = tran
    self._sym_tran = False
    if self._tran is None:
      return
    assert self._prop is None, \
        "Cannot assign both proposition and transition probabilities"
    self._tran = Func(self._tran, *args, **kwds)
    self._sym_tran = not self._tran.ret_istuple()

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    # Set cdf and inverse cdf of transitional for conditional sampling
    self._tfun = tfun if tfun is None else Func(tfun, *args, **kwds)
    if self._tfun is None:
      return
    raise NotImplemented(
        "Multidimensional transitional CDF sampling not yet implemented")
    assert self._tfun.ret_istuple(), "Tuple of two functions required"
    assert len(self._tfun) == 2, "Tuple of two functions required."

#-------------------------------------------------------------------------------
  def set_cfun(self, cfun=None, *args, **kwds):
    # Set covariance function for kernel-based sampling
    self._cfun = cfun
    self._cfun_lud = None
    if self._cfun is None:
      return
    self._cfun = Func(self._cfun, *args, **kwds)
    if not self._cfun.ret_callable():
      message = "Non-callable cfun objects must be a square 2D Numpy array " + \
                "of size corresponding to number of variables {}".format(self._nrvs)
      assert isinstance(cfun, np.ndarray), message
      assert cfun.ndim == 2, message
      assert np.all(np.array(cfun.shape) == self._nrvs), message
      self._cfun_lud = np.linalg.cholesky(cfun)

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
  def eval_length(self):
    rvs = self.ret_rvs(aslist=True)
    self._lengths = np.array([rv.ret_length() for rv in rvs], dtype=float)
    self._length = np.sqrt(np.sum(self._lengths))
    return self._length

#-------------------------------------------------------------------------------
  def ret_length(self):
    return self._length

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def ret_id(self):
    return self._id
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
  def ret_pscale(self):
    return self._pscale

#-------------------------------------------------------------------------------
  def parse_args(self, *args, **kwds):
    """ Returns (values, iid) from *args and **kwds """
    args = tuple(args)
    kwds = dict(kwds)
    pass_all = False if 'pass_all' not in kwds else kwds.pop('pass_all')
    
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
      if not pass_all:
        assert seen_keys[-1] in self._keys, \
            "Unrecognised key {} among available RVs {}".format(
                seen_keys[-1], self._keys)
    for key in self._keys:
      if key not in seen_keys:
        values.update({key: None})
    if pass_all:
      list_keys = list(values.keys())
      for key in list_keys:
        if key not in self._keys:
          values.pop(key)

    return values

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values=None, suffix=None):
    # Evaluates the string used to set the distribution name
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
    elif isinstance(self._prob, Func) and self._prob.ret_isscipy():
      index = 1 if iscomplex(self._pscale) else 0
      return self._prob[index](values)
    return self._prob(values)

#-------------------------------------------------------------------------------
  def eval_delta(self, delta=None):

    # Handle native delta types within RV deltas
    if delta is None: 
      if self._delta is None:
        return None
      elif isinstance(self._delta, Func):
        delta = self._delta()
      elif isinstance(self._delta, self._delta_type):
        delta_dict = collections.OrderedDict()
        rvs = self.ret_rvs(aslist=True)
        for i, key in enumerate(self._keys):
          delta_dict.update({key: rvs[i].eval_delta()})
        delta = self._delta_type(**delta_dict)
      else:
        delta = self._delta
    elif isinstance(delta, Func):
      delta = delta()
    elif isinstance(delta, self._delta_type):
      delta_dict = collections.OrderedDict()
      rvs = self.ret_rvs(aslist=True)
      for i, key in enumerate(self._keys):
        delta_dict.update({key: rvs[i].eval_delta(delta[i])})
      delta = self._delta_type(**delta_dict)

    # Non spherical case
    if not isinstance(self._delta_type, Func) and \
         isinstance(delta, self._delta_type): # i.e. non-spherical
      if self._cfun is None:
        return delta
      elif self._cfun_lud is not None:
        delta = np.ravel(delta)
        delta = self._cfun_lud.dot(delta)
        return self._delta_type(*delta)
      else:
        delta = self._cfun(delta)
        assert isinstance(delta, self._delta_type), \
            "User supplied cfun did not output delta typoe {}".format(self._delta_type)
        return delta


    # Rule out possibility of all RVs contained in unscaling argument
    assert isinstance(delta, tuple), \
        "Unknown delta type: {}".format(delta)
    unscale = {} if not self._delta_args else self._delta_args
    if not len(self._spherise):
      return self._delta_type(**unscale)

    # Spherical version
    delta = delta[0]
    spherise = self._spherise
    keys = self._spherise.keys()
    rss = real_sqrt(np.sum(np.array(list(spherise.values()))**2))
    if self._delta_kwds['scale']:
      delta *= rss
    deltas = np.random.uniform(-delta, delta, size=len(spherise))
    rss_deltas = real_sqrt(np.sum(deltas ** 2.))
    deltas = (deltas * delta) / rss_deltas
    delta_dict = collections.OrderedDict()
    rvs = [self[key] for key in keys]
    idx = 0
    for i, key in enumerate(keys):
      if key in unscale:
        val = unscale[key]
      else:
        val = deltas[idx]
        idx += 1
        if self._delta_kwds['scale']:
          val *= self._lengths[i]
      delta_dict.update({key: val})
    delta = self._delta_type(**delta_dict)
    if self._cfun is None:
      return delta
    elif self._cfun_lud is not None:
      delta = self._cfun_lud.dot(np.array(delta, dtype=float))
      return self._delta_type(*deltas)
    else:
      delta = self._cfun(delta)
      assert isinstance(delta, self._delta_type), \
          "User supplied cfun did not output delta type {}".format(self._delta_type)
      return delta

#-------------------------------------------------------------------------------
  def apply_delta(self, values, delta=None):
    delta = delta or self._delta
    if delta is None:
      return values
    assert isinstance(delta, self._delta_type), \
          "Cannot apply delta without providing delta type {}".format(self._delta_type)
    bound = False if 'bound' not in self._delta_kwds \
           else self._delta_kwds['bound']
    vals = collections.OrderedDict(values)
    keys = delta._fields
    rvs = [self[key] for key in keys]
    for i, key in enumerate(keys):
      vals.update({key: rvs[i].apply_delta(values[key], delta[i], bound=bound)})
    return vals

#-------------------------------------------------------------------------------
  def eval_prop(self, values, **kwargs):
    if self._tran is not None:
      return self.eval_tran(values, **kwargs)
    if values is None:
      values = {}
    if self._prop is None:
      return self.eval_prob(values, **kwargs)
    if not self._prop.ret_callable():
      return self._prop()
    return self._prop(values)

#-------------------------------------------------------------------------------
  def eval_step(self, pred_vals, succ_vals, reverse=False):
    """ Returns adjusted succ_vals """

    # Evaluate deltas if required
    if succ_vals is None:
      succ_vals = self.eval_delta()
    elif isinstance(succ_vals, Func) or \
        isinstance(succ_vals, (tuple, self._delta_type)):
      succ_vals = self.eval_delta(succ_vals)

    # Apply deltas or (TODO: sample)
    if isinstance(succ_vals, self._delta_type):
      succ_vals = self.apply_delta(pred_vals, succ_vals)
    elif isunitsetint(succ_vals):
      assert self._tfun is not None and self._tfun_ret_callable(),\
          "Transitional CDF calling requires callable tfun"

    # TODO: to increase capability of this section to cope beyond scalars

    # Initialise outputs with predecessor values
    dims = {}
    kwargs = {'reverse': reverse}
    vals = collections.OrderedDict()
    for key in self._keys:
      vals.update({key: pred_vals[key]})
    if succ_vals is None and self._tran is None:
      return vals, dims, kwargs

    # If stepping or have a transition function, add successor values
    for key in self._keys:
      mod_key = key+"'"
      succ_key = key if mod_key not in succ_vals else mod_key
      vals.update({key+"'": succ_vals[succ_key]})

    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def eval_tran(self, values, **kwargs):
    reverse = False if 'reverse' not in kwargs else kwargs['reverse']
    if self._tran is None:
      rvs = self.ret_rvs(aslist=True)
      pred_vals = dict()
      succ_vals = dict()
      for key_, val in values.items():
        prime = key_[-1] == "'"
        key = key_[:-1] if prime else key_
        if key in self._keys:
          if prime:
            succ_vals.update({key: val})
          else:
            pred_vals.update({key: val})
      cond, _ = rv_prod_rule(pred_vals, succ_vals, rvs=rvs, pscale=self._pscale)
    else:
      assert self._tran.ret_callable(), \
          "Only callable transitional functions supported for multidimensionals"
      cond = self._tran(values) if self._sym_tran else \
             self._tran[int(reverse)](values)
    return cond

#-------------------------------------------------------------------------------
  def reval_tran(self, dist):
    """ Evaluates the conditional reverse-transition function for corresponding 
    transition conditional distribution dist. This requires a tuple input for
    self.set_tran() to evaluate a new conditional.
    """
    assert isinstance(dist, Dist), \
        "Input must be a distribution, not {} type.".format(type(dist))
    marg, cond = dist.cond, dist.marg
    name = margcond_str(marg, cond)
    vals = dist.vals
    dims = dist.dims
    prob = dist.prob if self._sym_tran else self._tran[1](dist.vals)
    pscale = dist.ret_pscale()
    return Dist(name, vals, dims, prob, pscale)

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

    # Deal with IID cases
    max_dim = None
    for dim in dims.values():
      if dim is not None:
        max_dim = dim if max_dim is None else max(dim, max_dim)

    # If scalar or prob is expected shape then perform product here
    if max_dim is None or max_dim == prob.ndim - 1:
      return Dist(dist_name, vals, dims, prob, self._pscale).prod(iid)

    # Otherwise it is left to the user function to perform the iid product
    for key in iid:
      vals[key] = {len(vals[key])}
      dims[key] = None
    return Dist(dist_name, vals, dims, prob, self._pscale)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    """ Returns a proposal distribution p(args[0]) for values """
    suffix = "'" if 'suffix' not in kwds else kwds.pop('suffix')
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values, suffix)
    vals, dims = self.eval_vals(values, _skip_parsing=True)
    prop = self.eval_prop(vals) if self._prop is not None else \
           self.eval_prob(vals)
    if suffix:
      keys = list(vals.keys())
      for key in keys:
        mod_key = key + suffix
        vals.update({mod_key: vals.pop(key)})
        if key in dims:
          dims.update({mod_key: dims.pop(key)})
    return Dist(dist_name, vals, dims, prop, self._pscale)

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    """ Returns a proposal distribution p(args[1]) given args[0], depending on
    whether using self._prop, that denotes a simple proposal distribution,
    or self._tran, that denotes a transitional distirbution. """

    reverse = False if 'reverse' not in kwds else kwds.pop('reverse')
    pred_vals, succ_vals = None, None 
    if len(args) == 1:
      if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        pred_vals, succ_vals = args[0][0], args[0][1]
      else:
        pred_vals = args[0]
    elif len(args) == 2:
      pred_vals, succ_vals = args[0], args[1]
    if succ_vals is None:
      if self._delta is not None:
        succ_vals = self._delta
      elif issingleton(pred_vals) and self._delta is None:
        succ_vals = {0}

    # Evaluate predecessor values
    pred_vals = self.parse_args(pred_vals, pass_all=True)
    dist_pred_name = self.eval_dist_name(pred_vals)
    pred_vals, pred_dims = self.eval_vals(pred_vals)

    # Evaluate successor evaluates
    vals, dims, kwargs = self.eval_step(pred_vals, succ_vals, reverse=reverse)
    succ_vals = {key[:-1]: val for key, val in vals.items() if key[-1] == "'"}
    cond = self.eval_tran(vals, **kwargs)
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
    if not self._name:
      return super().__repr__()
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
