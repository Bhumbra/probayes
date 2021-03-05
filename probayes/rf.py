""" Provides Random Field class: RF """

#-------------------------------------------------------------------------------
import collections
import numpy as np

from probayes.field import Field
from probayes.rv import RV
from probayes.prob import Prob
from probayes.pd import PD
from probayes.pd_utils import margcond_str
from probayes.vtypes import isscalar, isunitsetint
from probayes.pscales import iscomplex, prod_pscale
from probayes.rf_utils import rv_prod_rule, sample_cond_cov
from probayes.expr import Expr
from probayes.expression import Expression
from probayes.cf import CF
from probayes.cond_cov import CondCov

DEFAULT_CONDITIONAL_PROBABILITY = {False: 1., True: 0.}

#-------------------------------------------------------------------------------
class RF (Field, Prob):
  """
  A random field is a collection of a variables, that includes at least one RV, 
  that participate in a joint probability distribution function without 
  explicit directional conditionality.
  
  Since this class is intended as a building block for SD instances and networkx 
  cannot mix undirected and directed graphs, edges cannot be defined explicitly 
  within this class. Use SD if directed edges are required. Implicit support for
  undirected edges is provided by the set_prob(), set_prop(), and set_tran()
  methods.
  """

  # Protected
  _defiid = None     # Default IID random variables for calling distributions
  _passdims = None   # Flag to pass dimensions to probability function
  _prop = None       # Non-transitional proposition function
  _prop_deps = None  # Set of proposition dependencies
  _tran = None       # Transitional proposition function
  _tfun = None       # CDF/IDF of transition function 
  _tsteps = None     # Number of steps per transitional modificiation
  _cvars = None      # Conditional random variable sampling specification
  _sym_tran = None   # Flag for symmetrical transitional conditional functions
  _is_stochastic = True # Flag of whether stochastic

  # Private
  __def_prob = None   # Flag to denote prob is defaulted
  __cond_mod = None  # conditional RV index modulus
  __cond_cov = None  # conditional covariance matrix

#-------------------------------------------------------------------------------
  def __init__(self, *args): # over-rides NX_GRAPH.__init__()
    """ Initialises a random field with RVs for in args. See set_vars(). """
    super().__init__(*args)
    self.set_prob()
    self.set_prop()
    self.set_tran()

#-------------------------------------------------------------------------------
  def add_cd(self, *args, **kwds):
    """ Adding of conditional dependences disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def _refresh(self):
    """ Updates RV summary objects, RF name and id, and delta factory. """
    super()._refresh()
    if self._nvars:
      pscales = [var.pscale for var in self._vars.values()]
      self.pscale = prod_pscale(pscales)
    self.eval_length()
    self._prop = None
    self._tran = None
    self._tfun = None

#-------------------------------------------------------------------------------
  @property
  def prob(self):
    return self._prob

  @property
  def passdims(self):
    return self.passdims

  @property
  def def_prob(self):
    return self.__def_prob

  def set_prob(self, prob=None, *args, **kwds):
    """ Sets the joint probability with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.

    'pscale' is a reserved keyword. See Prob.pscale for explanation of how 
    pscale is used. Keyword 'passdims' is reserved to pass a flag to callable
    prob functions to pass the dimensionality dictionary for values.
    """
    kwds = dict(kwds)
    self._passdims = False if 'passdims' not in kwds else kwds.pop('passdims')
    if 'pscale' not in kwds and self._nvars:
      pscales = [var.pscale for var in self._varlist]
      kwds.update({'pscale': prod_pscale(pscales)})
    super().set_prob(prob, *args, **kwds)
    if self._prob is None:
      self._default_prob()
      return
    assert self._anystoch, \
        "Cannot set field probability without any random variables"
    self.__def_prob = False

#-------------------------------------------------------------------------------
  def _default_prob(self):
    if not self._anystoch or (self._prob is not None and not self.__def_prob):
      return
    if all([var.isiconic for var in self._varlist]):
      prob = None
      for var in self._varlist:
        if var.is_stochastic:
          var_prob = var.prob
          mul = var_prob if not isinstance(var_prob, Expr) else var_prob._expr
          if prob is not None:
            prob = prob * mul
          else:
            prob = mul
      self.set_prob(prob, pscale=self._pscale)
      self.__def_prob = True

#-------------------------------------------------------------------------------
  def eval_length(self):
    """ Evaluates and returns the joint length of the field. """
    self._lengths = np.array([var.length for var in self._varlist \
                              if var.is_stochastic], dtype=float)
    self._length = np.sqrt(np.sum(self._lengths**2))
    return self._length

#-------------------------------------------------------------------------------
  @property
  def prop(self):
    """ Returns the proposal probability expression if specified """
    return self._prop

  def set_prop(self, prop=None, *args, **kwds):
    """ Sets the joint proposition function with optional arguments and keywords.

    :param prop: may be a scalar, array, or callable function.
    :param *args: optional arguments to pass if prop is callable.
    :param **kwds: optional keywords to pass if prop is callable.
    """
    self._prop = prop
    self._prop_deps = self._keylist if 'deps' not in kwds else kwds.pop['deps']
    if self._prop is None:
      return
    assert self._anystoch, \
        "Cannot set field proposition without any random variables"
    assert self._tran is None, \
        "Cannot assign both proposition and transition probabilities"
    self._prop = Expression(self._prop, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def tran(self):
    """ Returns the transitional probability expression if specified """
    return self._tran

  def set_tran(self, tran=None, *args, **kwds):
    """ Sets the conditional transition function with optional arguments and 
    keywords.

    :param tran: may be a scalar, covariance array, or callable function.
    :param *args: optional arguments to pass if tran is callable.
    :param **kwds: optional keywords to pass if tran is callable.
    """
    # Set transition function
    self._tran = tran
    self._sym_tran = False
    self._tsteps = None
    if self._tran is None:
      return
    assert self._anystoch, \
        "Cannot set field transitional without any random variables"
    assert self._prop is None, \
        "Cannot assign both proposition and transition probabilities"
    kwds = dict(kwds)
    if 'tsteps' in kwds:
      self._tsteps = kwds.pop('tsteps')
      if self._tsteps:
        assert type(self._tsteps) is int, "Input tsteps must be int"

    # Pass CF objects directly
    if isinstance(tran, CF):
      assert not args and not kwds, \
        "Setting args and kwds not supported if inputting a CF instance"
      self._tran = CF(self, 
                      tran.ret_inp(), 
                      tran.ret_func(), 
                      *tran.ret_args(), 
                      **tran.ret_kwds())
      self._sym_tran = not self._tran.ret_ismulti()
      return

    # Set as general function
    self._tran = Prob(tran, *args, **kwds)
    self._sym_tran = not self._tran.ismulti

    # If a covariance matrix, set the LU decomposition as the tfun
    if not self._tran.callable and not self._tran.isscalar:
      message = "Non-callable non-scalar tran objects must be a square 2D " + \
                "Numpy array of size corresponding to number of variables {}".\
                 format(self._nvars)
      assert isinstance(tran, np.ndarray), message
      assert tran.ndim == 2, message
      assert np.all(np.array(tran.shape) == self._nvars), message
      self.set_tfun(np.linalg.cholesky(tran))
      assert self._tsteps is None, \
          "Setting tsteps not supported for covariance transitions"
      return

    # If a scipy object, set the tfun
    elif self._tran.isscipy:
      scipyobj = self._tran.expr
      self.set_tfun(self._tran, scipyobj=scipyobj)
      return

    # Return for unit variable RFs
    elif self._unitvar:
      return

    # Otherwise reinstantiate self._tran as an explicit conditional function
    inp = None
    if len(args):
      if isinstance(args[0], (dict, list, tuple)):
        args = tuple(args)
        inp, args = args[0], args[1:]
    self._tran = CF(self, inp, tran, *args, **kwds)
    self._sym_tran = not self._tran.ret_ismulti()

#-------------------------------------------------------------------------------
  @property
  def tfun(self):
    """ Returns the transitional CDF/ICDF expression if specified """
    return self._tfun

  def set_tfun(self, tfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_tran(). It is necessary to set these
    functions for conditional sampling variables with non-flat distributions.

    :param tfun: two-length tuple of callable functions or an LU decomposition
    :param *args: arguments to pass to tfun functions
    :param **kwds: keywords to pass to tfun functions
    """
    scipyobj = None if 'scipyobj' not in kwds else kwds['scipyobj']
    self._tfun = tfun 
    if self._tfun is None:
      return
    if 'tsteps' in kwds:
      self._tsteps = kwds.pop('tsteps')

    # Pass CF objects directly
    if isinstance(tfun, CF):
      assert not args and not kwds, \
        "Setting args and kwds not supported if inputting a CF instance"
      self._tfun = CF(self, 
                      tfun.ret_inp(), 
                      tfun.ret_func(), 
                      *tfun.ret_args(), 
                      **tfun.ret_kwds())
      return

    # Handle SciPy objects specifically
    elif scipyobj is not None:      
      lims = np.array([rv.ulims for rv in self._varlist])
      mean = scipyobj.mean
      cov = scipyobj.cov
      self._cond_cov = CondCov(mean, cov, lims)
      scipy_cond = sample_cond_cov if self._sym_tran else \
                   (sample_cond_cov, sample_cond_cov)
      self._tfun = CF(self, None, scipy_cond, cond_cov=self._cond_cov)
      return

    # Non-callables are treated as LUD matrices
    elif not callable(self._tfun): 
        self._tfun = Prob(self._tfun, *args, **kwds)
        message = "Non-callable tran objects must be a triangular 2D Numpy array " + \
                  "of size corresponding to number of variables {}".format(self._nvars)
        assert isinstance(tfun, np.ndarray), message
        assert tfun.ndim == 2, message
        assert np.all(np.array(tfun.shape) == self._nvars), message
        assert np.allclose(tfun, np.tril(tfun)) or \
               np.allclose(tfun, np.triu(tfun)), message
        return

    # Otherwise instantiate a formal conditional function
    inp = None
    if len(args):
      if isinstance(args[0], (dict, list, tuple)):
        args = tuple(args)
        inp, args = args[0], args[1:]
    self._tfun = CF(self, inp, self._tfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None, dims=None):
    if values is None:
      values = {}
    else:
      assert isinstance(values, dict), \
          "Input to eval_prob() requires values dict"
      assert set(values.keys()) == self._keyset, \
        "Sample dictionary keys {} mismatch with RV names {}".format(
          values.keys(), self._keylist)

    # If not specified, treat as independent variables
    if self._prob is None or self.__def_prob:
      rvs = self._varlist
      if len(rvs) == 1 and rvs[0]._prob is not None:
        prob = rvs[0].eval_prob(values[rvs[0].name])
      else:
        prob, _ = rv_prod_rule(values, rvs=rvs, pscale=self._pscale)
      return prob

    # Otherwise distinguish between uncallable and callables
    if not self._callable:
      return self._call()
    if self.issympy:
      prob = self._partials['logp'](values) if iscomplex(self._pscale) else \
             self._partials['prob'](values)
      return prob

    # Pass-dims is to replaced when passing Distributions()
    if self._passdims:
      return super().eval_prob(values, dims=dims)
    return super().eval_prob(values)

#-------------------------------------------------------------------------------
  def eval_delta(self, delta=None):
    delta = super().eval_delta(delta)

    # Adjust delta if there is LUD tfun
    if self._tfun is None or self._tfun.isscalar:
      return delta
    elif not self._tfun.callable:
      delta = self._tfun().dot(np.array(delta, dtype=float))
      return self._delta_type(*delta)
    else:
      delta = self._tfun(delta)
      assert isinstance(delta, self._delta_type), \
          "User supplied tfun did not output delta type {}".\
          format(self._delta_type)
      return delta

#-------------------------------------------------------------------------------
  def eval_prop(self, values, **kwargs):
    if self._tran is not None:
      return self.eval_tran(values, **kwargs)
    if values is None:
      values = {}
    if self._prop is None:
      return self.eval_prob(values, **kwargs)
    if not self._prop.callable:
      return self._prop()
    return self._prop(values)

#-------------------------------------------------------------------------------
  def eval_step(self, pred_vals, succ_vals, reverse=False):
    """ Returns adjusted succ_vals """
    if succ_vals is None:
      if self._delta is None:
        if all([isscalar(pred_value) for pred_value in pred_vals]):
          succ_vals = {0}

    # If not sampling succeeding values, use deterministic call
    if not isunitsetint(succ_vals):
      return super().eval_step(pred_vals, succ_vals, reverse=reverse)

    if self._tfun is not None and self._tfun.callable:
      succ_vals = self.eval_tfun(pred_vals)
    elif self._nvars == 1:
      var = self._varlist[0]
      tran = var.tran
      tfun = var.tfun
      if (tran is not None and not tran.callable) or \
          (tfun is not None and tfun.callable):
        vals, dims, kwargs = var.eval_step(pred_vals[var.name], 
                                           succ_vals, reverse=reverse)
        return vals, dims, kwargs
      raise ValueError("Transitional CDF calling requires callable tfun")
    else:
      raise ValueError("Transitional CDF calling requires callable tfun")

    # Initialise outputs with predecessor values
    dims = {}
    kwargs = {'reverse': reverse}
    vals = collections.OrderedDict()
    for key in self._keylist:
      vals.update({key: pred_vals[key]})
    if succ_vals is None and self._tran is None:
      return vals, dims, kwargs

    # If stepping or have a transition function, add successor values
    for key in self._keylist:
      mod_key = key+"'"
      succ_key = key if mod_key not in succ_vals else mod_key
      vals.update({key+"'": succ_vals[succ_key]})

    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def eval_tfun(self, values, *args, reverse=False, **kwds):
    """ Evaluates tfun from values """

    # Handle non-callables first
    if self._tfun is None:
      return None
    if self._tfun.isscalar:
      if self._tfun.ismulti:
        return self._tfun[int(reverse)]() * values()
      else:
        return self._tfun() * values
    if not self._tfun.callable:
      if self._tfun.ismulti:
        return self._tfun[int(reverse)]().dot(values)
      else:
        return self._tfun().dot(values)

    # If values is not a dictionary then allow a custom call
    if not isinstance(values, dict):
      if self._tfun.ismulti:
        return self._tfun[int(reverse)](values, *args, **kwds).dot(values)
      else:
        return self._tfun(values, *args, **kwds)

    # Support variable sampling one-or-more at-a-time
    succ_vals = collections.OrderedDict(values)

    # For conditional functions, iterate through keys
    assert isinstance(self._tfun, CF), "Callable conditionals must be CF type"

    # Determine keys distinguishing between default and custom specifications
    inp = self._tfun.ret_inp() # ?
    keys = list(succ_vals.keys()) if not inp else list(inp.keys())
    if self._tsteps:
      if self.__cond_mod is None:
        self.__cond_mod = 0
      keys = keys[self.__cond_mod:(self.__cond_mod+self._tsteps)]
      self.__cond_mod += self._tsteps
      if self.__cond_mod >= len(succ_vals):
        self.__cond_mod = 0

    # Handle unspecified input key handling separately
    if not inp:
      if self._tfun.ismulti:
        for key in keys:
          succ_vals[key] = self._tfun[int(reverse)](succ_vals, *args, unknown=key, **kwds)
      else:
        for key in keys:
          succ_vals[key] = self._tfun(succ_vals, *args, unknown=key, **kwds)
      return succ_vals

    # Explicit conditionals require more sophisticated handling
    assert not reverse, \
        "Reverse-direction sampling not supported for explicit conditionals"
    assert self._tfun.ismulti, \
        "Transitional sampling function not specified as a multiple sampler"
    for key in keys:
      values = self._tfun[key](succ_vals, *args, **kwds)
      rv_names = [key] if ',' not in keys else key.split(',')
      if len(rv_names) > 1:
        assert isinstance(values, dict), \
            "Ambiguous output from CF {} for RV {}".format(self._tfun, key)
        for rv_key, rv_val in values.items():
          assert rv_key in succ_vals.keys(), \
              "Key {} not found in {}".format(rv_key, succ_vals.keys())
          succ_vals[key] = rv_val
      else:
        rv_name = rv_names[0]
        if isinstance(values, dict):
          assert rv_name in values.keys(), \
              "Value key {} not found from output {}".format(rv_name, values.keys())
          succ_vals[rv_name] = values[rv_name]
        else:
          succ_vals[rv_name] = values
    return succ_vals

#-------------------------------------------------------------------------------
  def eval_tran(self, values, **kwargs):
    if 'cond' in kwargs:
      return kwargs['cond']
    cond = DEFAULT_CONDITIONAL_PROBABILITY[iscomplex(self._pscale)]
    reverse = False if 'reverse' not in kwargs else kwargs['reverse']
    if self._tran is None:
      if self._tfun is not None: # tfun without tran means no cond. prob.
        return cond
      rvs = self._varlist
      if len(rvs) == 1 and rvs[0]._tran is not None:
        return rvs[0].eval_tran(values, **kwargs)
      pred_vals = dict()
      succ_vals = dict()
      for key_, val in values.items():
        prime = key_[-1] == "'"
        key = key_[:-1] if prime else key_
        if key in self._keylist:
          if prime:
            succ_vals.update({key: val})
          else:
            pred_vals.update({key: val})
      cond, _ = rv_prod_rule(pred_vals, succ_vals, rvs=rvs, pscale=self._pscale)
    elif not self._tran.callable or self._tran.isscipy:
      return cond
    else:
      cond = self._tran(values) if self._sym_tran else \
             self._tran[int(reverse)](values)
    return cond

#-------------------------------------------------------------------------------
  def reval_tran(self, dist):
    """ Evaluates the conditional reverse-transition function for corresponding 
    transition conditional distribution dist. This requires a tuple input for
    self.set_tran() to evaluate a new conditional.
    """
    assert isinstance(dist, PD), \
        "Input must be a distribution, not {} type.".format(type(dist))
    marg, cond = dist.cond, dist.marg
    name = margcond_str(marg, cond)
    vals = collections.OrderedDict(dist)
    dims = dist.dims
    # This next line needs to be modified to handle new API
    """
    prob = dist.prob if self._sym_tran or self._tran is None \
           else self._tran[1](dist)
    """
    prob = dist.prob
    pscale = dist.pscale
    return PD(name, vals, dims=dims, prob=prob, pscale=pscale)

#-------------------------------------------------------------------------------
  def _eval_iid(self, dist_name, vals, dims, prob, iid):
    if not iid: 
      return PD(dist_name, vals, dims=dims, prob=prob, pscale=self._pscale)

    # Deal with IID cases
    max_dim = None
    for dim in dims.values():
      if dim is not None:
        max_dim = dim if max_dim is None else max(dim, max_dim)

    # If scalar or prob is expected shape then perform product here
    if max_dim is None or max_dim == prob.ndim - 1:
      dist = PD(dist_name, vals, dims=dims, prob=prob, pscale=self._pscale)
      return dist.prod(iid)

    # Otherwise it is left to the user function to perform the iid product
    for key in iid:
      vals[key] = {len(vals[key])}
      dims[key] = None

    # Tidy up probability
    return PD(dist_name, vals, dims=dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Returns a joint distribution p(args) """
    if not self.anystoch:
      return Field.__call__(self, *args, **kwds)
    if not self._nvars:
      return None
    iid = False if 'iid' not in kwds else kwds.pop('iid')
    if type(iid) is bool and iid:
      iid = self._defiid
    if not kwds and len(args) == 1 and not isinstance(args[0], dict):
      arg = {key: args[0] for key in self._keyset}
      args = arg,
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values)
    vals, dims = self.evaluate(values, _skip_parsing=True)
    prob = self.eval_prob(vals, dims)
    return self._eval_iid(dist_name, vals, dims, prob, iid)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    """ Returns a proposal distribution p(args[0]) for values """
    suffix = "'" if 'suffix' not in kwds else kwds.pop('suffix')
    if not kwds and len(args) == 1 and not isinstance(args[0], dict):
      arg = {key: args[0] for key in self._keyset}
      args = arg,
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values, suffix)
    vals, dims = self.evaluate(values, _skip_parsing=True)
    prop = self.eval_prop(vals) if self._prop is not None else \
           self.eval_prob(vals, dims)
    if suffix:
      keys = list(vals.keys())
      for key in keys:
        mod_key = key + suffix
        vals.update({mod_key: vals.pop(key)})
        if key in dims:
          dims.update({mod_key: dims.pop(key)})
    return PD(dist_name, vals, dims=dims, prob=prop, pscale=self._pscale)

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

    # Evaluate predecessor values
    if not isinstance(pred_vals, dict):
      pred_vals = {key: pred_vals for key in self._keyset}
    pred_vals = self.parse_args(pred_vals, pass_all=True)
    dist_pred_name = self.eval_dist_name(pred_vals)
    pred_vals, pred_dims = self.evaluate(pred_vals)

    # Default successor values if None and delta is None
    if succ_vals is None and self._delta is None:
      pred_values = list(pred_vals.values())
      if all([isscalar(pred_value) for pred_value in pred_values]):
        succ_vals = {0}
      else:
        succ_vals = pred_vals

    # Evaluate successor evaluates
    vals, dims, kwargs = self.eval_step(pred_vals, succ_vals, reverse=reverse)
    succ_vals = {key[:-1]: val for key, val in vals.items() if key[-1] == "'"}
    cond = self.eval_tran(vals, **kwargs)
    dist_succ_name = self.eval_dist_name(succ_vals, "'")
    dist_name = '|'.join([dist_succ_name, dist_pred_name])

    return PD(dist_name, vals, dims=dims, prob=cond, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __eq__(self, other):
    """ Equality for RFs is defined as comprising the same RVs """
    if type(self) is not RF or type(other) is not RF:
      return super().__eq__(other)
    return self._keyset == other.keyset

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
    if not self.issympy or isinstance(arg, int) or \
        (isinstance(arg, str) and arg in self._keyset):
      return Field.__getitem__(self, arg)
    return Prob.__getitem__(self, arg)

#-------------------------------------------------------------------------------
