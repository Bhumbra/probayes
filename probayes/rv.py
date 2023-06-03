""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import warnings
import numpy as np
import sympy
from probayes.variable import Variable, DEFAULT_VNAME
from probayes.prob import Prob
from probayes.vtypes import uniform, VTYPES, isscalar, \
                        isunitset, isunitsetint, isunitsetfloat, issingleton
from probayes.pscales import rescale
from probayes.expression import Expression
from probayes.sympy_prob import bernoulli_prob, bernoulli_sfun
from probayes.rv_utils import uniform_prob, matrix_cond_sample, \
                          lookup_square_matrix
from probayes.distribution import Distribution
from probayes.pd import PD

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
It therefore requires a name for x (id), a variable alphabet set (vset), and its 
associated probability distribution function (prob).
"""

#-------------------------------------------------------------------------------
class RV (Variable, Prob):
  """ A random variable is a variable with a defined probability function.
  It therefore inherits from classes Variable and and Prob. Each instance therefore 
  requires a name, a variable set, and probability function. Additionally RV
  supports transitional probabilities the cdf/icdf equivalents specified using
  RV.set_tran() and RV.set_tfun() equivalents, and accessed using RV.step().

  :example:
  >>> import numpy as np
  >>> import probayes as pb
  >>> var = pb.RV('var', vtype=bool)
  >>> var.set_tran(np.array([0.2, 0.8, 0.3, 0.7]).reshape(2,2,))
  >>> step = var.step()
  >>> print(step.prob)
  [[0.2 0.8]
   [0.3 0.7]]
  """

  # Protected
  _tran = None          # Transitional prob - can be a matrix
  _tfun = None          # Like pfun for transitional conditionals
  _is_stochastic = True # Flag of whether stochastic

  # Private
  __def_prob = None   # Flag to denote prob is defaulted
  __sym_tran = None   # Flag to denote symmetric transitional
  __prime_key = None  # Modified-string to denote prime key for variable

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vtype=None,
                     prob=None,
                     *args,
                     **kwds):
    """ Initialises a random variable combining Variable and Prob initialisation
    except invertible monotonic must be specified separately using set_ufun().

    :param name: Name of the domain - string as valid identifier.
    :param vset: variable set over which domain defined (see Variable.vset).
    :param vtype: variable type (bool, int, or float).
    :param prob : uncallable or callable probability expression (see set_prob).
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.
    """

    Variable.__init__(self, name, vtype, vset)
    Prob.__init__(self, prob, *args, **kwds)
    self._tran, self._tfun = None, None

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name=DEFAULT_VNAME):
    self._name = name
    self._Delta = collections.namedtuple('ð', [self._name])
    self.__prime_key = self._name + "'"

#-------------------------------------------------------------------------------
  @property
  def vset(self):
    return self._vset

  @Variable.vset.setter
  def vset(self, vset=None):
    Variable.vset.fset(self, vset)
    if self.__def_prob:
      self._prob = None
      self._default_prob()

#-------------------------------------------------------------------------------
  @property
  def vtype(self):
    return self._vtype

  @Variable.vtype.setter
  def vtype(self, vtype=None):
    Variable.vtype.fset(self, vtype)
    if self.__def_prob:
      self._prob = None
      self._default_prob()

#-------------------------------------------------------------------------------
  @property
  def prob(self):
    return self._prob

  @property
  def def_prob(self):
    return self.__def_prob

  def set_prob(self, prob=None, *args, **kwds):
    """ Sets the probability and pscale with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.

    'pscale' is a reserved keyword. See Prob.pscale for explanation of how 
    pscale is used.
    """
    super().set_prob(prob, *args, **kwds)
    if self._prob is None:
      self._default_prob()
    else:
      self.__def_prob = False

    # Check uncallable probabilities commensurate with self._vset
    if self._vset is not None and \
        not self.callable and not self.isscalar:
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))

    # Reassign non-0.5 Bernoulli probabilities to a formal distribution
    if self._vtype in VTYPES[bool] and self.isscalar:
      prob = float(self._prob)
      if not np.isclose(prob, 0.5):
        super().set_prob(bernoulli_prob(self.icon, bias=prob))
        super().set_sfun(bernoulli_sfun, bias=prob)

#-------------------------------------------------------------------------------
  def _default_prob(self):
    """ Defaults unspecified probabilities to uniform over self._vset.
    This will not override a previously specified non-default probability.
    """

    if self._prob is not None and not self.__def_prob:
      return
    prob = rescale(self._nlhv, 'log', self._pscale)
    if self._ufun is not None and self.no_ucov:
      prob = prob + sympy.log(self._ufun.derinv.expr) if self._logp else \
             prob * self._ufun.derinv.expr
    super().set_prob(prob)
    self.set_tran(prob)
    self.__def_prob = True

#-------------------------------------------------------------------------------
  def set_pfun(self, pfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_prob(). It is necessary to set these
    functions if sampling variables with non-flat distributions.

    :param pfun: two-length tuple of callable functions
    :param *args: arguments to pass to callable functions
    :param **kwds: keywords to pass to callable functions
    """
    super().set_pfun(pfun, *args, **kwds)
    if self._ufun is None or self._pfun is None:
      return
    assert self._ufun is None, \
      "Cannot assign non-uniform distribution alongside " + \
      "values transformation functions"

#-------------------------------------------------------------------------------
  def set_ufun(self, ufun=None, *args, **kwds):
    """ Sets a monotonic invertible tranformation for the domain as a tuple of
    two functions in the form (transforming_function, inverse_function) 
    operating on the first argument with optional further args and kwds.

    :param ufun: two-length tuple of monotonic functions.
    :param *args: args to pass to ufun functions.
    :param **kwds: kwds to pass to ufun functions.

    Support for this transformation is only valid for float-type vtypes.
    """
    super().set_ufun(ufun, *args, **kwds)
    if self._ufun is None:
      return

    # Recalibrate defaulted probabilities for floating point vtypes
    if self._vtype in VTYPES[float]:
      self._default_prob()

    # Assert pfun is unspecified
    assert self._pfun is None, \
      "Cannot assign univariate function alongside CDF/ICDF specification"

#-------------------------------------------------------------------------------
  @property
  def tran(self):
    """ Returns the transitional probability expression if specified """
    return self._tran

  def set_tran(self, tran=None, *args, **kwds):
    """ Sets a transitional function as a conditional probability. This can
    be specified numerically or one or two callable functions.

    :param tran: conditional scalar, array, or callable function (see below).
    :param *args: args to pass to tran functions.
    :param **kwds: kwds to pass to tran functions.

    If tran is a scalar, array, or callable function, then the transitional
    conditionality is treated as symmetrical. If tran is a two-length tuple,
    then assymetry is assumed in the form: (p[var'|var], p[var|var']).

    If intending to sample from a transitional conditional probability density
    function, the corresponding (CDF, ICDF) must be set using set_tfun().
    """
    self._tran = tran
    self.__sym_tran = None
    if self._tran is None:
      return
    self._tran = Expression(self._tran, *args, **kwds)
    self.__sym_tran = not self._tran.ismulti
    if self._tran.callable or self._tran.isscalar:
      return
    assert self._vtype not in VTYPES[float],\
      "Scalar or callable transitional required for floating point data types"
    tran = self._tran() if self.__sym_tran else self._tran[0]()
    message = "Transition matrix must a square 2D Numpy array " + \
              "covering variable set of size {}".format(len(self._vset))
    assert isinstance(tran, np.ndarray), message
    assert tran.ndim == 2, message
    assert np.all(np.array(tran.shape) == len(self._vset)), message
    self.__sym_tran = np.allclose(tran, tran.T)

#-------------------------------------------------------------------------------
  @property
  def tfun(self):
    """ Returns the transitional CDF/ICDF expression if specified """
    return self._tfun

  def set_tfun(self, tfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_tran(). It is necessary to set these
    functions if conditionally sampling variables with continuous distributions.

    :param tfun: two-length tuple of callable functions
    :param *args: arguments to pass to tfun functions
    :param **kwds: keywords to pass to tfun functions
    """
    self._tfun = tfun if tfun is None else Expression(tfun, *args, **kwds)
    if self._tfun is None:
      return
    assert self._tfun.ismulti, "Tuple of two functions required"

#-------------------------------------------------------------------------------
  def evaluate(self, values, use_pfun=True):
    """ Evaluates value(s) belonging to the domain of the variable.

    :param values: None, set of a single integer, array, or scalar.
    :param use_pfun: boolean flag to make use of pfun if previously set.

    :return: a NumPy array of the values (see Variable.evaluate()):
    """
    use_pfun = use_pfun and self._pfun is not None and isunitsetint(values)

    # If not using pfun while random sampling, we use sfun for booleans
    if not use_pfun:
      if self._vtype in VTYPES[bool] and hasattr(self._sfun, 'expr'):
        if isunitsetint(values) and self._sfun.expr == bernoulli_sfun:
          number = list(values)[0]
          if not number or number < 0:
            number = number if not number else -number
            return Distribution(self._name, 
                                {self.name: self._sfun[None](number)})
      return super().evaluate(values)

    # Evaluate values from inverse cdf bounded within cdf limits
    number = list(values)[0]
    assert self.isfinite, \
        "Cannot evaluate {} values for bounds: {}".format(values, self._ulims)
    lims = self.pfun[0](self._ulims)
    values = uniform(
                     lims[0], lims[1], number, 
                     isinstance(self._vset[0], tuple),
                     isinstance(self._vset[1], tuple)
                    )
    return Distribution(self._name, {self.name: self.pfun[1](values)})

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    """ Evaluates the probability inputting optional args for callable cases

    :param values: values of the variable used for evaluating probabilities.
    :param *args: optional arguments for callable probability objects.
    :param **kwds: optional arguments to include pscale for rescaling.

    :return: evaluated probabilities
    """
    values = values[self.name] if isinstance(values, dict) else values
    if not self.isscalar:
      return super().eval_prob(values)
    return uniform_prob(values, 
                        prob=float(self._prob), 
                        inside=self._inside,
                        pscale=self._pscale)

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values, suffix=None):
    """ Evaluates a distribution name for a probability distribution based on
    the values set in the first input argument with an optional suffix. """
    name = self._name if not suffix else self._name + suffix
    if values is None:
      dist_str = name
    elif np.isscalar(values):
      dist_str = "{}={}".format(name, values)
    else:
      dist_str = name + "=[]"
    return dist_str

#-------------------------------------------------------------------------------
  def eval_step(self, pred_vals, succ_vals, reverse=False):
    """ Evaluates a successive values from previous values with an optional
    direction reversal flag, outputting a three-length tuple that includes the
    successive values in the first argument.

    :param pred_vals: predecessor values (NumPy array).
    :param succ_vals: succecessor values (see step()).
    :param reverse: boolean flag (default False) to reverse direction.

    :return vals: a dictionary including both predecessor and successor values.
    :return dims: a dictionary with dimension indices for the values in vals.
    :return kwargs: a dictionary that includes optional keywords for eval_tran()
    """

    if succ_vals is None:
      assert self._tran is not None, "No transitional function specified"
    if isinstance(pred_vals, dict):
      pred_vals = pred_vals[self.name]
    kwargs = dict() # to pass over to eval_tran()
    if succ_vals is None:
      if self._delta is None:
        succ_vals = {0} if isscalar(pred_vals) else pred_vals
      else:
        delta = self.eval_delta()
        succ_vals = self.apply_delta(pred_vals, delta)

    #---------------------------------------------------------------------------
    def _reshape_vals(pred, succ):
      dims = {}
      ndim = 0

      # Now reshape the values according to succ > prev dimensionality
      if issingleton(succ):
        dims.update({self._name+"'": None})
      else:
        dims.update({self._name+"'": ndim})
        ndim += 1
      if issingleton(pred):
        dims.update({self._name: None})
      else:
        dims.update({self._name: ndim})
        ndim += 1

      if ndim == 2: # pred_vals distributed along inner dimension:
        pred = pred.reshape([1, pred.size])
        succ = succ.reshape([succ.size, 1])
      return pred, succ, dims

    #---------------------------------------------------------------------------
    # Scalar treatment is the most trivial and ignores reverse
    if self._tran is None or self._tran.isscalar:
      if isunitsetint(succ_vals):
        succ_vals = self.evaluate(succ_vals, use_pfun=False)[self._name]
      elif isunitsetfloat(succ_vals):
        assert self._vtype in VTYPES[float], \
            "Inverse CDF sampling for scalar probabilities unavailable for " + \
            "{} data type".format(self._vtype)
        cdf_val = list(succ_vals)[0]
        lo, hi = min(self._limits), max(self._limits)
        succ_val = lo*(1.-cdf_val) + hi*cdf_val
        if self._ufun is not None:
          succ_val = self.ufun[-1](succ_val)

      prob = self._tran() if self._tran is not None else None
      pred_vals, succ_vals, dims = _reshape_vals(pred_vals, succ_vals)
                  
    # Handle discrete non-callables
    elif not self._tran.callable:
      if reverse and not self._tran.ismulti and not self.__sym_tran:
        warnings.warn("Reverse direction called from asymmetric transitional")
      prob = self._tran() if not self._tran.ismulti else \
             self._tran[int(reverse)]()
      if isunitset(succ_vals):
        succ_vals, pred_idx, succ_idx = matrix_cond_sample(pred_vals, 
                                                           succ_vals, 
                                                           prob=prob, 
                                                           vset=self._vset) 
        kwargs.update({'pred_idx': pred_idx, 'succ_idx': succ_idx})
      pred_vals, succ_vals, dims = _reshape_vals(pred_vals, succ_vals)

    # That just leaves callables
    else:
      kwds = {self._name: pred_vals}
      if isunitset(succ_vals):
        assert self._tfun is not None, \
            "Conditional sampling requires setting CDF and ICDF " + \
            "conditional functions using rv.set.tfun()"
        assert isscalar(pred_vals), \
            "Successor sampling only possible with scalar predecessors"
        succ_vals = list(succ_vals)[0]
        if type(succ_vals) in VTYPES[int] or type(succ_vals) in VTYPES[np.uint]:
          lo, hi = min(self._ulims), max(self._ulims)
          kwds.update({self._name+"'": np.array([lo, hi], dtype=float)})
          lohi = self._tfun[0](**kwds)
          lo, hi = float(min(lohi)), float(max(lohi))
          succ_vals = uniform(lo, hi, succ_vals,
                              isinstance(self._vset[0], tuple),
                              isinstance(self._vset[1], tuple))
        else:
          succ_vals = np.atleast_1d(succ_vals)
        kwds.update({self._name: pred_vals,
                     self._name+"'": succ_vals})
        succ_vals = self._tfun[1](**kwds)
      elif not isscalar(succ_vals):
        succ_vals = np.atleast_1d(succ_vals)
      pred_vals, succ_vals, dims = _reshape_vals(pred_vals, succ_vals)

    vals = collections.OrderedDict({self._name+"'": succ_vals,
                                    self._name: pred_vals})
    kwargs.update({'reverse': reverse})
    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def eval_tran(self, vals, **kwargs):
    """ Evaluates the transitional conditional probability for the dictionary 
    arguments in vals with optional keywords in **kwargs.
    """
    reverse = False if 'reverse' not in kwargs else kwargs['reverse']
    pred_vals, succ_vals = vals[self._name], vals[self._name+"'"]
    pred_idx = None if 'pred_idx' not in kwargs else kwargs['pred_idx'] 
    succ_idx = None if 'succ_idx' not in kwargs else kwargs['succ_idx'] 
    cond = None

    # No transitional means no conditional
    if self._tran is None:
      pass

    # Scalar treatment is the most trivial and ignores reverse
    elif self._tran.isscalar:
      prob = None if self._tran is None else self._tran()
      cond = uniform_prob(pred_vals,
                          succ_vals, 
                          prob=float(prob), 
                          inside=self._inside) 
                  

    # Handle discrete non-callables
    elif not self._tran.callable:
      prob = self._tran() if not self._tran.ismulti else \
             self._tran[int(reverse)]()
      cond = lookup_square_matrix(pred_vals,
                                  succ_vals, 
                                  sq_matrix=prob, 
                                  vset=self._vset,
                                  col_idx=pred_idx,
                                  row_idx=succ_idx) 


    # That just leaves callables
    else:
      prob = self._tran if not self._tran.ismulti else \
             self._tran[int(reverse)]
      kwds = {self._name: pred_vals,
              self._name+"'": succ_vals}
      cond = prob(**kwds)

    return cond

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ Return a probability distribution for the quantities in values. """
    dist_name = self.eval_dist_name(values)
    vals = self.evaluate(values)
    prob = self.eval_prob(vals)
    dims = {self._name: None} if isscalar(vals[self.name]) else {self._name: 0}
    return PD(dist_name, vals, dims=dims, prob=prob, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def step(self, *args, reverse=False):
    """ Returns a conditional probability distribution for quantities in args.

    :param *args: predecessor, successor values to evaluate conditionals.
    :param reverse: Boolean flag to evaluate conditional probability in reverse.

    :return a Dist instance of the conditional probability distribution
    """
    pred_vals, succ_vals = None, None 
    if len(args) == 1:
      if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        pred_vals, succ_vals = args[0][0], args[0][1]
      else:
        pred_vals = args[0]
    elif len(args) == 2:
      pred_vals, succ_vals = args[0], args[1]
    dist_pred_name = self.eval_dist_name(pred_vals)
    dist_succ_name = None
    if pred_vals is None and succ_vals is None and \
        self._vtype not in VTYPES[float]:
      dist_succ_name = self.eval_dist_name(succ_vals, "'")
    pred_vals = self.evaluate(pred_vals)
    vals, dims, kwargs = self.eval_step(pred_vals, succ_vals, reverse=reverse)
    cond = self.eval_tran(vals, **kwargs)
    if dist_succ_name is None:
      dist_succ_name = self.eval_dist_name(vals[self.__prime_key], "'")
    dist_name = '|'.join([dist_succ_name, dist_pred_name])

    # TODO - distinguish between probabilistic and non-probabistic outputs
    if cond is None:
      cond = 1.
      for val in vals.values():
        if isinstance(val, np.ndarray):
          cond = cond * np.ones(val.shape, dtype=float)

    return PD(dist_name, vals, dims=dims, prob=cond, pscale=self._pscale)

#-------------------------------------------------------------------------------
  def __and__(self, other):
    """ Combination operator between RV and another RV, RF, or SD. """
    from probayes.rf import RF
    from probayes.sd import SD
    if isinstance(other, SD):
      leafs = [self] + list(other.leafs.vars.values())
      stems = other.stems
      roots = other.roots
      args = RF(*tuple(leafs))
      if stems:
        args += list(stems.values())
      if roots:
        args += list(roots.values())
      return SD(*args)

    if isinstance(other, RF):
      rvs = [self] + list(other.vars.values())
      return RF(*tuple(rvs))

    if isinstance(other, RV):
      return RF(self, other)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __or__(self, other):
    """ Conditional operator between RV and another RV, RF, or SD. """
    from probayes.sd import SD
    return SD(self, other)

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
    """ If arg is a slice (i.e. RV[:]) then Variable icon is returned,
    otherwise the Prob.__getitem__ method is called.
    """
    if isinstance(arg, slice):
      return Variable.__getitem__(self, arg)
    return Prob.__getitem__(self, arg)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def RVs(names, vtype=None, vset=None, prob=None, *args, **kwds):
  if isinstance(names, str):
    names = names.split(',')
  varlist = [RV(name, vtype, vset, prob, *args, **kwds) for name in names]
  return tuple(varlist)

#-------------------------------------------------------------------------------
