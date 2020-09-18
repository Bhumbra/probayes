""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
import scipy.stats
from probayes.domain import Domain
from probayes.prob import Prob, is_scipy_stats_cont
from probayes.dist import Dist
from probayes.vtypes import eval_vtype, uniform, VTYPES, isscalar, \
                        isunitset, isunitsetint, isunitsetfloat, issingleton
from probayes.pscales import div_prob, rescale, eval_pscale
from probayes.func import Func
from probayes.rv_utils import nominal_uniform_prob, matrix_cond_sample, \
                          lookup_square_matrix

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
It therefore requires a name for x (id), a variable alphabet set (vset), and its 
associated probability distribution function (prob).
"""

#-------------------------------------------------------------------------------
class RV (Domain, Prob):
  """ A random variable is a domain with a defined probability function.
  It therefore inherits from classes Domain and and Prob. Each instance therefore 
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
  _tran = None      # Transitional prob - can be a matrix
  _tfun = None      # Like pfun for transitional conditionals

  # Private
  __sym_tran = None
  __prime_key = None

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vtype=None,
                     prob=None,
                     pscale=None,
                     *args,
                     **kwds):
    """ Initialises a random variable combining Domain and Prob initialisation
    except invertible monotonic must be specified separately using set_mfun().

    :param name: Name of the domain - string as valid identifier.
    :param vset: variable set over which domain defined (see set_vset).
    :param vtype: variable type (bool, int, or float).
    :param prob: may be a scalar, array, or callable function.
    :param pscale: represents the scale used to represent probabilities.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.
    """

    self.set_name(name)
    self.set_vset(vset, vtype)
    self.set_prob(prob, pscale, *args, **kwds)
    self.set_mfun()
    self.set_delta()

#-------------------------------------------------------------------------------
  def set_name(self, name):
    """ Sets name of variable over which domain is defined:

    :param name: Name of the domain - string as valid identifier.
    """
    super().set_name(name)
    self.__prime_key = self._name + "'"

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, pscale=None, *args, **kwds):
    """ Sets the probability and pscale with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param pscale: represents the scale used to represent probabilities.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.

    See set_pscale() for explanation of how pscale is used.
    """
    self._tran, self._tfun = None, None
    if prob is not None:
      super().set_prob(prob, pscale, *args, **kwds)
    else:
      self._default_prob(pscale)

    # Check uncallable probabilities commensurate with self._vset
    if self._vset is not None and \
        not self.ret_callable() and not self.ret_isscalar():
      assert len(self._prob()) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))

    # If using scipy stats, ensure vset is float type
    pset = self.ret_pset()
    if is_scipy_stats_cont(pset):
      if self._vtype not in VTYPES[float]:
        self.set_vset(self._vset, vtype=float)
   
#-------------------------------------------------------------------------------
  def _default_prob(self, pscale=None):
    """ Defaults unspecified probabilities to uniform over self._vset """
    self._pscale = eval_pscale(pscale)
    if self._prob is None:
      if self._vset is None:
        return self.ret_callable()
      else:
        prob = div_prob(1., float(self._length))
        if self._pscale != 1.:
          prob = rescale(prob, 1., self._pscale)
        super().set_prob(prob, self._pscale)
        self.set_tran(prob)

#-------------------------------------------------------------------------------
  def set_pfun(self, pfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_prob(). It is necessary to set these
    functions if sampling variables with non-flat distributions.

    :param pfun: two-length tuple of callable functions
    :param *args: arguments to pass to pfun functions
    :param **kwds: keywords to pass to pfun functions
    """
    super().set_pfun(pfun, *args, **kwds)
    if self._mfun is None or self._pfun is None:
      return
    if self.ret_pfun(0) != scipy.stats.uniform.cdf or \
        self.ret_pfun(1) != scipy.stats.uniform.ppf:
      assert self._mfun is None, \
        "Cannot assign non-uniform distribution alongside " + \
        "values transformation functions"

#-------------------------------------------------------------------------------
  def set_mfun(self, mfun=None, *args, **kwds):
    """ Sets a monotonic invertible tranformation for the domain as a tuple of
    two functions in the form (transforming_function, inverse_function) 
    operating on the first argument with optional further args and kwds.

    :param mfun: two-length tuple of monotonic functions.
    :param *args: args to pass to mfun functions.
    :param **kwds: kwds to pass to mfun functions.

    Support for this transformation is only valid for float-type vtypes.
    """
    super().set_mfun(mfun, *args, **kwds)
    if self._mfun is None:
      return

    # Recalibrate scalar probabilities for floating point vtypes
    if self.ret_isscalar() and \
        self._vtype in VTYPES[float]:
      self._default_prob(self._pscale)

    # Check pfun is unspecified or uniform
    if self._pfun is None:
      return
    if self.ret_pfun(0) != scipy.stats.uniform.cdf or \
        self.ret_pfun(1) != scipy.stats.uniform.ppf:
      assert self._pfun is None, \
        "Cannot assign values tranformation function alongside " + \
        "non-uniform distribution"

#-------------------------------------------------------------------------------
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
    self._tran = Func(self._tran, *args, **kwds)
    self.__sym_tran = not self._tran.ret_istuple()
    if self._tran.ret_callable() or self._tran.ret_isscalar():
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
  def set_tfun(self, tfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_tran(). It is necessary to set these
    functions if conditionally sampling variables with continuous distributions.

    :param tfun: two-length tuple of callable functions
    :param *args: arguments to pass to tfun functions
    :param **kwds: keywords to pass to tfun functions
    """
    self._tfun = tfun if tfun is None else Func(tfun, *args, **kwds)
    if self._tfun is None:
      return
    assert self._tfun.ret_istuple(), "Tuple of two functions required"
    assert len(self._tfun) == 2, "Tuple of two functions required."

#-------------------------------------------------------------------------------
  def ret_tran(self):
    """ Returns the set transitional conditional probability object(s) """
    return self._tran

#-------------------------------------------------------------------------------
  def ret_tfun(self):
    """ Returns the set transitional conditional (CDF, ICDF) object(s) """
    return self._tfun

#-------------------------------------------------------------------------------
  def eval_vals(self, values, use_pfun=True):
    """ Evaluates value(s) belonging to the domain of the variable.

    :param values: None, set of a single integer, array, or scalar.
    :param use_pfun: boolean flag to make use of pfun if previously set.

    :return: a NumPy array of the values (see Domain.eval_vals()):
    """
    use_pfun = use_pfun and self._pfun is not None and isunitsetint(values)
    if not use_pfun:
      return super().eval_vals(values)

    # Evaluate values from inverse cdf bounded within cdf limits
    number = list(values)[0]
    assert np.all(np.isfinite(self._lims)), \
        "Cannot evaluate {} values for bounds: {}".format(values, self._lims)
    lims = self.ret_pfun(0)(self._lims)
    values = uniform(
                     lims[0], lims[1], number, 
                     isinstance(self._vset[0], tuple),
                     isinstance(self._vset[1], tuple)
                    )
    return self.ret_pfun(1)(values)

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    """ Evaluates the probability inputting optional args for callable cases

    :param values: values of the variable used for evaluating probabilities.
    :param *args: optional arguments for callable probability objects.
    :param **kwds: optional arguments to include pscale for rescaling.

    :return: evaluated probabilities
    """
    if not self.ret_isscalar():
      return super().eval_prob(values)
    return nominal_uniform_prob(values, 
                                prob=self._prob(), 
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

    assert self._tran is not None, "No transitional function specified"
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
    if self._tran.ret_isscalar():
      if isunitsetint(succ_vals):
        succ_vals = self.eval_vals(succ_vals, use_pfun=False)
      elif isunitsetfloat(succ_vals):
        assert self._vtype in VTYPES[float], \
            "Inverse CDF sampling for scalar probabilities unavailable for " + \
            "{} data type".format(self._vtype)
        cdf_val = list(succ_vals)[0]
        lo, hi = min(self._limits), max(self._limits)
        succ_val = lo*(1.-cdf_val) + hi*cdf_val
        if self._mfun is not None:
          succ_val = self.ret_mfun(1)(succ_val)

      prob = self._tran()
      pred_vals, succ_vals, dims = _reshape_vals(pred_vals, succ_vals)
                  
    # Handle discrete non-callables
    elif not self._tran.ret_callable():
      if reverse and not self._tran.ret_istuple() and not self.__sym_tran:
        warning.warn("Reverse direction called from asymmetric transitional")
      prob = self._tran() if not self._tran.ret_istuple() else \
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
          lo, hi = min(self._lims), max(self._lims)
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

    # Scalar treatment is the most trivial and ignores reverse
    if self._tran.ret_isscalar():
      cond = nominal_uniform_prob(pred_vals,
                                  succ_vals, 
                                  prob=self._tran(), 
                                  inside=self._inside) 
                  

    # Handle discrete non-callables
    elif not self._tran.ret_callable():
      prob = self._tran() if not self._tran.ret_istuple() else \
             self._tran[int(reverse)]()
      cond = lookup_square_matrix(pred_vals,
                                  succ_vals, 
                                  sq_matrix=prob, 
                                  vset=self._vset,
                                  col_idx=pred_idx,
                                  row_idx=succ_idx) 


    # That just leaves callables
    else:
      prob = self._tran if not self._tran.ret_istuple() else \
             self._tran[int(reverse)]
      kwds = {self._name: pred_vals,
              self._name+"'": succ_vals}
      cond = prob(**kwds)

    return cond

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ Return a probability distribution for the quantities in values. """
    dist_name = self.eval_dist_name(values)
    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    dims = {self._name: None} if isscalar(vals) else {self._name: 0}
    vals = collections.OrderedDict({self._name: vals})
    return Dist(dist_name, vals, dims, prob, self._pscale)

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
    pred_vals = self.eval_vals(pred_vals)
    vals, dims, kwargs = self.eval_step(pred_vals, succ_vals, reverse=reverse)
    cond = self.eval_tran(vals, **kwargs)
    if dist_succ_name is None:
      dist_succ_name = self.eval_dist_name(vals[self.__prime_key], "'")
    dist_name = '|'.join([dist_succ_name, dist_pred_name])
    return Dist(dist_name, vals, dims, cond, self._pscale)
    
#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Print representation of RV name """
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    """ Logical 'AND' operator between RV and another RV, RJ, or RF. """
    from probayes.rj import RJ
    from probayes.rf import RF
    if isinstance(other, RF):
      marg = [self] + other.ret_marg().ret_rvs()
      cond = other.ret_cond().ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RJ):
      rvs = [self] + other.ret_rvs()
      return RJ(*tuple(rvs))

    if isinstance(other, RV):
      return RJ(self, other)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    """ Conditional operator between RV and another RV, RJ, or RF. """
    from probayes.rj import RJ
    from probayes.rf import RF
    if isinstance(other, RF):
      marg = [self] + other.ret_cond().ret_rvs()
      cond = other.ret_marg().ret_rvs()
      return RF(marg, cond)

    if isinstance(other, RJ):
      return RF(self, other)

    if isinstance(other, RV):
      return RF(self, other)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
