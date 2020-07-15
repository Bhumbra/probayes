""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
import scipy.stats
from prob.vals import _Vals
from prob.prob import _Prob, is_scipy_stats_cont
from prob.dist import Dist
from prob.vtypes import eval_vtype, uniform, VTYPES, \
                        isscalar, isunitsetint, isunitsetfloat
from prob.pscales import rescale, NEARLY_POSITIVE_INF
from prob.func import Func
from prob.rv_utils import nominal_uniform_prob, nominal_uniform_cond


"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
It therefore requires a name for x (id), a variable alphabet set (vset), and its 
asscociated probability distribution function (prob).
"""

#-------------------------------------------------------------------------------
class RV (_Vals, _Prob):

  # Protected
  _name = "rv"      # Name of the random variable
  _tran = None      # Transitional prob - can be a matrix
  _tfun = None      # Like pfun for transitional conditionals

  # Private
  __sym_tran = None

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vtype=None,
                     prob=None,
                     pscale=None,
                     *args,
                     **kwds):
    self.set_name(name)
    self.set_vset(vset, vtype)
    self.set_prob(prob, pscale, *args, **kwds)
    self.set_vfun()

#-------------------------------------------------------------------------------
  def set_name(self, name):
    # Identifier name required
    self._name = name
    assert isinstance(self._name, str), \
        "Mandatory RV name must be a string: {}".format(self._name)
    assert self._name.isidentifier(), \
        "RV name must ba a valid identifier: {}".format(self._name)

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, pscale=None, *args, **kwds):
    super().set_prob(prob, pscale, *args, **kwds)

    # Default unspecified probabilities to uniform over self._vset is given
    if self._prob is None:
      if self._vset is None:
        return self.ret_callable()
      else:
        prob = 1.
        if self._vtype in (bool, int):
          nvset = len(self._vset)
          prob = NEARLY_POSITIVE_INF if not nvset else 1. / float(nvset)
        elif self._vtype in VTYPES[float]:
          lo, hi = self.get_bounds()
          prob = NEARLY_POSITIVE_INF if lo==hi else 1./float(hi - lo)
        if self._pscale != 1.:
          prob = rescale(prob, self._pscale)
        super().set_prob(prob, self._pscale)
        self.set_tran(prob)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable() and not self.ret_isscalar():
      assert len(self._prob()) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
    pset = self.ret_pset()
    if is_scipy_stats_cont(pset):
      if self._vtype not in VTYPES[float]:
        self.set_vset(self._vset, vtype=float)
    return self.ret_callable()
   
#-------------------------------------------------------------------------------
  def set_pfun(self, *args, **kwds):
    super().set_pfun(*args, **kwds)
    if self._vfun is None or self._pfun is None:
      return
    if self.ret_pfun(0) != scipy.stats.uniform.cdf or \
        self.ret_pfun(1) != scipy.stats.uniform.ppf:
      assert self._vfun is None, \
        "Cannot assign non-uniform distribution alongside " + \
        "values transformation functions"

#-------------------------------------------------------------------------------
  def set_vfun(self, *args, **kwds):
    super().set_vfun(*args, **kwds)
    if self._vfun is None:
      return

    # Recalibrate scalar probabilities
    if self.ret_isscalar() and \
        self._vtype in VTYPES[float]:
      lo, hi = self.get_bounds(use_vfun=True)
      prob = NEARLY_POSITIVE_INF if lo==hi else 1./float(hi - lo)
      if self._pscale != 1.:
        prob = rescale(prob, self._pscale)
      super().set_prob(prob, self._pscale)
    if self._pfun is None:
      return
    if self.ret_pfun(0) != scipy.stats.uniform.cdf or \
        self.ret_pfun(1) != scipy.stats.uniform.ppf:
      assert self._pfun is None, \
        "Cannot assign values tranformation function alongside " + \
        "non-uniform distribution"

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    self._tran = tran
    self.__sym_tran = None
    if self._tran is None:
      return self.__sym_tran
    self._tran = Func(self._tran, *args, **kwds)
    self.__sym_tran = not self._tran.ret_istuple()
    if self._tran.ret_callable() or self._tran.ret_isscalar():
      return self.__sym_tran
    assert self._vtype not in VTYPES[float],\
      "Scalar or callable transitional required for floating point data types"
    tran = self._tran() if self.__sym_tran else self._tran[0]()
    message = "Transition matrix must a square 2D Numpy array " + \
              "covering variable set of size {}".format(len(self._vset))
    assert isinstance(tran, np.ndarray), message
    assert tran.ndim == 2, message
    assert np.all(np.array(tran.shape) == len(self._vset)), message
    self.__sym_tran = np.allclose(tran, tran.T)
    return self.__sym_tran

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    # Provide cdf and inverse cdf for conditional sampling
    self._tfun = tfun if tfun is None else Func(tfun, *args, **kwds)
    if self._tfun is None:
      return
    self._tfun = Func(self._tfun, *args, **kwds)
    assert self._tfun.ret_istuple(), "Tuple of two functions required"
    assert len(self._tfun) == 2, "Tuple of two functions required."

#-------------------------------------------------------------------------------
  def eval_vals(self, values, use_pfun=True):
    use_pfun = use_pfun and self._pfun is not None and isunitsetint(values)
    if not use_pfun:
      return super().eval_vals(values)

    # Evaluate values from inverse cdf bounded within cdf limits
    number = list(values)[0]
    lo, hi = self.get_bounds(use_vfun=False)
    lohi = np.array([lo, hi], dtype=float)
    assert np.all(np.isfinite(lohi)), \
        "Cannot evaluate {} values for bounds: {}".format(values, vset)
    lims = self.ret_pfun(0)(lohi)
    lo, hi = float(min(lims)), float(max(lims))
    values = uniform(lo, hi, number)
    return self.ret_pfun(1)(values)

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    if not self.ret_isscalar():
      return super().eval_prob(values)
    prob = self._prob()
    vset = self._vset
    return nominal_uniform_prob(values, prob=prob, vset=vset)

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values, suffix=None):
    name = self._name if not suffix else self._name + suffix
    if values is None:
      dist_str = name
    elif np.isscalar(values):
      dist_str = "{}={}".format(name, values)
    else:
      dist_str = name + "=[]"
    return dist_str

#-------------------------------------------------------------------------------
  def eval_tran(self, prev_vals, next_vals, reverse=False):
    """ Returns adjusted next_vals and transitional probability """
    assert self._tran is None, "No transitional function specified"

    # Scalar treatment is the most trivial
    if self._tran.ret_isscalar():
      if isunitsetint(next_vals):
        next_vals = self.eval_vals(next_vals, use_pfun=False)
      elif isunitsetfloat(next_vals):
        assert self._vtype in VTYPES[float]
        # RESUME HERE
      prob = self._tran()
      vset = self._vset
      return next_vals, nominal_uniform_cond(prev_vals, 
                                             next_vals,
                                             prob=prob,
                                             vset=vset)

    # Handle discrete non-callables
    if not self._tran.ret_callable():
      pass

#-------------------------------------------------------------------------------
  def step(self, *args, reverse=False):
    prev_values, next_values = None, None 
    if len(args) == 1:
      if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        prev_values, next_values = args[0][0], args[0][1]
      else:
        prev_values, next_values = args[0], args[0]
    elif len(args) == 2:
      prev_values, next_values = args[0], args[1]
    if next_vals is None:
      if self._vtype in VTYPES[float]:
        next_vals = prev_vals
      else:
        next_vals = np.array(list(self._vset), dtype=self._vtype)
    dist_prev_name = self.eval_dist_name(prev_values)
    dist_next_name = self.eval_dist_name(next_values, "'")
    dist_name = '|',join([dist_next_name, dist_prev_name])
    prev_vals = self.eval_vals(prev_values)
    next_vals, prob = eval_tran(prev_vals, next_vals)

    # MORE NEEDED HERE
    
#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    dist_name = self.eval_dist_name(values)
    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    vals_dict = collections.OrderedDict({self._name: vals})
    dims = {self._name: None} if isscalar(vals) else {self._name: 0}
    return Dist(dist_name, vals_dict, dims, prob, self._pscale)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
