""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
import scipy.stats
from prob.vals import _Vals
from prob.prob import _Prob, is_scipy_stats_cont
from prob.dist import Dist
from prob.vtypes import eval_vtype, isscalar, isunitsetint
from prob.pscales import NEARLY_POSITIVE_INF

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
It therefore requires a name for x (id), a variable alphabet set (vset), and its 
asscociated probability distribution function (prob).
"""
#-------------------------------------------------------------------------------
def nominal_uniform(vals=None, prob=1., vset=None):

  # Default to prob if no values
  if vals is None:
    return prob
  vtype = eval_vtype(vset)

  # If scalar, check within variable set
  if isscalar(vals):
    if vtype in [float, np.dtype('float32'), np.dtype('float64') ]:
      prob = 0. if vals < min(vset) or vals > max(vset) else prob
    else:
      prob = prob if vals in vset else 0.
    return prob

  # Otherwise treat as arrays
  vals = np.atleast_1d(vals)
  prob = np.tile(prob, vals.shape)

  # Handle nominal probabilities
  if vtype is bool:
    isfalse = np.logical_not(vals)
    prob[isfalse] = 1. - prob[isfalse]
    return prob

  # Otherwise treat as uniform within range
  if vtype in [float, np.dtype('float32'), np.dtype('float64')]:
    outside = np.logical_or(vals < min(vset), vals > max(vset))
    prob[outside] = 0.
  else:
    outside = np.array([val not in vset for val in vals], dtype=bool)
    prob[outside] = 0.

  return prob

#-------------------------------------------------------------------------------
class RV (_Vals, _Prob):

  # Protected
  _name = "rv"                # Name of the random variable

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
        elif self._vtype in [float, np.dtype('float32'), np.dtype('float64')]:
          lo, hi = self.get_bounds()
          prob = NEARLY_POSITIVE_INF if lo==hi else 1./float(hi - lo)
        if self._pscale != 1.:
          prob = rescale(prob, self._pscale)
        super().set_prob(prob, self._pscale)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable() and not self.ret_isscalar():
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
    pset = self.ret_pset()
    if is_scipy_stats_cont(pset):
      if self._vtype not in [float, np.dtype('float32'), np.dtype('float64')]:
        self.set_vset(self._vset, vtype=float)
    return self.ret_callable()
   
#-------------------------------------------------------------------------------
  def set_pfun(self, *args, **kwds):
    super().set_pfun(*args, **kwds)
    if self._pfun is not None:
      if self._pfun[0] != scipy.stats.uniform.cdf or \
          self._pfun[1] != scipy.stats.uniform.ppf:
        assert self._vfun is None, \
          "Cannot assign non-uniform distribution alongside " + \
          "values transformation functions"

#-------------------------------------------------------------------------------
  def set_vfun(self, *args, **kwds):
    super().set_vfun(*args, **kwds)
    if self._pfun is not None:
      if self._pfun[0] != scipy.stats.uniform.cdf or \
          self._pfun[1] != scipy.stats.uniform.ppf:
        assert self._vfun is None, \
          "Cannot values tranformation function alongside " + \
          "non-uniform distirbutions"

#-------------------------------------------------------------------------------
  def eval_vals(self, values):
    if self._pfun is None or not isunitsetint(values):
      return super().eval_vals(values)

    # Evaluate values from inverse cdf bounded within cdf limits
    number = list(values)[0]
    lo, hi = self.get_bounds()
    lohi = np.atleast_1d([lo, hi])
    assert np.all(np.isfinite(lohi)), \
        "Cannot evaluate {} values for bounds: {}".format(values, vset)
    lims = self.pfun_0(lohi)
    lo, hi = float(min(lims)), float(max(lims))
    if number == 1:
      values = np.atleast_1d(0.5 * (lo+hi))
    elif number >= 0:
      values = np.linspace(lo, hi, number)
    else:
      values = np.random.uniform(lo, hi, size=-number)
    return self.pfun_1(values)

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    if not self.ret_isscalar():
      return super().eval_prob(values)
    return nominal_uniform(values, self._prob, self._vset)

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values):
    if values is None:
      dist_str = self._name
    elif np.isscalar(values):
      dist_str = "{}={}".format(self._name, values)
    else:
      dist_str = self._name + "=[]"
    return dist_str

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    if values is None:
      if isinstance(self._vset, set):
        assert self.ret_callable() is False, \
          "Values required for callable prob over continuous variable sets" 
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
