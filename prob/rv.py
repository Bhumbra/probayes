""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.vals import _Vals
from prob.prob import _Prob, is_scipy_stats_cont
from prob.dist import Dist
from prob.vtypes import eval_vtype
from prob.ptypes import NEARLY_POSITIVE_INF

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
  if np.isscalar(vals):
    if vtype in [bool, int]:
      prob = prob if vals in vset else 0.
    else:
      prob = 0. if vals < min(vset) or vals > max(vset) else prob
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
  if vtype in [int, np.dtype('int32'), np.dtype('int64')]:
    outside = np.array([val not in vset for val in vals], dtype=bool)
    prob[outside] = 0.
  else:
    outside = np.logical_or(vals < min(vset), vals > max(vset))
    prob[outside] = 0.

  return prob

#-------------------------------------------------------------------------------
def io_use_vfun(use_vfun=True):
  """ Fiddly function to interpret use_vfun for input/output purpses """
  use_vfun = use_vfun.lower() if isinstance(use_vfun, str) else use_vfun
  _use_vfun = None
  if use_vfun is None or use_vfun == 'none':
    use_vfun = False
  if isinstance(use_vfun, (list, tuple)) and len(use_vfun) == 2:
    _use_vfun = tuple(use_vfun)
  elif type(use_vfun) is bool:
    _use_vfun = (use_vfun, use_vfun)
  elif isinstance(use_vfun, str):
    if use_vfun == 'in':
      _use_vfun = (True, False)
    elif use_vfun == 'out':
      _use_vfun = (False, True)
    elif use_vfun == 'both':
      _use_vfun = (True, True)
  if _use_vfun is None:
    raise ValueError("Unable to interpret use_vfun input {}".format(use_vfun))
  return _use_vfun

#-------------------------------------------------------------------------------
class RV (_Vals, _Prob):

  # Protected
  _name = "rv"                # Name of the random variable
  _use_vfun = None            # 'in', 'out', 'both' (or True), 'None' (or None,False)

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vtype=None,
                     prob=None,
                     ptype=None,
                     *args,
                     **kwds):
    self.set_name(name)
    self.set_vset(vset, vtype)
    self.set_prob(prob, ptype, *args, **kwds)
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
  def set_prob(self, prob=None, ptype=None, *args, **kwds):
    super().set_prob(prob, ptype, *args, **kwds)

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
          lo, hi = self.get_bounds(use_vfun=False)
          prob = NEARLY_POSITIVE_INF if lo==hi else 1./float(hi - lo)
        if self._ptype != 1.:
          prob = rescale(prob, self._ptype)
        super().set_prob(prob, self._ptype)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable() and not self.ret_scalar():
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
    pset = self.ret_pset()
    if is_scipy_stats_cont(pset):
      if self._vtype not in [float, np.dtype('float32'), np.dtype('float64')]:
        self.set_vset(self._vset, vtype=float)
    return self.ret_callable()
   
#-------------------------------------------------------------------------------
  def set_vfun(self, vfun=None, *args, **kwds):
    super().set_vfun(vfun, *args, **kwds)
    self.set_use_vfun()

#-------------------------------------------------------------------------------
  def set_use_vfun(self, use_vfun=True):
    self._use_vfun = io_use_vfun(use_vfun)
    return self._use_vfun

#-------------------------------------------------------------------------------
  def ret_use_vfun(self):
    return self._use_vfun

#-------------------------------------------------------------------------------
  def eval_dist(self, values=None):
    if values is None:
      return self._name

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None):
    if not self.ret_scalar():
      return super().eval_prob(values)
    return nominal_uniform(values, self._prob, self._vset)

#-------------------------------------------------------------------------------
  def dist_dict(self, values):
    if values is None:
      dist_str = self._name
    elif np.isscalar(values):
      dist_str = "{}={}".format(self._name, values)
    else:
      dist_str = self._name + "=[]"
    return {self._name: dist_str}

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    if values is None:
      if isinstance(self._vset, set):
        assert self.ret_callable() is False, \
          "Values required for callable prob over continuous variable sets" 
    dist_dict = self.dist_dict(values)
    vals = self.eval_vals(values, self._use_vfun[0])
    prob = self.eval_prob(vals)
    vals = self.vfun_1(vals, self._use_vfun[1])
    dist_name = ','.join(dist_dict.values())
    vals_dict = collections.OrderedDict({self._name: vals})
    return Dist(dist_name, vals_dict, prob, self._ptype)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
