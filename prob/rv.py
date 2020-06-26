""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.vals import _Vals
from prob.prob import _Prob, NEARLY_POSITIVE_INF
from prob.dist import Dist

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

  # If scalar, check within variable set
  if np.isscalar(vals):
    if isinstance(vset, np.array):
      prob = prob if vals in vset else 0.
    elif isinstance(vset, set):
      prob = 0. if vals < min(vset) or vals > max(vset) else prob
    return prob

  # Otherwise treat as arrays
  vals = np.atleast_1d(vals)
  prob = np.tile(prob, vals.shape)

  # Handle nominal probabilities
  if isinstance(vset, np.ndarray) and vset.dtype is np.dtype('bool'):
    isfalse_vals = np.logical_not(vals)
    prob[isfalse_vals] = 1. - prob[isfalse_vals]
    return prob

  # Otherwise treat as uniform within range
  if isinstance(vset, np.ndarray):
    outside = np.array([val not in vset for val in vals], dtype=bool)
    prob[outside] = 0.
  elif isinstance(vset, tuple) and len(vset) == 2:
    lo, hi = min(vset), max(vset)
    outside = np.logical_or(vals < lo, vals > hi)
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
        if isinstance(self._vset, np.ndarray):
          nvset = len(self._vset)
          prob = NEARLY_POSITIVE_INF if not nvset else 1. / float(nvset)
        elif isinstance(self._vset, set) and len(self._vset) == 2:
          vset = np.array(list(self._vset), dtype=float)
          prob = float(1. / (np.max(vset) - np.min(vset)))
        if self._ptype is not None and self._ptype != 1.:
          prob = rescale(prob, self._ptype)
        super().set_prob(prob, self._ptype)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable() and not self.ret_scalar():
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
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
  def eval_prob(self, values=None):
    if not self.ret_scalar():
      return super().eval_prob(values)
    return nominal_uniform(values, self._prob, self._vset)

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    if values is None:
      if isinstance(self._vset, set):
        assert self.ret_callable() is False, \
          "Values required for callable prob over continuous variable sets" 
    vals = self.eval_vals(values, self._use_vfun[0])
    prob = self.eval_prob(vals)
    vals = self.vfun_1(vals, self._use_vfun[1])
    return Dist(self._name, {self._name: vals}, prob, self._ptype)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
