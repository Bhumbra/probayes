""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.vals import _Vals
from prob.prob import _Prob
from prob.dist import Dist

"""
A random variable is a triple (x, A_x, P_x) defined for an outcome x for every 
possible realisation defined over the alphabet set A_x with probabilities P_x.
It therefore requires a name for x (id), a variable alphabet set (vset), and its 
asscociated probability distribution function (prob).
"""
#-------------------------------------------------------------------------------
def _uniform(vals, p=1., vset=None):
  p = float(p)
  probs = np.tile(p, vals.shape)
  if isinstance(vset, np.ndarray):
    outside = np.array([val not in vset for val in vals], dtype=bool)
    probs[outside] = 0.
  elif isinstance(vset, tuple) and len(vset) == 2:
    lo, hi = min(vset), max(vset)
    outside = np.logical_or(vals < lo, vals > hi)
    probs[outside] = 0.
  return probs

#-------------------------------------------------------------------------------
class RV (_Vals, _Prob):

  # Protected
  _name = "rv"                # Name of the random variable

  # Private
  __callable = None          # Flag to denote if prob is callable

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
        p = 1.
        if isinstance(self._vset, np.ndarray):
          nvset = len(self._vset)
          p = np.inf if not nvset else 1. / float(nvset)
        elif isinstance(self._vset, set) and len(self._vset) == 2:
          vset = np.array(list(self._vset), dtype=float)
          p = float(1. / (np.max(vset) - np.min(vset)))
        return super().set_prob(_uniform, ptype=self._ptype, p=p, vset=self._vset)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable():
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
    return self.ret_callable()
   
#-------------------------------------------------------------------------------
  def __call__(self, values=None, use_vfun=None):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    if use_vfun is None: use_vfun = values is None or isinstance(values, set)
    vals = self.eval_vals(values, use_vfun)
    prob = self.eval_prob(vals)
    vals = self.vfun_1(vals, use_vfun)
    return Dist(self._name, {self._name: vals}, prob, self._ptype)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
