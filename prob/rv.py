""" Random variable module """

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.vals import _Vals
from prob.prob import _Prob

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

  # Public
  name = "rv"                # Name of the random variable
  _rvid = None               # Same as name (used internally)

  # Protected
  _get = None                # A namedtuple for called realisations

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
    self.name = name
    assert isinstance(self.name, str), \
        "Mandatory RV name must be a string: {}".format(self.name)
    assert self.name.isidentifier(), \
        "RV name must ba a valid identifier: {}".format(self.name)
    self._rvid = self.name

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
        elif isinstance(self._vset, tuple) and len(self._vset) == 2:
          if self._vset[0] == self._vset[1]:
            p = np.inf
          else:
            p = 1. / (float(max(self._vset)) - float(min(self._vset)))
        return super().set_prob(_uniform, ptype=self._ptype, p=p, vset=self._vset)

    # Otherwise check uncallable probabilities commensurate with self._vset
    elif not self.ret_callable():
      assert len(self._prob) == len(self._vset), \
          "Probability of length {} incommensurate with Vset of length {}".format(
              len(self._prob), len(self._vset))
    return self.ret_callable()
   
#-------------------------------------------------------------------------------
  def __call__(self, values=None, **kwds):
    ''' 
    Returns a namedtuple of samp and prob.
    '''
    mutable = isinstance(values, (np.ndarray, list))
    vals = self.eval_vals(values)
    prob = self.eval_prob(vals)
    self._get = collections.namedtuple(self._rvid, ['vals', 'prob'], **kwds)

    # Reciprocate vals evaluated using vfun or just recall if mutable array
    vals = self.vfun_1(vals, mutable)
    return self._get(vals, prob)

#-------------------------------------------------------------------------------
  def __repr__(self):
    return super().__repr__() + ": '" + self.name + "'"

#-------------------------------------------------------------------------------
