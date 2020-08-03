"""
A domain defines a variable set over which a function can be defined. It thus
comprises a name and variable set. The function itself is not supported although
invertible variable transformations are.
"""

#-------------------------------------------------------------------------------
import numpy as np
import collections
from prob.vtypes import eval_vtype, isunitsetint, isscalar, \
                        revtype, uniform, VTYPES
from prob.func import Func

#-------------------------------------------------------------------------------
DEFAULT_VSET = {False, True}

#-------------------------------------------------------------------------------
class Domain:

  # Public
  delta = None       # A named tuple generator
                    
  # Protected       
  _name = "var"      # Name of the variable
  _vset = None       # Variable set (array or 2-length tuple range)
  _vtype = None      # Variable type
  _vfun = None       # 2-length tuple of mutually inverting functions
  _lims = None       # Numpy array of bounds of vset
  _limits = None     # Transformed self._lims
  _length = None     # Difference in self._limits
  _inside = None     # Lambda function for defining inside vset
  _delta = None      # Default delta operation
  _delta_args = None # Optional delta arguments 
  _delta_kwds = None # Optional delta keywords 

#-------------------------------------------------------------------------------
  def __init__(self, name=None,
                     vset=None, 
                     vtype=None,
                     vfun=None,
                     *args,
                     **kwds):
    self.set_name(name)
    self.set_vset(vset, vtype)
    self.set_vfun(vfun, *args, **kwds)
    self.set_delta()

#-------------------------------------------------------------------------------
  def set_name(self, name):
    # Identifier name required
    self._name = name
    assert isinstance(self._name, str), \
        "Mandatory variable name must be a string: {}".format(self._name)
    assert self._name.isidentifier(), \
        "Variable name must ba a valid identifier: {}".format(self._name)
    self.delta = collections.namedtuple('ð', [self._name])

#-------------------------------------------------------------------------------
  def set_vset(self, vset=None, vtype=None):

    # Default vset to nominal
    if vset is None: 
      vset = list(DEFAULT_VSET)
    elif isinstance(vset, (set, range)):
      vset = sorted(vset)
    elif np.isscalar(self._vset):
      vset = [self._vset]
    elif isinstance(vset, tuple):
      assert len(vset) == 2, \
          "Tuple vsets contain pairs of values, not {}".format(vset)
      vset = sorted(vset)
      vset = [(vset[0],), (vset[1],)]
    elif isinstance(vset, np.ndarray):
      vset = np.sort(vset).tolist()
    else:
      assert isinstance(vset, list), \
          "Unrecognised vset specification: {}".format(vset)

    # At this point, vset can only be a list, but may contain tuples
    aretuples = ([isinstance(_vset, tuple) for _vset in vset])
    if not any(aretuples):
      if vtype is None:
        vset = np.array(vset)
        vtype = eval_vtype(vset)
      else:
        vset = np.array(vset, dtype=vtype)
    else:
      if vtype is not None:
        assert vtype in VTYPES[float], \
            "Bounded variables supported only for float vtypes, not {}".\
            format(vtype)
      vtype = float
      assert len(vset) == 2, \
          "Unrecognised set specification: {}".vset
      for i in range(len(vset)):
        if isinstance(vset[i], tuple):
          assert len(vset[i]) == 1, \
              "Unrecognised set specification: {}".vset[i]
          vset[i] = vtype(vset[i][0])
    self._vset = vset
    for i, istuple in enumerate(aretuples):
      if istuple:
        self._vset[i] = self._vset[i],
    self._vtype = eval_vtype(vtype)
    self._eval_lims()
    return self._vtype

#-------------------------------------------------------------------------------
  def _eval_lims(self):
    self._lims = None
    self._limits = None
    self._length = None
    self._inside = None
    
    if self._vset is None:
      return self._length

    if self._vtype not in VTYPES[float]:
      self._lims = np.array([min(self._vset), max(self._vset)])
      self._limits = self._lims
      self._length = True if self._vtype in VTYPES[bool] else len(self._vset)
      self._inside = lambda x: np.isin(x, self._vset, assume_unique=True)
      return self._length

    """ Evaluates the limits from vset assuming vtype is set """
    # Set up limits and inside function if float
    if any([isinstance(_vset, tuple) for _vset in self._vset]):
      lims = np.concatenate([np.array(_vset).reshape([1]) 
                             for _vset in self._vset])
    else:
      lims = np.array(self._vset)
    assert len(lims) == 2, \
        "Floating point vset must be two elements, not {}".format(self._vset)
    if lims[1] < lims[0]:
      vset = vset[::-1]
      self._vset = vset
    self._lims = np.sort(lims)
    self._limits = self._lims if self._vfun is None \
                   else self.ret_vfun(0)(self._lims)
    self._length = max(self._limits) - min(self._limits)

    # Now set inside function
    if not isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._lims[0],
                                              x <= self._lims[1])
    elif not isinstance(self._vset[0], tuple) and \
        isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._lims[0],
                                              x < self._lims[1])
    elif isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x > self._lims[0],
                                              x <= self._lims[1])
    else:
      self._inside = lambda x: np.logical_and(x > self._lims[0],
                                              x < self._lims[1])

#-------------------------------------------------------------------------------
  def set_vfun(self, vfun=None, *args, **kwds):
    self._vfun = vfun
    if self._vfun is None:
      return

    assert self._vtype in VTYPES[float], \
        "Values transformation function only supported for floating point"
    message = "Input vfun be a two-sized tuple of callable functions"
    assert isinstance(self._vfun, tuple), message
    assert len(self._vfun) == 2, message
    assert callable(self._vfun[0]), message
    assert callable(self._vfun[1]), message
    self._eval_lims()

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    """ Input argument delta may be:

    1. A callable function (for which args and kwds are passed on as usual).
    2. An rv.delta instance (this defaults all RV deltas).
    3. A scalar that may contained in a list:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly sampled from [-scalar to +scalar].
      c) Tuple - operation is +/-delta within the polarity randomised

      Optional keywords are (default False):
        'scale': Flag to denote scaling deltas to RV lengths
        'bound': Flag to constrain delta effects to RV bounds
    """
    self._delta = delta
    self._delta_args = args
    self._delta_kwds = dict(kwds)
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

#-------------------------------------------------------------------------------
  def ret_name(self):
    return self._name

#-------------------------------------------------------------------------------
  def ret_vtype(self):
    return self._vtype

#-------------------------------------------------------------------------------
  def ret_vfun(self, index=None):
    if self._vfun is None:
      return lambda x:x
    if index is None:
      return self._vfun
    return self._vfun[index]

#-------------------------------------------------------------------------------
  def ret_lims(self):
    return self._lims

#-------------------------------------------------------------------------------
  def ret_limits(self):
    return self._limits

#-------------------------------------------------------------------------------
  def ret_length(self):
    return self._length

#-------------------------------------------------------------------------------
  def ret_delta(self):
    return self._delta

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):

    # Default to arrays of complete sets
    if values is None:
      if self._vtype in VTYPES[float]:
        values = {0}
      else:
        return np.array(list(self._vset), dtype=self._vtype)

    # Sets may be used to sample from support sets
    if isunitsetint(values):
      number = list(values)[0]

      # Non-continuous
      if self._vtype not in VTYPES[float]:
        values = np.array(list(self._vset), dtype=self._vtype)
        if not number:
          values = values[np.random.randint(0, len(values))]
        else:
          if number > 0:
            indices = np.arange(number, dtype=int) % self._length
          else:
            indices = np.random.permutation(-number, dtype=int) % self._length
          values = values[indices]
        return values
       
      # Continuous
      else:
        assert np.all(np.isfinite(self._limits)), \
            "Cannot evaluate {} values for bounds: {}".format(
                values, self._limits)
        values = uniform(self._limits[0], self._limits[1], number, 
                           isinstance(self._vset[0], tuple), 
                           isinstance(self._vset[1], tuple)
                        )

      # Only use vfun when isunitsetint(values)
      if self._vfun:
        return self.ret_vfun(1)(values)
    return values
   

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    return eval_vals(values)

#------------------------------------------------------------------------------- 
  def eval_delta(self, delta=None):
    # Evaluates the delta but does not apply it
    delta = delta or self._delta
    if delta is None:
      return None
    if isinstance(delta, Func):
      if delta.ret_callable():
        return delta
      delta = delta()
    if isinstance(delta, self.delta):
      delta = delta[0]
    orand = isinstance(delta, tuple)
    urand = isinstance(delta, list)
    if orand:
      assert len(delta) == 1, "Tuple delta must contain one element"
      delta = delta[0]
      if self._vtype not in VTYPES[bool]:
        delta = delta if np.random.uniform() > 0.5 else -delta
    elif urand:
      assert len(delta) == 1, "List delta must contain one element"
      delta = delta[0]
      if self._vtype in VTYPES[bool]:
        pass
      elif self._vtype in VTYPES[int]:
        delta = np.random.randint(-delta, delta)
      else:
        delta = np.random.uniform(-delta, delta)
    assert isscalar(delta), "Unrecognised delta type: {}".format(delta)
    if self._delta_kwds['scale']:
      assert np.isfinite(self._length), "Cannot scale by infinite length"
      delta *= self._length
    return self.delta(delta)

#------------------------------------------------------------------------------- 
  def apply_delta(self, values, delta=None, bound=None):

    # Call eval_delta() if values is a list and return values if delta is None
    delta = delta or self._delta
    if isinstance(delta, Func):
      if delta.ret_callable():
        return delta(values)
      delta = delta()
    elif self._vtype not in VTYPES[bool]:
      if isinstance(delta, (list, tuple)):
        delta = self.eval_delta(delta)
    if isinstance(delta, self.delta):
      delta = delta[0]
    if delta is None:
      return values

    # Apply the delta, treating bool as a special case
    if self._vtype in VTYPES[bool]:
      orand = isinstance(delta, tuple)
      urand = isinstance(delta, list)
      if orand or urand:
        assert len(delta) == 1, "Tuple/list delta must contain one element"
        delta = delta[0]
        if isscalar(values) or orand:
          vals = values if delta > np.random.uniform() > 0.5 \
                 else np.logical_not(values)
        else:
          flip = delta > np.random.uniform(size=values.shape)
          vals = np.copy(values)
          vals[flip] = np.logical_not(vals[flip])
      else:
        vals = np.array(values, dtype=int) + np.array(delta, dtype=int)
        vals = np.array(np.mod(vals, 2), dtype=bool)
    elif self._vfun is None:
      vals = values + delta
    else:
      vals = self.ret_vfun(1)(self.ret_vfun(0)(values) + delta)
    vals = revtype(vals, self._vtype)

    # Apply bounds
    if bound is None:
      bound = False if 'bound' not in self._delta_kwds \
             else self._delta_kwds['bound']
    if not bound:
      return vals
    maybe_bounce = [False] if self._vtype not in VTYPES[float] else \
                   [isinstance(self._vset[0], tuple), 
                    isinstance(self._vset[1], tuple)]
    if not any(maybe_bounce):
      return np.maximum(self._lims[0], np.minimum(self._lims[1], vals))

    # Bouncing scalars and arrays without and with boolean indexing respectively
    if isscalar(vals):
      if all(maybe_bounce):
        if not self._inside(vals):
          vals = values
      elif maybe_bounce[0]:
        if vals < self._lims[0]:
          vals = values
        else:
          vals = np.minimum(self._lims[1], vals)
      else:
        if vals > self._lims[1]:
          vals = values
        else:
          vals = np.maximum(self._lims[0], vals)
    else:
      if all(maybe_bounce):
        outside = np.logical_not(self._inside(vals))
        vals[outside] = values[outside]
      elif maybe_bounce[0]:
        outside = vals <= self._lims[0]
        vals[outside] = values[outside]
        vals = np.minimum(self._lims[1], vals)
      else:
        outside = vals >= self._lims[1]
        vals[outside] = values[outside]
        vals = np.maximum(self._lims[0], vals)
    return vals

#-------------------------------------------------------------------------------
