"""
A domain defines a variable set over which a function can be defined. It thus
comprises a name and variable set. The function itself is not supported although
invertible variable transformations are.
"""

#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.vtypes import eval_vtype, isunitsetint, isscalar, \
                        revtype, uniform, VTYPES
from probayes.func import Func

#-------------------------------------------------------------------------------
DEFAULT_VSET = {False, True}

#-------------------------------------------------------------------------------
class Domain:
  """ Base class for probayes.RV although this class can be called itself.
  A domain defines a variable set over which a function can be defined. It
  therefore needs a name and corresponding variable. While this class
  does not support respective probability density functions (use RV for that), 
  it does include an optional to specify an invertible monotic variable 
  transformation function:

  :example:
  >>> import numpy as np
  >>> import probayes as pb
  >>> scalar = pb.Domain('scalar', [-np.inf, np.inf], vtype=float)
  >>> scalar.set_mfun((np.exp, np.log))
  >>> print(scalar.ret_limits())
  [ 0. inf]
  """

  # Public
  delta = None       # A named tuple generator
                    
  # Protected       
  _name = "var"      # Name of the variable
  _vset = None       # Variable set (array or 2-length tuple range)
  _vtype = None      # Variable type
  _mfun = None       # 2-length tuple of monotonic mutually inverting functions
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
                     mfun=None,
                     *args,
                     **kwds):
    """ Initialiser sets name, vset, and mfun:

    :param name: Name of the domain - string as valid identifier.
    :param vset: variable set over which domain defined (see set_vset).
    :param vtype: variable type (bool, int, or float).
    :param mfun: two-length tuple of monotonic functions (see set_mfun).
    :param *args: args to pass to mfun functions.
    :param **kwds: kwds to pass to mfun functions.

    Every Domain instance offers a factory function for delta specifications:

    :example:
    >>> import numpy as np
    >>> import probayes as pb
    >>> x = pb.Domain('x', [-np.inf, np.inf], vtype=float)
    >>> dx = x.delta(0.5)
    >>> print(x.apply_delta(1.5, dx))
    2.0
    """
    self.set_name(name)
    self.set_vset(vset, vtype)
    self.set_mfun(mfun, *args, **kwds)
    self.set_delta()

#-------------------------------------------------------------------------------
  def set_name(self, name):
    """ Sets name of variable over which domain is defined:

    :param name: Name of the domain - string as valid identifier.
    """
    # Identifier name required
    self._name = name
    assert isinstance(self._name, str), \
        "Mandatory variable name must be a string: {}".format(self._name)
    assert self._name.isidentifier(), \
        "Variable name must ba a valid identifier: {}".format(self._name)
    self.delta = collections.namedtuple('ð', [self._name])

#-------------------------------------------------------------------------------
  def set_vset(self, vset=None, vtype=None):
    """ Sets the variable set and variable type over which the domain is defined.

    :param vset: variable set over which domain defined (default [False, True])
    :param vtype: variable type (bool, int, or float; default bool).

    For non-float vtypes, vset may be a list, set, range, or NumPy array.

    For float vtypes, vset represents limits in the form:

    [lower, upper] - inclusive of both lower of upper values
    [(lower), upper] - exclusive of lower and inclusive of upper.
    [lower, (upper)] - inclusive of lower and exclusive of upper.
    [(lower), (upper)] - exclusive of both lower and upper values.

    The last case may also set using a simple two-value tuple:

    :example:
    >>> import probayes as pb
    >>> scalar = pb.Domain('scalar')
    >>> scalar.set_vset((1, 2), vtype=float)
    >>> print(scalar.ret_vset())
    [(1.0,), (2.0,)]
    """

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

#-------------------------------------------------------------------------------
  def _eval_lims(self):
    """ Evaluates untransformed (self._lims) and transformed (self._limits) 

    :returns: the length of the domain.
    """
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
    self._limits = self._lims if self._mfun is None \
                   else self.ret_mfun(0)(self._lims)
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
    return self._length

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
    self._mfun = mfun
    if self._mfun is None:
      return

    assert self._vtype in VTYPES[float], \
        "Values transformation function only supported for floating point"
    message = "Input mfun be a two-sized tuple of callable functions"
    assert isinstance(self._mfun, tuple), message
    assert len(self._mfun) == 2, message
    assert callable(self._mfun[0]), message
    assert callable(self._mfun[1]), message
    self._mfun = Func(self._mfun, *args, **kwds)
    self._eval_lims()

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    """ Sets the default delta operation for the domain.

    :param delta: a callable or uncallable argument (see below)
    :param *args: args to pass if delta is callable.
    :param **kwds: kwds to pass if delta is callable (except scale and bound)

    The first argument delta may be:

    1. A callable function (operating on the first term).
    2. A Domain.delta instance (this defaults all Domain deltas).
    3. A scalar that may or may not be contained in a container:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly sampled from [-scalar to +scalar].
      c) Tuple - operation is +/-delta within the polarity randomised

    Two reserved keywords can be passed for specifying (default False):
      'scale': Flag to denote scaling deltas to Domain lengths
      'bound': Flag to constrain delta effects to Domain bounds
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
    """ Returns the domain name """
    return self._name

#-------------------------------------------------------------------------------
  def ret_vset(self):
    """ Returns the variable set """
    return self._vset

#-------------------------------------------------------------------------------
  def ret_vtype(self):
    """ Returns the variable type """
    return self._vtype

#-------------------------------------------------------------------------------
  def ret_mfun(self, index=None):
    r""" Returns the monotonic invertible function(s). If not specified, then
    an identity lambda is passed.
    
    :param index: optional index $i$ if to isolate the $i$th function.

    :return: monotonic inverible function(s).

    .. warnings:: the flexibility of this interface comes at the cost of requiring
                  a maximum of ret_mfun() being called per line of code.
    """
    if self._mfun is None:
      return lambda x:x
    if index is None:
      return self._mfun
    return self._mfun[index]

#-------------------------------------------------------------------------------
  def ret_lims(self):
    """ Returns the untransformed limits """
    return self._lims

#-------------------------------------------------------------------------------
  def ret_limits(self):
    """ Returns the transformed limits """
    return self._limits

#-------------------------------------------------------------------------------
  def ret_length(self):
    """ Returns the length of the domain """
    return self._length

#-------------------------------------------------------------------------------
  def ret_delta(self):
    """ Returns the delta function if set """
    return self._delta

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):
    r""" Evaluates value(s) belonging to the domain.

    :param values: None, set of a single integer, array, or scalar.

    :return: a NumPy array of the values in accordance to the following:

    If values is a NumPy array, it is returned unchanged.

    If values is None, it defaults to the entire variable set (vset) if not
    the variable type vtype is not float; otherwise a single scalar within the
    vset is randomly evaluated (see below).

    If values is a set containing a single integer (i.e. $\{n\}$), , then the 
    output depends on the number $n$:

    If positive ($n$), then $n$ values are uniformly sampled.
    If zero ($n=0$), then a scalar value is randomly sampled.
    if negative($-n$), then $n$ values are randomly sampled.

    For non-float types, the values are evaluated from by ordered (if $n>0) or 
    random permutations of vset. For float types, then uniformly sampled is
    performed in accordance for any transformations set by Domain.set_mfun().
    """

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

      # Only use mfun when isunitsetint(values)
      if self._mfun:
        return self.ret_mfun(1)(values)
    return values

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ See Domain.eval_vals() 

    :example:

    >>> import numpy as np
    >>> import probayes as pb
    >>> freq = pb.Domain('freq', [1,8], vtype=float)
    >>> freq.set_mfun((np.log, np.exp))
    >>> print(freq({4})
    [1. 2. 4. 8.]
    """
    return self.eval_vals(values)

#------------------------------------------------------------------------------- 
  def eval_delta(self, delta=None):
    """ Evaluates the value(s) of a delta operation without applying them.

    :param delta: delta value(s) to offset (see Domain.apply_delta).
    :return: the evaluated delta offset values.
    :rtype Domain.delta()

    If delta is not entered, then the default set by Domain.set_delta() is used.
    """
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
    if delta == self._delta and self._delta_kwds['scale']:
      assert np.isfinite(self._length), "Cannot scale by infinite length"
      delta *= self._length
    return self.delta(delta)

#------------------------------------------------------------------------------- 
  def apply_delta(self, values, delta=None, bound=None):
    """ Applies delta operation  to values optionally contrained by bounds.

    :param values: Numpy array values to apply.
    :param delta: delta value(s) to offset to the values
    :param bound: optional argument to contrain outputs.

    :return: Returns the values following the delta operation.

    If delta is not entered, then the default set by Domain.set_delta() is used.
    Delta may be a scalar or a single scalar value contained in a tuple or list.

    1. A scalar value: is summated to values (transformed if mfun is specified).
    2. A tuple: the polarity of the scalar value is randomised for the delta.
    3. A list: the delta is uniformly sampled in the range [0, scalar].
    """

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
    elif self._mfun is None:
      vals = values + delta
    else:
      transformed_vals = self.ret_mfun(0)(values) + delta
      vals = self.ret_mfun(1)(transformed_vals)
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
