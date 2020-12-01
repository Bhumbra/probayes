"""
A variable defines a quantity with a defined set over which a function is valid. 
It thus comprises a name, variable type, and variable set. The function 
itself is not supported although invertible variable transformations are.
"""

#-------------------------------------------------------------------------------
import numpy as np
import scipy.stats
import sympy as sy
import collections
import sympy as sy
from probayes.symbol import Symbol
from probayes.vtypes import eval_vtype, isunitsetint, isscalar, \
                        revtype, uniform, VTYPES, OO, OO_TO_NP
from probayes.func import Func

# Defaults
DEFAULT_VNAME = 'var'
DEFAULT_VTYPE = bool

# Defult vsets by vtype
DEFAULT_VSETS = {bool: [False, True],
                  int: [0, 1], 
                float: [(-OO,), (OO,)]}

#-------------------------------------------------------------------------------
class Variable (Symbol):
  """ Base class for probayes.RV although this class can be called itself.
  A domain defines a variable with a defined set over which a function can be 
  defined. It therefore needs a name, variable type, and variable set. 
  
  While this class xdoes not support respective probability density functions 
  (use RV for that), it does include an optional to specify an invertible 
  monotonic variable transformation function:

  :example:
  >>> import numpy as np
  >>> import probayes as pb
  >>> scalar = pb.Variable('scalar', [-np.inf, np.inf], vtype=float)
  >>> scalar.set_ufun((np.exp, np.log))
  >>> print(scalar.ret_ulims())
  [ 0. inf]
  """

  # Public
  delta = None       # A named tuple generator
                    
  # Protected       
  _vtype = None      # Variable type (bool, int, or float)
  _vset = None       # Variable set (array or 2-length tuple range)
  _vlims = None      # Numpy array of bounds of vset
  _ufun = None       # Univariate function for variable transformation
  _ulims = None      # Transformed self._vlims
  _length = None     # Difference in self._ulims
  _inside = None     # Lambda function for defining inside vset
  _delta = None      # Default delta operation
  _delta_args = None # Optional delta arguments 
  _delta_kwds = None # Optional delta keywords 

#-------------------------------------------------------------------------------
  def __init__(self, name=None,
                     vtype=None,
                     vset=None, 
                     *args,
                     **kwds):
    """ Initialiser sets name, vset, and ufun:

    :param name: Name of the variable - string as valid identifier.
    :param vtype: variable type (bool, int, or float).
    :param vset: variable set over which variable domain defined.
    :param *args: optional arguments to pass onto symbol representation.
    :param **kwds: optional keywords to pass onto symbol representation.

    Every Variable instance offers a factory function for delta specifications:

    :example:
    >>> import numpy as np
    >>> import probayes as pb
    >>> x = pb.Variable('x', [-np.inf, np.inf], vtype=float)
    >>> dx = x.delta(0.5)
    >>> print(x.apply_delta(1.5, dx))
    2.0
    """
    self.name = name
    self.vtype = vtype
    if vset:
      self.vset = vset
    self.set_delta()

    # Setting the symbol comes last in order to pass vtype/vset assumptions
    self.set_symbol(self.name, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name=DEFAULT_VNAME):
    self._name = name
    self.delta = collections.namedtuple('ð', [self._name])

#-------------------------------------------------------------------------------
  @property
  def vtype(self):
    return self._vtype

  @vtype.setter
  def vtype(self, vtype=None):
    """ Sets the variable type (default bool). If the variable set if not set,
    then it is defaulted according to the variable type. """

    if vtype:
      self._vtype = eval_vtype(vtype)
      if self._vset is None:
        self.vset = DEFAULT_VSETS[self._vtype]
    elif self._vset is not None:
      self._vtype = eval_vtype(self._vset)

#-------------------------------------------------------------------------------
  @property
  def vset(self):
    return self._vset

  @vset.setter
  def vset(self, vset=None):
    """ Sets the variable set over which the valid values are defined.
    :param vset: variable set over which domain defined (defaulted by vtype:
                 if bool:  vset = {False, True})
                 if int:   vset = {0, 1})
                 if float: vset = (-OO, OO)

    For non-float vtypes, vset may be a list, set, range, or NumPy array.

    For float vtypes, vset represents limits in the form:

    [lower, upper] - inclusive of both lower of upper values
    [(lower), upper] - exclusive of lower and inclusive of upper.
    [lower, (upper)] - inclusive of lower and exclusive of upper.
    [(lower), (upper)] - exclusive of both lower and upper values.

    The last case may also set using a simple two-value tuple:

    :example:
    >>> import probayes as pb
    >>> scalar = pb.Variable('scalar', vtype=float, vset=(1,2))
    >>> print(scalar.ret_vset())
    [(1.0,), (2.0,)]
    """

    # Default vset to nominal
    if vset is None and self._vtype: 
      vset = DEFAULT_VSETS[self._vtypes]
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
    vtype = self._vtype
    for i, value in enumerate(vset):
      if isinstance(value, tuple):
        if vtype is None:
          vtype = float
        val =  list(value)[0]
        if val != OO and val != -OO:
          vset[i] = ((vtype)(val),) 
      elif value == OO or value == -OO:
        if vtype is None:
          vtype = float
      else:
        if vtype:
          vtype = eval_vtype(value)
        elif vtype != eval_vtype(value):
          raise TypeError("Ambiguous value type {} vs {}".format(
            vtype, eval_vtype(value)))
        vset[i] = (vtype)(value)

    # Now vset contents should all be of the same vtype
    self._vset = vset
    vtype = eval_vtype(vtype)
    if self._vtype:
      assert self._vtype == vtype, \
          "Variable type {} incomptable with type for vset {}".format(
              self._vtype, vtype)
    else:
      self._vtype = vtype
    self._eval_vlims()

#-------------------------------------------------------------------------------
  def _eval_vlims(self):
    """ Evaluates untransformed (self._vlims) and transformed (self._ulims) 

    :returns: the length of the variable.
    """
    self._vlims = None
    if self._vset is None:
      return self._eval_ulims()

    # Non-float limits are simple
    if self._vtype not in VTYPES[float]:
      self._vlims = np.array([min(self._vset), max(self._vset)])

    # Evaluates the limits from vset float
    assert len(self._vset) == 2, \
        "Floating point vset must be two elements, not {}".format(self._vset)
    lims = [None] * 2
    for i, limit in enumerate(self._vset):
        lim = limit if not isinstance(limit, tuple) else list(limit)[0]
        lims[i] = OO_TO_NP.get(lim, lim)

    if lims[1] < lims[0]:
      vset = self._vset[::-1]
      self._vset = vset
    self._vlims = np.sort(lims)
    return self._eval_ulims()

#-------------------------------------------------------------------------------
  def _eval_ulims(self):
    """ Evaluates transformed limits (self._ulims) and inside functions
    
    :returns: the length of the variable.
    """
    self._ulims = None
    self._length = None
    self._inside = None
    if self._vlims is None:
      return self._length

    # Non-floats do not support transformation
    if self._vtype not in VTYPES[float]:
      self._inside = lambda x: np.isin(x, self._vset, assume_unique=True)
      self._ulims = self._vlims
      self._length = True if self._vtype in VTYPES[bool] else len(self._vset)
      return self._length

    # Floating point limits are susceptible to transormation
    self._ulims = self._vlims if self._ufun is None \
                   else self.ret_ufun(0)(self._vlims)
    self._length = max(self._ulims) - min(self._ulims)

    # Now set inside function
    if not isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._vlims[0],
                                              x <= self._vlims[1])
    elif not isinstance(self._vset[0], tuple) and \
        isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x >= self._vlims[0],
                                              x < self._vlims[1])
    elif isinstance(self._vset[0], tuple) and \
        not isinstance(self._vset[1], tuple):
      self._inside = lambda x: np.logical_and(x > self._vlims[0],
                                              x <= self._vlims[1])
    else:
      self._inside = lambda x: np.logical_and(x > self._vlims[0],
                                              x < self._vlims[1])

    return self._length

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    """ Sets the default delta operation for the domain.

    :param delta: a callable or uncallable argument (see below)
    :param *args: args to pass if delta is callable.
    :param **kwds: kwds to pass if delta is callable (except scale and bound)

    The first argument delta may be:

    1. A callable function (operating on the first term).
    2. A Variable.delta instance (this defaults all Variable deltas).
    3. A scalar that may or may not be contained in a container:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly sampled from [-scalar to +scalar].
      c) Tuple - operation is +/-delta within the polarity randomised

    Two reserved keywords can be passed for specifying (default False):
      'scale': Flag to denote scaling deltas to Variable lengths
      'bound': Flag to constrain delta effects to Variable bounds
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
  def set_symbol(self, symbol=None, *args, **kwds):
    """ Sets the variable symbol and carries members over to this Variable """
    symbol = symbol or self._name

    # If no arguments passed, default assumptions based on vtype and limits
    if not args and not kwds:
      kwds = dict(**kwds)
      if self._vtype in VTYPES[float]:
        kwds.update({'integer': False})
        kwds.update({'finite': np.all(np.isfinite(self._vlims))})
        if np.max(self._vlims) > 0. and np.min(self._vlims) < 0.:
          pass
        elif np.min(self._vlims) >= 0.:
          kwds.update({'positive': True})
        elif np.max(self._vlims) <= 0:
          kwds.update({'negative': True})
        elif np.max(self._vlims) > 0.: 
          if isinstance(self._vset[0], tuple):
            kwds.update({'positive': True})
          else:
            kwds.update({'nonnegative': True})
        elif np.min(self._vlims) < 0. :
          if isinstance(self._vset[1], tuple):
            kwds.update({'negative': True})
          else:
            kwds.update({'nonpositive': True})
      elif self._vtype in VTYPES[int]:
        kwds.update({'integer': True})
        if np.max(self._vlims) > 0 and np.min(self._vlims) < 0:
          pass
        elif np.max(self._vlims) >= 0:
          kwds.update({'positive': True})
        elif np.min(self._vlims) <= 0:
          kwds.update({'negative': True})
    return Symbol.set_symbol(self, symbol, *args, **kwds)

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
    self._ufun = ufun
    if self._ufun is None:
      return

    assert self._vtype in VTYPES[float], \
        "Values transformation function only supported for floating point"
    message = "Input ufun be a two-sized tuple of callable functions"
    assert isinstance(self._ufun, tuple), message
    assert len(self._ufun) == 2, message
    assert callable(self._ufun[0]), message
    assert callable(self._ufun[1]), message
    self._ufun = Func(self._ufun, *args, **kwds)
    self._eval_ulims()

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
  def ret_ufun(self, index=None):
    r""" Returns the monotonic invertible function(s). If not specified, then
    an identity lambda is passed.
    
    :param index: optional index $i$ if to isolate the $i$th function.

    :return: monotonic inverible function(s).

    .. warnings:: the flexibility of this interface comes at the cost of requiring
                  a maximum of ret_ufun() being called per line of code.
    """
    if self._ufun is None:
      return lambda x:x
    if index is None:
      return self._ufun
    return self._ufun[index]

#-------------------------------------------------------------------------------
  def ret_vlims(self):
    """ Returns the untransformed limits """
    return self._vlims

#-------------------------------------------------------------------------------
  def ret_ulims(self):
    """ Returns the transformed limits """
    return self._ulims

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
    performed in accordance for any transformations set by Variable.set_ufun().
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
        assert np.all(np.isfinite(self._ulims)), \
            "Cannot evaluate {} values for bounds: {}".format(
                values, self._ulims)
        values = uniform(self._ulims[0], self._ulims[1], number, 
                           isinstance(self._vset[0], tuple), 
                           isinstance(self._vset[1], tuple)
                        )

      # Only use ufun when isunitsetint(values)
      if self._ufun:
        return self.ret_ufun(1)(values)
    return values

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ See Variable.eval_vals() 

    :example:

    >>> import numpy as np
    >>> import probayes as pb
    >>> freq = pb.Variable('freq', [1,8], vtype=float)
    >>> freq.set_ufun((np.log, np.exp))
    >>> print(freq({4})
    [1. 2. 4. 8.]
    """
    return self.eval_vals(values)

#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Printable representation of variable including name """
    return self._name

#------------------------------------------------------------------------------- 
  def eval_delta(self, delta=None):
    """ Evaluates the value(s) of a delta operation without applying them.

    :param delta: delta value(s) to offset (see Variable.apply_delta).
    :return: the evaluated delta offset values.
    :rtype Variable.delta()

    If delta is not entered, then the default set by Variable.set_delta() is used.
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

    If delta is not entered, then the default set by Variable.set_delta() is used.
    Delta may be a scalar or a single scalar value contained in a tuple or list.

    1. A scalar value: is summated to values (transformed if ufun is specified).
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
    elif self._ufun is None:
      vals = values + delta
    else:
      transformed_vals = self.ret_ufun(0)(values) + delta
      vals = self.ret_ufun(1)(transformed_vals)
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
      return np.maximum(self._vlims[0], np.minimum(self._vlims[1], vals))

    # Bouncing scalars and arrays without and with boolean indexing respectively
    if isscalar(vals):
      if all(maybe_bounce):
        if not self._inside(vals):
          vals = values
      elif maybe_bounce[0]:
        if vals < self._vlims[0]:
          vals = values
        else:
          vals = np.minimum(self._vlims[1], vals)
      else:
        if vals > self._vlims[1]:
          vals = values
        else:
          vals = np.maximum(self._vlims[0], vals)
    else:
      if all(maybe_bounce):
        outside = np.logical_not(self._inside(vals))
        vals[outside] = values[outside]
      elif maybe_bounce[0]:
        outside = vals <= self._vlims[0]
        vals[outside] = values[outside]
        vals = np.minimum(self._vlims[1], vals)
      else:
        outside = vals >= self._vlims[1]
        vals[outside] = values[outside]
        vals = np.maximum(self._vlims[0], vals)
    return vals

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
