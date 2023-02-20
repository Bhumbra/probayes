"""
A variable is a representation of a quantity with a variable set that 
defines a domain over which a function is valid. It therefore needs a name, 
variable type, and variable set. 
"""

#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.icon import Icon, isiconic
from probayes.vtypes import eval_vtype, isunitset, isscalar, \
                        revtype, uniform, VTYPES, OO, OO_TO_NP
from probayes.variable_utils import parse_as_str_dict
from probayes.pscales import log_prob
from probayes.expression import Expression
from probayes.distribution import Distribution
from probayes.ops import and_op, or_op

# Defaults
DEFAULT_VNAME = 'var'
DEFAULT_VTYPE = bool

# Defult vsets by vtype
DEFAULT_VSETS = {bool: [False, True],
                  int: [0, 1], 
                float: [(-OO,), (OO,)]}

#-------------------------------------------------------------------------------
class Variable (Icon):
  """ A variable is a representation of a quantity with a variable set that 
  defines a domain over which a function is valid. It therefore needs a name, 
  variable type, and variable set. 
  
  While this class does not support defining probability density functions 
  (use RV for that), it does include an option to specify a univariate
  function that can be used for example to specify a monotonic variable 
  transformation:

  :example:
  >>> import numpy as np
  >>> import probayes as pb
  >>> x = pb.Variable('x', vtype=float)
  >>> x.set_ufun((np.exp, np.log))
  >>> print(x.ufun[0](1))
  2.718281828459045
  >>> print(x.ufun[-1](1))
  0.0
  >>> print(x.vset)
  [(-oo,), (oo,)]
  >>> print(x.vlims)
  [-inf  inf]
  >>> print(x.ulims)
  [ 0. inf]
  """
                    
  # Protected       
  _Delta = None      # A namedtuple generator for delta operations
  _vtype = None      # Variable type (bool, int, or float)
  _vset = None       # Variable set (array or 2-length tuple range)
  _vlims = None      # Numpy array of bounds of vset
  _ufun = None       # Univariate function for variable transformation
  _ulims = None      # self._vlims if not no_ucov else transformed self._vlims
  _isfinite = None   # all(isfinite(ulims))
  _length = None     # Difference in self._ulims
  _lhv = None        # Log hypervolume
  _nlhv = None       # Negative log hypervolume - always of transformed space
  _inside = None     # Lambda function for defining inside vset
  _delta = None      # Default delta operation
  _delta_args = None # Optional delta arguments 
  _delta_kwds = None # Optional delta keywords 
  _is_stochastic = False # Flag of whether stochastic
  _dexpr = None      # Deterministic expression

  # Private       
  __no_ucov = None   # Boolean flag to denote no univariate change of variables

#-------------------------------------------------------------------------------
  @property
  def is_stochastic(self):
    return self._is_stochastic

  def __init__(self, name=None,
                     vset=None, 
                     vtype=None,
                     *args,
                     **kwds):
    """ Initialiser sets name, vset, and ufun:

    :param name: Name of the variable - string as valid identifier.
    :param vset: variable set over which variable domain defined.
    :param vtype: variable type (bool, int, or float).
    :param *args: optional arguments to pass onto symbol representation.
    :param **kwds: optional keywords to pass onto symbol representation.

    Every Variable instance offers a factory function for delta specifications:

    :example:
    >>> import numpy as np
    >>> import probayes as pb
    >>> x = pb.Variable('x', vtype=float)
    >>> dx = x.Delta(0.5)
    >>> print(x.apply_delta(1.5, dx))
    2.0
    """
    self.name = name

    # Allow positions of vtype and vset to be swapped
    if isinstance(vtype, (list, tuple, set, range, np.ndarray)):
      vtype, vset = vset, vtype

    # If vtype and/or vset is/are specified, set it/them
    if vtype or vset:
      if vtype:
        self.vtype = vtype
      if vset:
        self.vset = vset
    else:
      self.vtype = vtype

    # Setting icon comes afterwards to allow passing vtype/vset assumptions
    self.set_icon(self.name, *args, **kwds)

    # Default delta to nothing
    self.set_delta()

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @property
  def Delta(self):
    return self._Delta

  @name.setter
  def name(self, name=DEFAULT_VNAME):
    self._name = name
    self._Delta = collections.namedtuple('ð', [self._name])


#-------------------------------------------------------------------------------
  @property
  def vset(self):
    """ Property vset is the set over which the valid values are defined.
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
    >>> x = pb.Variable('x', vtype=float, vset=(1,2))
    >>> print(x.vset)
    [(1.0,), (2.0,)]
    """
    return self._vset

  @vset.setter
  def vset(self, vset=None):

    # Default vset to nominal
    if vset is None and self._vtype: 
      vset = DEFAULT_VSETS[self._vtype]
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
        if not vtype:
          vtype = eval_vtype(value)
        elif not self._vtype and vtype != eval_vtype(value):
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
  @property
  def vtype(self):
    """ Property vtype is the variable type (default bool). If the variable set 
    if not set, then it is defaulted according to the variable type. """
    return self._vtype

  @vtype.setter
  def vtype(self, vtype=None):
    """ Sets the variable type (default bool). If the variable set if not set,
    then it is defaulted according to the variable type. """

    if not vtype:
      if self._vset is not None:
        self._vtype = eval_vtype(self._vset)
      else:
        self._vtype = DEFAULT_VTYPE
    else:
      self._vtype = eval_vtype(vtype)
    if self._vset is None:
      self.vset = DEFAULT_VSETS[self._vtype]

#-------------------------------------------------------------------------------
  @property
  def vlims(self):
    """ The untransformed (self._vlims) limits of the variable. """
    return self._vlims

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
      return self._eval_ulims()

    # Evaluates the limits from vset float
    assert len(self._vset) == 2, \
        "Floating point vset must be two elements, not {}".format(self._vset)
    lims = [None] * 2
    for i, limit in enumerate(self._vset):
        lim = limit if not isinstance(limit, tuple) else list(limit)[0]
        lims[i] = OO_TO_NP.get(lim, lim)

    # Re-order vset if limits are not sorted
    if lims[1] < lims[0]:
      vset = self._vset[::-1]
      self._vset = vset
    self._vlims = np.sort(lims)
    return self._eval_ulims()

#-------------------------------------------------------------------------------
  @property
  def ulims(self):
    """ The transformed (self._vlims) limits of the variable. """
    return self._ulims

  @property
  def inside(self):
    """ A lambda function for the whether an input is within the vset """
    return self._inside

  @property
  def isfinite(self):
    """ Boolean flag denoting whether variable range is finite """
    return self._isfinite

  @property
  def length(self):
    """ Length of the variable according to its (transformed) limits """
    return self._length

  @property
  def lhv(self):
    """ Log hypervolume (=log(length)) of the variable """
    return self._length

  def __len__(self):
    """ Length of the variable according to its (transformed) limits """
    return self._length

  def __contains__(self, *args, **kwds):
    if self._inside:
      return self._inside(*args, **kwds)
    return None

  def _eval_ulims(self):
    """ Evaluates transformed limits (self._ulims) and inside functions
    
    :returns: the length of the variable.
    """
    self._ulims = None
    self._length = None
    self._lhv = None
    self._inside = None
    self._isfinite = None
    if self._vlims is None:
      return self._length

    # Non-floats do not support transformation
    if self._vtype not in VTYPES[float]:
      self._inside = lambda x: np.isin(x, self._vset, assume_unique=True)
      self._ulims = self._vlims
      self._length = 2 if self._vtype in VTYPES[bool] else len(self._vset)
      self._lhv = log_prob(self._length)
      self._nlhv = -self._lhv
      self._isfinite = False
      return self._length

    # Floating point limits are subject to transformation
    self._ulims = self._vlims if self._ufun is None else \
                  self.ufun[0](self._vlims)
    self._isfinite = np.all(np.isfinite(self._ulims))
    self._length = max(self._ulims) - min(self._ulims)
    self._lhv = log_prob(self._length) if self._isfinite else np.inf
    self._nlhv = -self._lhv # This value serves as the coefficient for no_ucov

    # If not change of variables, restore all lims-related members except nlhv
    if self.__no_ucov:
      self._ulims = self._vlims
      self._isfinite = np.all(np.isfinite(self._ulims))
      self._length = max(self._ulims) - min(self._ulims)
      self._lhv = log_prob(self._length) if self._isfinite else np.inf

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
  @property
  def delta(self):
    """ Returns the default delta object if specified """
    return self._delta

  def set_delta(self, delta=None, *args, **kwds):
    """ Sets the default delta operation for the domain.

    :param delta: a callable or uncallable argument (see below)
    :param *args: args to pass if delta is callable.
    :param **kwds: kwds to pass if delta is callable (except scale and bound)

    The first argument delta may be:

    1. A callable function (operating on the first term).
    2. A Variable.Delta instance (this defaults all Variable deltas).
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
      self._delta = Expression(self._delta, *args, **kwds)
      return

    # Default scale and bound
    if 'scale' not in self._delta_kwds:
      self._delta_kwds.update({'scale': False})
    if 'bound' not in self._delta_kwds:
      self._delta_kwds.update({'bound': False})

#-------------------------------------------------------------------------------
  @property
  def icon(self):
    return self._icon

  def set_icon(self, icon=None, *args, **kwds):
    """ Sets the variable symbol and carries members over to this Variable """
    if icon is None:
      icon = self._name
    """
    # If no arguments passed, default assumptions based on vtype and limits
    - unfortunately this messes up with subs(...)
    if isinstance(icon, str) and not args and not kwds:
      kwds = dict(**kwds)
      if self._vtype in VTYPES[float]:
        kwds.update({'integer': False, 'real': True})
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
    """
    return Icon.set_icon(self, icon, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def ufun(self):
    return self._ufun

  @property
  def no_ucov(self):
    """ Flag to denote no univariate change of variables for iconic ufuns. """
    return self.__no_ucov


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
    self.__no_ucov = False if 'no_ucov' not in kwds else kwds.pop('no_ucov')

    # Non-iconic ufun inputs must a two-length-tuple
    if not isiconic(self._ufun):
      assert self._vtype in VTYPES[float], \
          "Values transformation function only supported for floating point"
      message = "Non-iconic input ufun be a two-sized tuple of callable functions"
      assert isinstance(self._ufun, tuple), message
      assert len(self._ufun) == 2, message
      assert callable(self._ufun[0]), message
      assert callable(self._ufun[1]), message
    elif 'invertible' not in kwds:
      kwds.update({'invertible': True})
    self._ufun = Expression(self._ufun, *args, **kwds)
    if self.__no_ucov:
      assert self._ufun.isiconic, \
          "Can only set no_ucon=True for iconic univariate functions"
    self._eval_ulims()

#-------------------------------------------------------------------------------
  def evaluate(self, values=None):
    r""" Evaluates value(s) belonging to the variable.

    :param values: None, set of a single integer, array, or scalar.

    :return: a distribution dictionary {key: val} in which key is the variable 
             name and val is a NumPy array of values in accordance to the 
             following:

    If values is a NumPy array, it is returned unchanged.

    If values is None, it defaults to the entire variable set (vset) if not
    the variable type vtype is not float; otherwise a single scalar within the
    vset is randomly evaluated (see below for $n=0$).

    If values is a set containing a single integer (i.e. $\{n\}$), then the 
    output depends on the number $n$:

    If positive ($n$), then $n$ values are uniformly sampled.
    If zero ($n=0$), then a scalar value is randomly sampled.
    if negative($-n$), then $n$ values are randomly sampled.

    For non-float types, the values are evaluated from vset according to $n$:

    If positive ($n$), then $n$ values are serially sampled from ordered vset.
    If zero ($n=0$), then a scalar value is randomly sampled.
    if negative($-n$), then $n$ values are sampled from random vset permutations.
    
    For float types, then any uniformly sampled is performed in accordance of 
    any transformations set by Variable.set_ufun(), except for iconic ufun
    specifications with no change of variables (i.e. no_ucov=False).

    :example:
    >>> import numpy as np
    >>> import probayes as pb
    >>> freq = pb.Variable('freq', vtype=float, vset=[1., 8.])
    >>> freq.set_ufun((np.log, np.exp))
    >>> print(freq.evaluate({4}))
    freq: Distribution([('freq', array([1., 2., 4., 8.]))])
    """
    """ Evaluates the values ordered dictionary for __call__ """


    # If dictionary input type, values are keyed by variable name
    values = parse_as_str_dict(values) if isinstance(values, dict) \
             else {self.name: values}
    if isinstance(values, dict):
      values = values[self.name] 

    # Default to arrays of complete sets
    if values is None:
      if self._vtype in VTYPES[float]:
        values = {0}
      else:
        return Distribution(self._name, {self.name: 
                   np.array(list(self._vset), dtype=self._vtype)})

    # Sets may be used to sample from support sets
    if isunitset(values):
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
            indices = np.random.permutation(-number) % self._length
          values = values[indices]
        return Distribution(self._name, {self.name: values})
       
      # Continuous
      else:
        assert self._isfinite, \
            "Cannot evaluate {} values for bounds: {}".format(
                values, self._ulims)
        values = uniform(self._ulims[0], self._ulims[1], number, 
                           isinstance(self._vset[0], tuple), 
                           isinstance(self._vset[1], tuple)
                        )

      # Only use ufun when isunitsetint(values)
      if self._ufun and not self.__no_ucov:
        return Distribution(self._name, {self.name: self.ufun[-1](values)})
    return Distribution(self._name, {self.name: values})

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ See Variable.evaluate() """
    return self.evaluate(values)

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
    if isinstance(delta, Expression):
      if delta.ret_callable():
        return delta
      delta = delta()
    if isinstance(delta, self._Delta):
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
    return self._Delta(delta)

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

    # If dictionary input type, values are keyed by variable name
    if isinstance(values, dict):
      values = values[self.name] 

    # Call eval_delta() if values is a list and return values if delta is None
    delta = delta or self._delta
    if isinstance(delta, Expression):
      if delta.ret_callable():
        return delta(values)
      delta = delta()
    elif self._vtype not in VTYPES[bool]:
      if isinstance(delta, (list, tuple)):
        delta = self.eval_delta(delta)
    if isinstance(delta, self._Delta):
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
    elif self._ufun is None or self.__no_ucov:
      vals = values + delta
    else:
      transformed_vals = self.ufun[0](values) + delta
      vals = self.ufun[1](transformed_vals)
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
  def inverse(self):
    """ Returns the icon unless an invertible Sympy ufun has been set, in which
    case the corresponding inverse icon is returned.
    """
    if self._ufun and self._ufun.inverse is not None:
      return self._ufun.inverse
    return self._icon

#-------------------------------------------------------------------------------
  def __invert__(self):
    """ See Variable.inverse() """
    return self.inverse()

#-------------------------------------------------------------------------------
  @property
  def dexpr(self):
    """ Sets iconic deterministic expression, substituting for icon in [:] """
    return self._dexpr

  @name.setter
  def dexpr(self, dexpr=None):
    self._dexpr = None if self._is_stochastic else dexpr

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
    """ Method [:] overloaded to return icon or dexpr object. """
    assert arg == slice(None), \
        "Only ':' input accepted for __getitem__ method  Icon[:], not: {}".\
        format(arg)
    return self._dexpr or self._icon

#-------------------------------------------------------------------------------
  def __and__(self, other):
    """ Combination operator between Variable and another Variable, Field, or
    Dependence. """
    return and_op(self, other)

#-------------------------------------------------------------------------------
  def __or__(self, other):
    """ Combination operator between Variable and another Variable, Field, or
    Dependence. """
    return or_op(self, other)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def variables(names, vtype=None, vset=None, *args, **kwds):
  if isinstance(names, str):
    names = names.split(',')
  varlist = [Variable(name, vtype, vset, *args, **kwds) for name in names]
  return tuple(varlist)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
