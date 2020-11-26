'''
A wrapper for expression functions that makes optional use of 'order' and 'delta' 
specifications. Multiple outputs are supported for non-symbolic outputs
'''
import collections
import functools
import sympy as sy
from probayes.vtypes import isscalar, issymbol
from probayes.prob import is_scipy_stats_dist, SCIPY_DIST_METHODS
SYMPY_EXPR = sy.Expr

#-------------------------------------------------------------------------------
class Expression (SYMPY_EXPR):
  """ A expression wrapper to enable object representations as an uncallable
  array, a callable function, or a tuple/dict of callable functions

  :example:
  >>> from probayes.expression import Expression
  >>> hw = Expression("Hello World!")
  >>> print(hw())
  Hello World!
  >>> inc = Expression(lambda x: x+1)
  >>> print(inc(2.))
  3.0
  >>> inc_dec = Expression( (lambda x:x+1, lambda x:x-1) )
  >>> print(inc_dec[0](3.))
  4.0
  >>> print(inc_dec[1](3.))
  2.0
  >>> sqr_sqrt = Expression( {'sqr': lambda x:x**2, 'sqrt': lambda x:x**0.5} )
  >>> print(sqr_sqrt['sqr'](3.))
  9.0
  >>> print(sqr_sqrt['sqrt'](4.))
  2.0

  Since multiple expressions are supported, the symbolic value of Expression
  is not related to the input.
  """

  # Protected
  _expr = None
  _args = None
  _kwds = None
  _keys = None
  _partials = None

  # Private
  __ismulti = None
  __isscalar = None
  __issymbol = None
  __scipyobj = None
  __callable = None
  __order = None
  __delta = None

#-------------------------------------------------------------------------------
  def __new__(cls, expr=None, *args, **kwds):
    return super(Expression, cls).__new__(cls, expr)

#-------------------------------------------------------------------------------
  def __init__(self, expr=None, *args, **kwds):
    """ Initialises instances according to object in expr, which may be an 
    uncallable object, a callable function, or a tuple of callable functions. 
    See set_expr()
    """
    self.set_expr(expr, *args, **kwds)
    
#-------------------------------------------------------------------------------
  def set_expr(self, expr=None, *args, **kwds):
    """ Set the Func instance's function object.

    :param expr: an uncallable object, callable function, or tuple of functions
    :param *args: arguments to pass onto callables
    :param **kwds: keywords to pass onto callables

    Note that the following two reserved keywords are disallowed:

    'order': which instead denotes a dictionary of remappings.
    'delta': which instead denotes a mapping of differences.
    """

    self._expr = expr
    self._args = tuple(args)
    self._kwds = dict(kwds)
    self.__order = None
    self.__delta = None
    self.__scipyobj = None
    self.__callable = None

    # Sanity check func
    if self._expr is None:
      assert not args and not kwds, "No optional args without a function"
    self.__issympy = issymbol(self._expr)
    self.__ismulti = isinstance(self._expr, (dict, tuple))
    if is_scipy_stats_dist(self._expr):
      self.__scipyobj = self._expr
      self.__ismulti = True
      self.__callable = True
      self.__issympy  = False
    elif self.__issympy:
      assert not args and not kwds, \
          "No optional arguments with symbolic expressions"
    elif not self.__ismulti:
      self.__callable = callable(self._expr)
      if not self.__callable:
        assert not args and not kwds, \
            "No optional arguments with uncallable expressions"
        self.__isscalar = isscalar(self._expr)
    else:
      exprs = self._expr if isinstance(self._expr, tuple) else \
              self._expr.values()
      self._callable = False
      self._isscalar = False
      self._issymbol = False
      each_callable = [callable(expr) for expr in exprs]
      each_isscalar = [isscalar(expr) for expr in exprs]
      each_issymbol = [issymbol(expr) for expr in exprs]
      assert len(set(each_callable)) < 2, \
          "Cannot mix callable and uncallable expressions"
      assert len(set(each_isscalar)) < 2, \
          "Cannot mix scalars and nonscalars"
      assert not any(each_issymbol), \
          "Symbolic expressions not supported for multiples"
      if len(self._expr):
        self.__callable = each_callable[0]
        self.__isscalar = each_isscalar[0]
        self.__issymbol = each_issymbol[0]
      if not self.__callable:
        assert not args and not kwds, "No optional args with uncallable function"
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    if 'delta' in self._kwds:
      self.set_delta(self._kwds.pop('delta'))
    self._set_partials()
    self._keys = list(self._partials.keys())

#-------------------------------------------------------------------------------
  def set_order(self, order=None):
    """ Sets an order remapping dictionary for functional calls in which
    keyword arguments are mapped to position (in numeric) or rekeyed (if str).
    """
    self.__order = order
    if self.__order is None:
      return
    assert self.__delta is None, "Cannot set both order and delta"
    assert not self.__symbol, "Cannot set order when using symbols"
    self._check_mapping(self.__order)

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None):
    """ Sets a difference remapping dictionary for functional calls in which
    keyword arguments are mapped to position (in numeric) or rekeyed (if str).
    """
    self.__delta = delta
    if self.__delta is None:
      return
    assert self.__order is None, "Cannot set both order and delta"
    assert not self.__symbol, "Cannot set delta when using symbols"
    self._check_mapping(self.__delta)

#-------------------------------------------------------------------------------
  def _check_mapping(self, mapping=None):
    """ Perform sanity checkings on mapping dictionary """
    if mapping is None:
      return
    # Used to sanity-check mapping dicts e.g. order and delta
    assert isinstance(mapping, dict), \
        "Mapping must be a dictionary type, not {}".format(type(mapping))
    key_list = list(mapping.keys())
    ind_list = list(mapping.values())
    keys = []
    inds = []
    for key, ind in zip(key_list, ind_list):
      keys.append(key)
      if type(ind) is int:
        inds.append(ind)
      elif ind is None:
        pass
      elif not isinstance(ind, str):
        raise TypeError("Cannot interpret index specification value: {}".ind)
    indset = set(inds)
    if len(indset):
      assert indset == set(range(min(indset), max(indset)+1)), \
          "Index specification non_sequitur: {}".format(indset)

#-------------------------------------------------------------------------------
  def _set_partials(self):
    # Private function to update partial function dictionary of calls
    self._partials = collections.OrderedDict()


    # Extract SciPy object member functions
    if self.__scipyobj:
      for method in SCIPY_DIST_METHODS:
        if hasattr(self.__scipyobj, method):
          call = functools.partial(Expression._partial_call, self, 
                                   getattr(self.__scipyobj, method),
                                   *self._args, **self._kwds)
          self._partials.update({method: call})

      # Provide a common interface for PDF/PMF and LOGPDF/LOGPMF
      if 'pdf' in self._partials.keys():
          self._partials.update({'prob': self._partials['pdf']})
          self._partials.update({'logp': self._partials['logpdf']})
      elif 'pmf' in self._partials.keys():
          self._partials.update({'prob': self._partials['pmf']})
          self._partials.update({'logp': self._partials['logpmf']})

    # Non-multiples are keyed by: None
    elif not self.__ismulti:
      call = functools.partial(Expression._partial_call, self, self._expr, 
                               *self._args, **self._kwds)
      self._partials.update({None: call})

    # Tuples are keyed by index
    elif isinstance(self._expr, tuple):
      for i, expr in enumerate(self._expr):
        call = functools.partial(Expression._partial_call, self, expr, 
                                 *self._args, **self._kwds)
        self._partials.update({i: call})

    # Dictionaries keys are mapped directly
    elif isinstance(self._expr, dict):
      for key, val in self._expr.items():
        call = functools.partial(Expression._partial_call, self, val, 
                                 *self._args, **self._kwds)
        self._partials.update({key: call})

#-------------------------------------------------------------------------------
  def ret_expr(self):
    """ Returns expression argument set by set_expr() """
    return self._expr

#-------------------------------------------------------------------------------
  def ret_args(self):
    """ Returns args as set by set_expr() """
    return self._args

#-------------------------------------------------------------------------------
  def ret_kwds(self):
    """ Returns kwds as set by set_expr() """
    return self._kwds

#-------------------------------------------------------------------------------
  def ret_scipyobj(self):
    """ Returns SciPy obj if set by set_expr() """
    return self.__scipyobj

#-------------------------------------------------------------------------------
  def ret_callable(self):
    """ Returns boolean flag as to whether expr is callable """
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    """ Returns boolean flag as to whether expr is a scalar """
    return self.__isscalar

#-------------------------------------------------------------------------------
  def ret_issymbol(self):
    """ Returns boolean flag as to whether expr is symbolic """
    return self.__issymbol

#-------------------------------------------------------------------------------
  def ret_ismulti(self):
    """ Returns boolean flag as to whether func comprises a multiple """
    return self.__ismulti

#-------------------------------------------------------------------------------
  def ret_order(self):
    """ Returns order object if set """
    return self.__order

#-------------------------------------------------------------------------------
  def ret_delta(self):
    """ Returns delta object if set """
    return self.__delta

#-------------------------------------------------------------------------------
  def _call(self, expr, *args, **kwds):
    """ Private call used by the wrapped Func interface that is _ismulti-blind.
    (see __call__ and __getitem__).
    """
    #argsin = args; kwdsin = kwds; import pdb; pdb.set_trace() # debugging

    # Non-callables
    if not self.__callable and not self.__issymbol:
      assert not args and not kwds, \
          "No optional args with uncallable or symbolic expressions"
      return expr 

    # Allow first argument to denote kwds
    if len(args) == 1 and isinstance(args[0], dict):
      args, kwds = (), {**kwds, **dict(args[0])}

    #argsmid = args; kwdsmid = kwds; import pdb; pdb.set_trace() # for debugging
    if not self.__issymbol or (not self.__order and not self.__delta):
      return expr(*args, **kwds)

    # Symbols are evaluated by substitution
    if self.__issymbol:
      subs = []
      for atom in expr.atoms:
        if hasattr(atom, name):
          key = atom.name
          assert key in kwds, \
              "Symbol name {} required as keyword input among".format(
                  key, kwds.keys())
          subs.append((atom, kwds[key],))
      return expr.subs(expr)

    # Append None to args according to mapping index specification
    n_args = len(args)
    mapping = self.__order or self.__delta
    for val in mapping.values():
      if type(val) is int:
        n_args = max(n_args, val+1)
    args = list(args)
    while len(args) < n_args:
      args.append(None)

    # Callables with order wrapper
    if self.__order:
      for key, val in self.__order.items():
        if type(val) is int:
          args[val] = kwds.pop(key)
        elif val is None:
          kwds.pop(key)
        elif isinstance(val, str):
          kwds.update({val: kwds.pop(key)})
        else:
          raise TypeError("Unrecognised order key: val type: {}:{}".\
                          format(key, val))
      return expr(*tuple(args), **kwds)

    # Callables with delta wrapper
    for key, val in self.__delta.items():
      if key[-1] != "'":
        value = kwds.pop(key)
      else:
        value = kwds.pop(key) - kwds.pop(key[:-1])
      if type(val) is int:
        args[val] = value
      elif val is None:
        pass
      elif isinstance(val, str):
        kwds.update({val: value})
      else:
        raise TypeError("Unrecognised delta key: val type: {}:{}".\
                        format(key, val))

    #argsout = args; kwdsout = kwds; import pdb; pdb.set_trace() # for debugging
    return expr(*tuple(args), **kwds)

#-------------------------------------------------------------------------------
  def _partial_call(self, *args, **kwds):
    return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   """ Wrapper call to the function with optional inclusion of additional
   args and kwds. """
   assert not self.__ismulti, \
       "Cannot call with multiple expr, use Expr[]()"
   return self._partials[None](*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, spec=None):
   r""" Returns the $i$th function from the expr tuple where if is $i$ is
   numeric, otherwise spec is treated as key returning the corresponding
   value in the expr dictionary. """
   if spec is not None: 
     assert self.__ismulti, \
         "Cannot call with non-multiple expr, use Expr()"
   return self._partials[spec]

#-------------------------------------------------------------------------------
  def __len__(self):
    """ Returns the number of expressions in the tuple set by set_expr() """
    if not self.__ismulti:
      return None
    return len(self._partials)

#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Print representation """
    if self.__issymbol or not self.__callable:
      return object.__repr__(self)+ ": '{}'".format(self._expr)
    if self._keys is None:
      return object.__repr__(self) 
    return object.__repr__(self)+ ": '" + self._keys + "'"

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
