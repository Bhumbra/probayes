'''
A wrapper for expression functions that makes optional use of 'order' and 'delta' 
specifications. Multiple outputs are supported for non-symbolic expressions. Note
that an expression may comprise a constant by not being callable, and therefore 
not all expressions are functions.
'''
import collections
import functools
import sympy as sy
from probayes.vtypes import isscalar
from probayes.icon import isiconic
from probayes.expr import Expr

#-------------------------------------------------------------------------------
class Expression (Expr):
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
  _args = None
  _kwds = None
  _keys = None
  _partials = None
  _ismulti = None
  _callable = None

  # Private
  __invertible = None
  __inverse = None
  __invexpr = None
  __isscalar = None
  __isiconic = None
  __order = None
  __delta = None

#-------------------------------------------------------------------------------
  def __init__(self, expr=None, *args, **kwds):
    """ Initialises instances according to object in expr, which may be an 
    uncallable object, a callable function, or a tuple of callable functions. 
    See set_expr()
    """
    self.set_expr(expr, *args, **kwds)
    
#-------------------------------------------------------------------------------
  @property
  def args(self):
    return self._args

  @property
  def kwds(self):
    return self._kwds

  @property
  def order(self):
    return self.__order

  @property
  def delta(self):
    return self.__delta

  @property
  def callable(self):
    return self._callable

  @property
  def ismulti(self):
    return self._ismulti

  @property
  def isscalar(self):
    return self.__isscalar

  @property
  def isiconic(self):
    return self.__isiconic

  @property
  def inverse(self):
    return self.__inverse

  @property
  def invexpr(self):
    return self.__invexpr

  @property
  def invertible(self):
    return self.__invertible

  @property
  def keys(self):
    return self._keys

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
    self._callable = None
    self._ismulti = None
    self.__order = None
    self.__delta = None
    self.__isscalar = None
    self.__isiconic = None
    self.__inverse = None
    self.__invexpr = None
    self.__invertible = False if 'invertible' not in kwds else \
                        self._kwds.pop('invertible')

    # Sanity check func
    if self._expr is None:
      assert not args and not kwds, "No optional args without a function"
    self.__isiconic = isiconic(self._expr)
    self._ismulti = isinstance(self._expr, (dict, tuple))

    # Icon
    if self.__isiconic:
      self.expr = expr # this invokes the inherited setter
      self._ismulti = self.__invertible and len(self._symbols) == 1
      self._callable = True

    # Unitary
    elif not self._ismulti:
      self._callable = callable(self.expr)
      if not self._callable:
        assert not args and not kwds, \
            "No optional arguments with uncallable expressions"
        self.__isscalar = isscalar(self.expr)

    # Multi
    else:
      exprs = self.expr if isinstance(self.expr, tuple) else \
              self.expr.values()
      self._callable = False
      self.__isscalar = False
      self.__isiconic = False
      each_callable = [callable(expr) for expr in exprs]
      each_isscalar = [isscalar(expr) for expr in exprs]
      each_isiconic = [isiconic(expr) for expr in exprs]
      assert len(set(each_callable)) < 2, \
          "Cannot mix callable and uncallable expressions"
      assert len(set(each_isscalar)) < 2, \
          "Cannot mix scalars and nonscalars"
      assert not any(each_isiconic), \
          "Symbolic expressions not supported for multiples"
      if len(self.expr):
        self._callable = each_callable[0]
        self.__isscalar = each_isscalar[0]
        self.__isiconic = each_isiconic[0]
      if not self._callable:
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
    # Protected function to update partial function dictionary of calls
    self._partials = collections.OrderedDict()


    # Evaluate symbolic call if symbol expression
    if self.__isiconic:
      call = functools.partial(Expr.__call__, self)
      self._partials.update({None: call})

      # If invertible, solve the expression
      if self.__invertible:
        self._partials.update({0: call})
        key = list(self._symbols.keys())[0]
        __key__ = "__{}__".format(key)
        self.__inverse = sy.Symbol(__key__)
        invexprs = sy.solve(self._expr - self.__inverse, self._symbols[key])
        n_exprs = len(invexprs)
        assert n_exprs, "No invertible solutions for expression {}".format(
            self._expr)
        invexpr = invexprs[0]
        if n_exprs > 1:
          for expr in invexprs[1:]:
            if len(expr.__repr__()) < len(invexpr.__repr__()):
              invexpr = expr
        self.__invexpr = Expr(invexpr)
        call = functools.partial(Expr.__call__, self.__invexpr)
        self._partials.update({-1: call})

    # Non-multiples are keyed by: None
    elif not self._ismulti:
      call = functools.partial(Expression._partial_call, self, self.expr, 
                               *self._args, **self._kwds)
      self._partials.update({None: call})

    # Tuples are keyed by index
    elif isinstance(self.expr, tuple):
      for i, expr in enumerate(self.expr):
        call = functools.partial(Expression._partial_call, self, expr, 
                                 *self._args, **self._kwds)
        self._partials.update({i: call})
      self._partials.update({-1: call})

    # Dictionaries keys are mapped directly
    elif isinstance(self.expr, dict):
      for key, val in self.expr.items():
        call = functools.partial(Expression._partial_call, self, val, 
                                 *self._args, **self._kwds)
        self._partials.update({key: call})

#-------------------------------------------------------------------------------
  def _call(self, expr, *args, **kwds):
    """ Private call used by the wrapped Func interface that is _ismulti-blind.
    (see __call__ and __getitem__).
    """
    #argsin = args; kwdsin = kwds; import pdb; pdb.set_trace() # debugging

    # Non-callables
    if not self._callable and not self.__isiconic:
      assert not args and not kwds, \
          "No optional args with uncallable or symbolic expressions"
      return expr 

    # Allow first argument to denote kwds
    if len(args) == 1 and isinstance(args[0], dict):
      args, kwds = (), {**kwds, **dict(args[0])}

    #argsmid = args; kwdsmid = kwds; import pdb; pdb.set_trace() # debugging
    if not self.__isiconic or (not self.__order and not self.__delta):
      return expr(*args, **kwds)

    # Symbols are evaluated by substitution
    if self.__isiconic:
      subs = []
      for atom in expr.atoms:
        if hasattr(atom, 'name'):
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

    #argsout = args; kwdsout = kwds; import pdb; pdb.set_trace() # debugging
    return expr(*tuple(args), **kwds)

#-------------------------------------------------------------------------------
  def _partial_call(self, *args, **kwds):
    return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   """ Wrapper call to the function with optional inclusion of additional
   args and kwds. """
   assert not self._ismulti, \
       "Cannot call with multiple expression, use Expression[{}]()".format(
           list(self._keys()))
   return self._partials[None](*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, spec=None):
   r""" Returns the $i$th function from the expr tuple where if is $i$ is
   numeric, otherwise spec is treated as key returning the corresponding
   value in the expr dictionary. """
   if spec is not None: 
     assert self._ismulti, \
         "Cannot call with non-multiple expression, use Expression()"
   return self._partials[spec]

#-------------------------------------------------------------------------------
  def __len__(self):
    """ Returns the number of expressions in the tuple set by set_expr() """
    if not self._ismulti:
      return None
    return len(self._partials)

#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Print representation """
    if self.__isiconic or not self._callable:
      return object.__repr__(self)+ ": '{}'".format(self.expr)
    if self._keys is None:
      return object.__repr__(self) 
    return object.__repr__(self)+ ": '{}'".format(self._keys)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
