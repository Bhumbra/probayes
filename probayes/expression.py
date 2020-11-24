'''
A wrapper for expression functions that makes optional use of 'order' and 'delta' 
specifications.
'''
import sympy as sy
from probayes.vtypes import isscalar
SYMPY_EXPR = sy.Expr

#-------------------------------------------------------------------------------
class Expression (SYMPY_EXPR):
  """ A expression wrapper to enable object representations as an uncallable
  array, a callable function, or a tuple of callable functions

  :example:
  >>> from probayes.expr import Expr
  >>> hw = Expr("Hello world!")
  >>> print(hw())
  Hello World!
  >>> inc = Expr(lambda x: x+1)
  >>> print(inc(2.)
  3.0
  >>> inc_dec = Expr( (lambda x:x+1, lambda x:x-1) )
  >>> print(inc_dec[0](3.))
  4.0
  >>> print(inc_dec[1](3.))
  2.0
  """

  # Protected
  _expr = None
  _args = None
  _kwds = None

  # Private
  __ismulti = None
  __isscalar = None
  __isscipy = None
  __callable = None
  __order = None
  __delta = None
  __index = None

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
    self.__callable = None

    # Sanity check func
    if self._expr is None:
      assert not args and not kwds, "No optional args without a function"
    self.__ismulti = isinstance(self._expr, tuple)
    self.__isscalar = False
    if not self.__ismulti:
      self.__callable = callable(self._expr)
      if not self.__callable:
        assert not args and not kwds, "No optional args with uncallable function"
        self.__isscalar = isscalar(self._expr)
    else:
      each_callable = [callable(expr) for expr in self._expr]
      each_isscalar = [isscalar(expr) for expr in self._expr]
      assert len(set(each_callable)) < 2, \
          "Cannot mix callable and uncallable functions"
      assert len(set(each_isscalar)) < 2, \
          "Cannot mix scalars and nonscalars"
      if len(each_callable):
        self.__callable = each_callable[0]
        self.__isscalar = each_isscalar[0]
      if not self.__callable:
        assert not args and not kwds, "No optional args with uncallable function"
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    if 'delta' in self._kwds:
      self.set_delta(self._kwds.pop('delta'))

#-------------------------------------------------------------------------------
  def set_order(self, order=None):
    """ Sets an order remapping dictionary for functional calls in which
    keyword arguments are mapped to position (in numeric) or rekeyed (if str).
    """
    self.__order = order
    if self.__order is None:
      return
    assert self.__delta is None, "Cannot set both order and delta"
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
  def ret_callable(self):
    """ Returns boolean flag as to whether func is callable """
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    """ Returns boolean flag as to whether func is a scalar """
    return self.__isscalar

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
  def _call(self, *args, **kwds):
    """ Private call used by the wrapped Func interface.
    (see __call__ and __getitem__).
    """

    """
    # For debugging:
    argsin = args
    kwdsin = kwds
    """
    # Check for indexing and reset if necessary
    expr = self._expr
    if self.__index is not None:
      expr = expr[self.__index]
      self.__index = None

    # Non-callables
    if not self.__callable:
      assert not args and not kwds, "No optional args with uncallable function"
      return func

    # Callables order-free
    if len(args) == 1 and isinstance(args[0], dict):
      args, kwds = (), {**kwds, **dict(args[0])}
    if self._args:
      args = tuple(list(self._args) + list(args))
    if self._kwds:
      kwds = {**kwds, **self._kwds}
    if not self.__order and not self.__delta:

      """
      # For debugging:
      argsout = args
      kwdsout = kwds
      import pdb; pdb.set_trace()
      """

      return expr(*args, **kwds)

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
    return expr(*tuple(args), **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   """ Wrapper call to the function with optional inclusion of additional
   args and kwds. """

   assert not self.__ismulti, \
       "Cannot call with multiple expr, use Expr[]"
   return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, spec=None):
   r""" Calls the $i$th function from the Func tuple where the spec is $i$ """
   if spec is None:
     return self._expr
   assert self.__ismulti, \
     "Cannot index without single func, use Expr()"
   self.__index = spec
   return self._call

#-------------------------------------------------------------------------------
  def __len__(self):
    """ Returns the number of expressions in the tuple set by set_expr() """
    if not self.__ismulti:
      return None
    return len(self._expr)

#-------------------------------------------------------------------------------
