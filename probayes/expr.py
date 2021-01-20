""" Expric representation wrapping SymPy's Expr. """

#-------------------------------------------------------------------------------
import collections
import sympy
import numpy as np
from sympy.utilities.autowrap import ufuncify
from probayes.variable_utils import parse_as_str_dict

DEFAULT_EFUNS = [
                  ({np.ndarray: ufuncify}, [], {})
                  ,
                  ({np.ndarray: sympy.lambdify}, ['numpy', 'scipy'], {})
                ]

#-------------------------------------------------------------------------------
class Expr:
  """ This class wraps sy.Expr. Sympy's dependence on __new__ to return
  modified class objects at instantiation doesn't play nicely with multiple
  inheritance wrap them in here as a class instead and copy over the attributes.

  :example
  >>> import sympy as sy
  >>> from probayes.expr import Expr
  >>> x = sy.Symbol('x')
  >>> x_inc = Expr(x + 1)
  >>> print(x_inc.subs({x:1}))
  2
  >>>
  """

  # Public
  _expr = None    # Expression object

  # Protected
  _symbols = None # Ordered dictionary of symbols keyed by name
  _efun = None    # Dictionary of evaluation functions
  _remap = None   # Optional dictionary for remapping

#-------------------------------------------------------------------------------
  def __init__(self, expr, **kwds):
    self.expr = expr
    self.remap = None if 'remap' not in kwds else kwds.pop('remap')

#-------------------------------------------------------------------------------
  @property
  def expr(self):
    return self._expr

  @property
  def symbols(self):
    return self._symbols

  @expr.setter
  def expr(self, expr=None):
    """ Sets the expr object for this instance. Either pass a sy.Expr object 
    directly or in accordance with the calling conventions for sy.Expr.__new__
    """
    self._expr = expr
    self._symbols = collections.OrderedDict()
    if isinstance(self._expr, sympy.Expr):
      for putative_symbol in self._expr.free_symbols:
        symbol = putative_symbol
        while hasattr(symbol, 'symbol') and \
            hasattr(symbol, 'name'):
          symbol = symbol.symbol
        if hasattr(symbol, 'name'):
          self._symbols.update({symbol.name: symbol})
    else:
      raise TypeError("Input type not Expr type but: {}".format(expr))

    # Make instance play nicely with Sympy by copying attributes and hash content
    members = dir(self._expr)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self._expr, member)
          setattr(self, member, attribute)
        except AttributeError:
          pass

    # In addition to substitution, add default evaluation functions
    for efun in DEFAULT_EFUNS:
      self.add_efun(efun[0], *tuple(efun[1]), **dict(efun[2]))

#-------------------------------------------------------------------------------
  @property
  def remap(self):
    return self._remap

  @remap.setter
  def remap(self, remap=None):
    self._remap = remap

#-------------------------------------------------------------------------------
  @property
  def efun(self):
    return self._efun

  def add_efun(self, efun=None, *args, **kwds):
    """ Adds an expression evaluation function.
    :param efun: a single dictionary entry in the form {type: function}
    :param args: optional args to pass to efun
    :param kwds: optional kwds to pass to efun

    :returns: updated efun dictionary.
    """
    if self._efun is None:
      self._efun = {None: (self._expr.subs)}
    if efun is None or not self._symbols:
      return
    symbols = tuple(self._symbols.values())
    assert isinstance(efun, dict), \
        "Input efun must be dictionary type, not {}".format(type(efun))
    for key, val in efun.items():
      if key not in self._efun:
        try:
          self._efun.update({key: val(symbols, self._expr, *args, **kwds)})
        except (TypeError, ImportError, AttributeError): # Bypass sympy bugs
          pass
    return self._efun

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Performs evaluation of expression inputting symbols as a dictionary 
    (see sympy.Symbol.subs() input specificion using dictionaries. """
    parse_values = True
    if not(len(kwds)) and len(args) == len(self._symbols):
      if not any([isinstance(arg, dict) for arg in args]):
        values = collections.OrderedDict()
        parse_values = False
        for i, key in enumerate(self._symbols.keys()):
          values.update({key: args[i]})
    if parse_values:
      if self._remap:
        kwds = dict(kwds)
        kwds.update({'remap': self.remap})
      values = parse_as_str_dict(*args, **kwds)

    # Check values in symbols
    keys = list(values.keys())
    for key in keys:
      if key not in self._symbols.keys():
        __key__ = "__{}__".format(key) # Convention for invertibles
        if __key__ not in self._symbols.keys():
          raise KeyError("Key {} not found among symbols {}".format(
            key, self._symbols.keys()))
        values[__key__] = values.pop(key)

    # While determining type, collect evalues in required order
    etype = None
    evalues = [None] * len(self._symbols)
    single_numpy_key = None
    for i, key in enumerate(self._symbols.keys()):
      evalues[i] = values[key]
      _etype = type(evalues[i])
      if _etype is np.ndarray:
        if single_numpy_key:
          single_numpy_key = False
        elif single_numpy_key is None:
          single_numpy_key = key
      if etype is None: 
        if _etype in self._efun.keys():
          etype = _etype
      elif _etype in self._efun.keys():
        assert etype == _etype, \
            "Cannot mix types {} vs {} ".format(etype, _etype)

    # Support iterating a single-array even without a NumPy eval func
    if single_numpy_key and np.ndarray not in self._efun:
      key = single_numpy_key
      efun = self._efun[None]
      evalues = dict(values)
      shape = values[key].shape
      revalues = np.ravel(values[key])
      reval = [None] * len(revalues)
      for i, revalue in enumerate(revalues):
        evalues.update({key: revalue})
        val = efun(evalues)
        if isinstance(val, sympy.Integer):
          reval[i] = int(val)
        else:
          reval[i] = float(val)
      return np.array(reval).reshape(shape)

    # Otherwise rely on evaluation function
    efun = self._efun[etype]
    vals = efun(*evalues) if etype else efun(values)
    if isinstance(vals, sympy.Integer):
      return int(vals)
    elif isinstance(vals, sympy.Float):
      return float(vals)
    if etype != np.ndarray or isinstance(vals, np.ndarray):
      return vals
    return np.array(vals)

#-------------------------------------------------------------------------------
  def __repr__(self):
    if self._expr is None:
      return super().__repr__()
    return self._expr.__repr__()

#-------------------------------------------------------------------------------
  def __inv__(self):
    """ Overload bitwise inverter operator to return symbol. """
    return self.symbol

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#------------------------------------------------------------------------------
