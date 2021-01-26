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
def collate_symbols(expr):
  """ Collates symbols from expression expr as an ordered dictionary """
  assert isinstance(expr, sympy.Expr), \
    "Input type not Expr type but: {}".format(expr)
  symbols = collections.OrderedDict()
  for putative_symbol in expr.free_symbols:
    symbol = putative_symbol
    while hasattr(symbol, 'symbol') and \
        hasattr(symbol, 'name'):
      symbol = symbol.symbol
    if hasattr(symbol, 'name'):
      symbols.update({symbol.name: symbol})
  return symbols

#-------------------------------------------------------------------------------
def sort_ordered_dict(ordereddict, order, reverse=False):
  """ Resorts a an ordered directionary by order (Nones are ignored) """
  assert isinstance(ordereddict, collections.OrderedDict), \
      "First input must be an ordered dictionary"
  keys = list(ordereddict.keys())
  if isinstance(order, (tuple, list)):
    assert len(ordereddict) == len(order), \
        "Inputs not commensurate"
    order = {key: val for key, val in zip(keys, order)}
  else:
    assert set(keys) == set(order.keys()), \
        "Mismatch between keys and order: {} vs {}".format(keys, order.keys())
  not_none_keys = [key for key in keys if order[key] is not None]
  if len(not_none_keys) < 2:
    return ordereddict
  not_none_vals = [order[key] for key in not_none_keys]
  sort_order = np.argsort(not_none_vals)
  sort_order = sort_order.tolist() if not reverse else \
               sort_order[::-1].tolist()
  if np.min(np.diff(sort_order)) > 0:
    return ordereddict
  sorted_keys = [not_none_keys[index] for index in sort_order]
  index = -1 
  sorted_dict = collections.OrderedDict()
  for i, key in enumerate(keys):
    if key not in not_none_keys:
      sorted_dict.update({key: ordereddict[key]})
    else:
      index += 1
      sorted_key = sorted_keys[index]
      sorted_dict.update({sorted_key: ordereddict[sorted_key]})
  return sorted_dict

#-------------------------------------------------------------------------------
class Expr:
  """ This class wraps sy.Expr. Sympy's dependence on __new__ to return
  modified class objects at instantiation doesn't play nicely with multiple
  inheritance wrap them in here as a class instead and copy over the attributes.

  :example
  >>> import sympy as sy
  >>> from probayes.expr import Expr
  >>> x = sy.Symbol('x')
  >>> x_inc = Expr(x + 1) >>> print(x_inc.subs({x:1}))
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
    self._symbols = collate_symbols(self._expr)

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

    # Convert bools to ints (Sympy doesn't like bool type)
    keys = list(values.keys())
    for key in keys:
      if isinstance(values[key], bool):
        values[key] = int(values[key])

    # While determining type, collect evalues in required order
    etype = None
    evalues = [None] * len(keys)
    isarray = [None] * len(keys)
    vec_dim = [None] * len(keys)
    found_multidimensional = False
    for i, key in enumerate(keys):
      evalues[i] = values[key]
      _etype = type(evalues[i])
      isarray[i] = _etype is np.ndarray
      if isarray[i]:
        non_sing = np.array(evalues[i].shape) > 1
        non_sing_sum = non_sing.sum()
        if non_sing_sum > 0:
          if non_sing_sum > 1:
            found_multidimensional = True
          vec_dim[i] = int(np.nonzero(non_sing)[0][0])
      if etype is None: 
        if _etype in self._efun.keys():
          etype = _etype
      elif _etype in self._efun.keys():
        assert etype == _etype, \
            "Cannot mix types {} vs {} ".format(etype, _etype)

    # Support iterating numpy-array even without a NumPy eval func
    if any(isarray) and np.ndarray not in self._efun:
      efun = self._efun[None]

      vec_dims = np.array([dim for dim in vec_dim if dim is not None])
      # Is a sort required (meshgrid seems to prefer reverse order)?
      if len(vec_dims) > 1 and np.max(np.diff(vec_dims)) > 0:
        values = sort_ordered_dict(values, vec_dim, reverse=True)
        evalues = list(values.values())
      mesh_values = [val for i, val in enumerate(evalues) if isarray[i]]
      mesh_grid = np.meshgrid(*tuple(mesh_values), copy=False)
      revalues = collections.OrderedDict()
      shape = None
      mesh_idx = -1
      for i, key in enumerate(keys):
        if isarray[i]:
          mesh_idx += 1
          shape = mesh_grid[mesh_idx].shape
          revalues.update({key: np.ravel(mesh_grid[mesh_idx])})
        else:
          revalues.update({key: values[key]})
      size = int(np.prod(shape))
      vals = dict(values)
      reval = [None] * size
      for k in range(size):
        for i, key in enumerate(keys):
          if isarray[i]:
            vals.update({key: revalues[key][k]})
        val = efun(vals)
        if isinstance(val, sympy.Integer):
          reval[k] = int(val)
        else:
          reval[k] = float(val)
      return np.array(reval).reshape(shape)

    # Otherwise rely on evaluation function
    efun = self._efun[etype]
    vals = efun(*evalues) if etype else efun(values)
    if isinstance(vals, np.ndarray):
      return vals
    elif isinstance(vals, sympy.Integer):
      return int(vals)
    elif isinstance(vals, sympy.Float):
      return float(vals)
    elif isinstance(vals, sympy.Expr) and not len(collate_symbols(vals)):
      return float(vals)
    elif etype == np.ndarray:
      return np.array(vals)
    return vals

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
  def __getitem__(self, arg):
    """ Returns the expression object if arg is slice(None) i.e. [:] """
    assert arg == slice(None), \
        "Expr[arg] must bec called using slice operator i.e. Expr[:], not: " + \
        "{}".format(arg)
    return self._expr

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#------------------------------------------------------------------------------
