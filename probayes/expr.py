""" Expric representation wrapping SymPy's Expr. """

#-------------------------------------------------------------------------------
import collections
import sympy as sy
import numpy as np
from sympy.utilities.autowrap import ufuncify
from probayes.variable_utils import parse_as_str_dict

DEFAULT_EFUNS = [
                  #({np.ndarray: sy.lambdify}, ('numpy',), {})
                  ({np.ndarray: ufuncify}, (), {})
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
  expr = None     # Expression object

  # Protected
  _symbols = None # Ordered dictionary of symbols keyed by name
  _efun = None    # Dictionary of evaluation functions

#-------------------------------------------------------------------------------
  def __init__(self, expr, *args, **kwds):
    self.set_expr(expr, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_expr(self, expr, *args, **kwds):
    """ Sets the expr object for this instance with optional args and kwds.
    Either pass a sy.Expr object directly or in accordance with the calling
    conventions for sy.Expr.__new__
    """

    # Pass expr or create expr named according to string
    self.expr = expr
    self._symbols = collections.OrderedDict()
    if isinstance(self.expr, sy.Expr):
      for atom in self.expr.atoms():
        if hasattr(atom, 'name'):
          self._symbols.update({atom.name: atom})
    else:
      raise TypeError("Input type not Expr type but: {}".format(expr))

    # Make instance play nicely with Sympy by copying attributes and hash content
    members = dir(self.expr)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self.expr, member)
          setattr(self, member, attribute)
        except AttributeError:
          pass
    for efun in DEFAULT_EFUNS:
      self.add_efun(efun[0], *tuple(efun[1]), **dict(efun[2]))

#-------------------------------------------------------------------------------
  def add_efun(self, efun=None, *args, **kwds):
    """ Adds an expression evaluation function.
    :param efun: a single dictionary entry in the form {type: function}
    :param args: optional args to pass to efun
    :param kwds: optional kwds to pass to efun

    :returns: updated efun dictionary.
    """
    if self._efun is None:
      self._efun = {None: (self.expr.subs)}
    if efun is None or not self._symbols:
      return
    symbols = tuple(self._symbols.values())
    assert isinstance(efun, dict), \
        "Input efun must be dictionary type, not {}".format(type(efun))
    for key, val in efun.items():
      self._efun.update({key: val(symbols, self.expr, *args, **kwds)})
    return self._efun

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Performs evaluation of expression inputting symbols as a dictionary 
    (see sympy.Symbol.subs() input specificion using dictionaries. """
    values = parse_as_str_dict(*args, **kwds)
    etype = None
    evalues = [None] * len(self._symbols)
    for i, key in enumerate(self._symbols.keys()):
      evalues[i] = values[key]
      _etype = type(evalues[i])
      if etype is None: 
        if _etype in self._efun.keys():
          etype = _etype
      elif _etype in self._efun.keys():
        assert etype == _etype, \
            "Cannot mix types {} vs {} ".format(etype, _etype)
    efun = self._efun[etype]
    vals = efun(*evalues) if etype else efun(values)
    if isinstance(vals, sy.Integer):
      return int(vals)
    elif isinstance(vals, sy.Float):
      return float(vals)
    if etype != np.ndarray or isinstance(vals, np.ndarray):
      return vals
    return np.array(vals)

#-------------------------------------------------------------------------------
  def __and__(self):
    return NotImplemented("And operators not supported: use sy.And()")

#-------------------------------------------------------------------------------
  def __or__(self):
    return NotImplemented("Or operators not supported: use sy.Or()")

#-------------------------------------------------------------------------------
  def __xor__(self):
    return NotImplemented("Xor operators not supported: use sy.Xor()")

#-------------------------------------------------------------------------------
  def __pos__(self):
    return self.expr.__pos__()

#-------------------------------------------------------------------------------
  def __neg__(self):
    return self.expr.__neg__()

#-------------------------------------------------------------------------------
  def __lt__(self, other):
    return self.expr.__lt__(other)

#-------------------------------------------------------------------------------
  def __le__(self, other):
    return self.expr.__le__(other)

#-------------------------------------------------------------------------------
  def __eq__(self, other):
    return self.expr.__eq__(other)

#-------------------------------------------------------------------------------
  def __ne__(self, other):
    return self.expr.__ne__(other)

#-------------------------------------------------------------------------------
  def __ge__(self, other):
    return self.expr.__ge__(other)

#-------------------------------------------------------------------------------
  def __gt__(self, other):
    return self.expr.__gt__(other)

#-------------------------------------------------------------------------------
  def __add__(self, other):
    return self.expr.__add__(other)

#-------------------------------------------------------------------------------
  def __sub__(self, other):
    return self.expr.__sub__(other)

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return self.expr.__mul__(other)

#-------------------------------------------------------------------------------
  def __matmul__(self, other):
    return self.expr.__matmul__(other)

#-------------------------------------------------------------------------------
  def __div__(self, other):
    return self.expr.__div__(other)

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    return self.expr.__truediv__(other)

#-------------------------------------------------------------------------------
  def __getitem__(self, *args):
    return self.expr

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#------------------------------------------------------------------------------
