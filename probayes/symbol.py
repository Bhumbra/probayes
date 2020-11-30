""" Symbolic representation wrapping SymPy's Symbol. """

#-------------------------------------------------------------------------------
import sympy as sy

#-------------------------------------------------------------------------------
class Symbol:
  """ This class wraps sy.Symbol. Sympy's dependence on __new__ to return
  modified class objects at instantiation doesn't play nicely with multiple
  inheritance wrap them in here as a class instead and copy over the attributes.

  The resulting instance can be treated in _almost_ the same way as SymPy's, but
  for identical behaviour, use probayes.Symbol[:] method:

  :example
  >>> import probayes as pb
  >>> sym = pb.Symbol('sym')
  >>> sym2 = sym * 2
  >>> sym2 = 2 * sym
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: unsupported operand type(s) for *: 'int' and 'Symbol'
  >>> sym2 = 2 * sym[:]
  """

  symbol = None

#-------------------------------------------------------------------------------
  def __init__(self, symbol, *args, **kwds):
    self.set_symbol(symbol, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_symbol(self, symbol, *args, **kwds):
    """ Sets the symbol object for this instance with optional args and kwds.
    Either pass a sy.Symbol object directly or in accordance with the calling
    conventions for sy.Symbol.__new__
    """

    # Pass symbol or create symbol named according to string
    self.symbol = symbol
    if isinstance(self.symbol, sy.Symbol):
      pass
    elif isinstance(symbol, str):
      self.symbol = sy.Symbol(self.symbol, *args, **kwds)
    else:
      raise TypeError("Symbol name must be string; {} entered".format(symbol))

    # Make instance play nicely with Sympy by copying attributes and hash content
    members = dir(self.symbol)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self.symbol, member)
          setattr(self, member, attribute)
        except AttributeError:
          pass

#-------------------------------------------------------------------------------
  def __hash__(self):
    if self.symbol is None:
      return super().__hash__()
    return self.symbol.__hash__()

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
    return self.symbol.__pos__()

#-------------------------------------------------------------------------------
  def __neg__(self):
    return self.symbol.__neg__()

#-------------------------------------------------------------------------------
  def __lt__(self, other):
    return self.symbol.__lt__(other)

#-------------------------------------------------------------------------------
  def __le__(self, other):
    return self.symbol.__le__(other)

#-------------------------------------------------------------------------------
  def __eq__(self, other):
    return self.symbol.__eq__(other)

#-------------------------------------------------------------------------------
  def __ne__(self, other):
    return self.symbol.__ne__(other)

#-------------------------------------------------------------------------------
  def __ge__(self, other):
    return self.symbol.__ge__(other)

#-------------------------------------------------------------------------------
  def __gt__(self, other):
    return self.symbol.__gt__(other)

#-------------------------------------------------------------------------------
  def __add__(self, other):
    return self.symbol.__add__(other)

#-------------------------------------------------------------------------------
  def __sub__(self, other):
    return self.symbol.__sub__(other)

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return self.symbol.__mul__(other)

#-------------------------------------------------------------------------------
  def __matmul__(self, other):
    return self.symbol.__matmul__(other)

#-------------------------------------------------------------------------------
  def __div__(self, other):
    return self.symbol.__div__(other)

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    return self.symbol.__truediv__(other)

#-------------------------------------------------------------------------------
  def __getitem__(self, *args):
    return self.symbol

#-------------------------------------------------------------------------------
