""" Symbolic representation with only a name """

#-------------------------------------------------------------------------------
import sympy as sy

#-------------------------------------------------------------------------------
class Term:
  """ This class wraps sy.Symbol. Sympy's dependence on hash tables and __new__
  members doesn't play nicely with inheriting them so we wrap them in a class
  instead and copy over the attributes.
  """
  symbol = None

#-------------------------------------------------------------------------------
  def __init__(self, symbol, *args, **kwds):
    self.set_symbol(symbol, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_symbol(self, symbol, *args, **kwds):
    # Pass symbol or create symbol named according to string
    self.symbol = symbol
    if isinstance(self.symbol, sy.Symbol):
      pass
    elif isinstance(symbol, str):
      self.symbol = sy.Symbol(self.symbol, *args, **kwds)
    else:
      raise TypeError("Symbol name must be string; {} entered".format(symbol))

    # Try to make Term play nicely with Sympy by copying attributes
    members = dir(self.symbol)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self.symbol, member)
          setattr(self, member, attribute)
        except AttributeError:
          pass

#-------------------------------------------------------------------------------
  def __and__(self):
    return NotImplemented("And operators not supported: use sy.And()")

#-------------------------------------------------------------------------------
  def __or__(self):
    return NotImplemented("Or operators not supported: use sy.Or()")

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
