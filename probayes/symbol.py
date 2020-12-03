""" Symbolic representation wrapping SymPy's Symbol. """

#-------------------------------------------------------------------------------
import sympy as sy
import collections

#-------------------------------------------------------------------------------
class Symbol:
  """ This class wraps sy.Symbol. Sympy's dependence on __new__ to return
  modified class objects at instantiation doesn't play nicely with multiple
  inheritance wrap them in here as a class instead and copy over the attributes.

  The resulting instance can be treated as a SymPy object using the __getitem__
  method (probayes.Symbol[:]):

  :example
  >>> import probayes as pb
  >>> x = pb.Symbol('x')
  >>> x2 = 2 * x[:]
  >>> print(x2.subs({x: 4.}))
  >>>
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

    # Copy attributes and hash content
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
  def __getitem__(self, *args):
    return self.symbol

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
