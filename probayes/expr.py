""" Expric representation wrapping SymPy's Expr. """

#-------------------------------------------------------------------------------
import sympy as sy

#-------------------------------------------------------------------------------
class Expr:
  """ This class wraps sy.Expr. Sympy's dependence on __new__ to return
  modified class objects at instantiation doesn't play nicely with multiple
  inheritance wrap them in here as a class instead and copy over the attributes.
  """

  expr = None

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
    if isinstance(self.expr, sy.Expr):
      pass
    elif isinstance(expr, str):
      self.expr = sy.Expr(self.expr, *args, **kwds)
    else:
      raise TypeError("Expr name must be string; {} entered".format(expr))

    # Make instance play nicely with Sympy by copying attributes and hash content
    members = dir(self.expr)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self.expr, member)
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
