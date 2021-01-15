""" Symbolic representation wrapping SymPy's Symbol and Expr. """

#-------------------------------------------------------------------------------
import sympy as sy
import sympy.stats
import collections

#-------------------------------------------------------------------------------
def isiconic(var):
  """ Returns whether object is a Sympy object """
  # Non-callables objects derive from sy.Basic
  if not callable(var):
    return isinstance(var, sy.Basic)
  return False

  """
  # A hacky solution for functions
  import sys
  module_path = str(sys.modules.get(var.__module__))
  return 'sympy.' in module_path and '/sympy/' in module_path
  """

#-------------------------------------------------------------------------------
class Icon:
  """ This class wraps sy.Symbol and sy.Expr. Sympy's dependence on __new__ 
  to return modified class objects at instantiation is makes multiple
  inheritance tricky so instead we wrap them in here as a class and copy over 
  the attributes.

  The resulting instance can be treated as a SymPy object using the __invert__
  method (~instance):

  :example
  >>> import probayes as pb
  >>> x = pb.Icon('x')
  >>> x2 = 2 * ~x
  >>> print(x2.subs({x: 4.}))
  2*x
  >>>
  """
  _icon = None

#-------------------------------------------------------------------------------
  def __init__(self, icon, *args, **kwds):
    self.set_icon(icon, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def icon(self):
    return self._icon

  def set_icon(self, icon, *args, **kwds):
    """ Sets the icon object for this instance with optional args and kwds.
    Either pass a Sympy expression or sy.Symbol object directly or in accordance 
    with the calling conventions for sy.Symbol.__new__
    """

    # Pass symbolic or create symbolc named according to string
    self._icon = icon
    if isiconic(self._icon):
      pass
    elif isinstance(self._icon, str):
      self._icon = sy.Symbol(self._icon, *args, **kwds)
    else:
      raise TypeError("Symbol name must be string; {} entered".format(self._icon))

    # Copy attributes and hash content
    members = dir(self._icon)
    for member in members:
      if not hasattr(self, member):
        try:
          attribute = getattr(self._icon, member)
          setattr(self, member, attribute)
        except AttributeError:
          pass

#-------------------------------------------------------------------------------
  def __repr__(self):
    if self._icon is None:
      return super().__repr__()
    return self._icon.__repr__()

#-------------------------------------------------------------------------------
  def __hash__(self):
    if self._icon is None:
      return super().__hash__()
    return self._icon.__hash__()

#-------------------------------------------------------------------------------
  def __invert__(self):
    """ Bitwise invert operator overloaded to return icon object. """
    return self._icon

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
