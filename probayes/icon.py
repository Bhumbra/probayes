""" Symbolic representation wrapping SymPy's Symbol and Expr. """

#-------------------------------------------------------------------------------
import sympy
import sympy.stats

#-------------------------------------------------------------------------------
def isiconic(var):
  """ Returns whether object is a Sympy object. This function is purposefull not
  called issymbolic because he use 'icon' to denote any symbolic object]
  including sympy Integer and Float types. """
  # Non-callables objects derive from sympy.Basic
  if not callable(var):
    return isinstance(var, sympy.Basic)
  return False

  """
  # A hacky solution for functions
  import sys
  module_path = str(sys.modules.get(var.__module__))
  return 'sympy.' in module_path and '/sympy/' in module_path
  """

#-------------------------------------------------------------------------------
class Icon:
  """ This class wraps sympy.Symbol and sympy.Expr denoting an 'icon' as any
  object derived from sym
  . Sympy's dependence on __new__ 
  to return modified class objects at instantiation is makes multiple
  inheritance tricky so instead we wrap them in here as a class and copy over 
  the attributes.

  The resulting instance can be treated as a SymPy object using the __invert__
  method (~instance):

  :example
  >>> import probayes as pb
  >>> x = pb.Icon('x')
  >>> x2 = 2 * x[:]
  >>> print(x2.subs({x[:]: 4}))
  8
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
    Either pass a Sympy expression or sympy.Symbol object directly or in accordance 
    with the calling conventions for sympy.Symbol.__new__
    """

    # Pass symbolic or create symbolc named according to string
    self._icon = icon
    if isiconic(self._icon):
      pass
    elif isinstance(self._icon, str):
      self._icon = sympy.Symbol(self._icon, *args, **kwds)
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
  def __getitem__(self, arg):
    """ Method [:] overloaded to return icon object. """
    assert arg == slice(None), \
        "Only ':' input accepted for __getitem__ method  Icon[:], not: {}".\
        format(arg)
    return self._icon

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
