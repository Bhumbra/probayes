'''
A wrapper for sympy-based probability objects
'''
from probayes.expr import Expr


#-------------------------------------------------------------------------------
class SympyProb (Expr):
  """ An expression wrapper for Sympy-based probabilities.
  """

  # Protected
  _args = None
  _kwds = None
  _keys = None
  _partials = None # Ordered Dictionary of calls which may include partials

#-------------------------------------------------------------------------------
  def __init__(self, expr=None, *args, **kwds):
    """
    See set_expr()
    """
    self.set_expr(expr, *args, **kwds)
    
#-------------------------------------------------------------------------------
  @property
  def args(self):
    return self._args

  @property
  def kwds(self):
    return self._kwds

  @property
  def keys(self):
    return self._keys

  def set_expr(self, expr=None, *args, **kwds):
    """ Set the Func instance's function object.

    :param expr: an uncallable object, callable function, or tuple of functions
    :param *args: arguments to pass onto callables
    :param **kwds: keywords to pass onto callables
    """
    self._expr = expr
    self._args = tuple(args)
    self._kwds = dict(kwds)

    # Sanity check func
    if self._expr is None:
      assert not args and not kwds, "No optional args without a function"
    self._set_partials()

#-------------------------------------------------------------------------------
  def _set_partials(self):
    # Protected function to update partial function dictionary of calls
    self._partials = collections.OrderedDict()
    self._keys = list(self._partials.keys())


#-------------------------------------------------------------------------------
