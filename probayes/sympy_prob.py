'''
A wrapper for sympy-based probability objectsj
Note we eschew sympy.stats.density etc...
'''
import collections
import functools
import numpy as np
import sympy
import sympy.stats
from probayes.expr import Expr

#-------------------------------------------------------------------------------
SYMPY_STATS_DIST = {sympy.stats.rv.RandomSymbol}
def is_sympy_stats_dist(arg, sympy_stats_dist=SYMPY_STATS_DIST):
  """ Returns if arguments belongs to sympy.stats.continuous or discrete """
  return isinstance(arg, tuple(sympy_stats_dist))

#-------------------------------------------------------------------------------
def sympy_obj_from_dist(dist):
  """ Attempts to return the object class instance for sympy distribution.
  :example:
  >>> import sympy
  >>> import sympy.stats
  >>> import probayes as pb
  >>> x = sympy.Symbol('x')
  >>> p_x = sympy.stats.Normal(x, mean=0, std=1.)
  >>> p_x_obj = pb.sympy_obj_from_dist(p_x)
  >>> print(p_x_obj.pdf)
  0.5*sqrt(2)*exp(-0.5*x**2)/sqrt(pi)
  """
  obj = dist
  if hasattr(obj, '_cdf'):
    return obj
  if not hasattr(obj, 'args'):
    return None
  objs = [sympy_obj_from_dist(arg) for arg in obj.args]
  if any(objs):
    for obj in objs:
      if obj:
        return obj
  return None

#-------------------------------------------------------------------------------
def sympy_sfun(distr, size=0, dtype=None, _sfunc=sympy.stats.sample):
  """ Sampling function for Sympy distributions where:

  :param distr: Sympy stats distribution
  :param size: size specification (as per scipy rvs)
  :param dtype: Numpy-style dtype specification

  :return randomly samples from distribution.
  """
  if not size:
    samples = _sfunc(distr)
    if dtype:
      return (dtype)(samples)
    return samples
  if dtype:
    samples = [dtype(_sfunc(distr)) for _ in range(size)]
    return np.array(samples)
  samples = [_sfunc(distr) for _ in range(size)]
  return samples

#-------------------------------------------------------------------------------
class SympyProb:
  """ An expression wrapper for Sympy-based probabilities:

  :example

  >>> import sympy
  >>> import sympy.stats
  >>> import probayes as pb
  >>> import numpy as np
  >>> x = sympy.Symbol('x')
  >>> normal = pb.SympyProb(sympy.stats.Normal(x, mean=0, std=1))
  >>> print(normal['prob'](np.array([-1., 0., 1.])))
  [0.24197072 0.39894228 0.24197072]
  """

  # Protected
  _distr = None # Distribution function
  _probj = None # Probability distribution object
  _exprs = None # Distribution expression dictionary
  _cterm = None # Cumulative term for CDFs 
  __icdf = None # Inverse CDF symbol

#-------------------------------------------------------------------------------
  def __init__(self, distr=None):
    self.distr = distr
    
#-------------------------------------------------------------------------------
  @property
  def distr(self):
    return self._distr

  @property
  def probj(self):
    return self._probj

  @property
  def exprs(self):
    return self._exprs

  @distr.setter
  def distr(self, distr=None):
    """ Sets the distribution object. This must be a Sympy.stats distribution.
    """
    self._distr = distr
    self._probj = None
    self._exprs = None
    if self._distr is None:
      return
    assert is_sympy_stats_dist(self._distr), \
        "Input dist must a Sympy statistical distribution: {} entered".format(
            self._distr)
    self._probj = sympy_obj_from_dist(self._distr)
    self._cterm = self._distr.args[0] # assume first term is cumulative
    self._exprs = collections.OrderedDict()

    if not isinstance(self._cterm, sympy.Symbol):
      raise TypeError("Unexpected first argument type: {}".format(type(self._cterm)))
    
    # Set PDF, CDF, ICDF
    if hasattr(self._probj, 'pdf'):
      self._exprs.update({'prob': Expr(self._probj.pdf(self._cterm))})
      self._exprs.update({'logp': Expr(sympy.log(self._exprs['prob'].expr))})
    if hasattr(self._probj, '_cdf'):
      self._exprs.update({'cdf': Expr(self._probj._cdf(self._cterm))})
      icdf_name = "_{}_cdf".format(self._cterm.name)
      self.__icdf = sympy.Symbol(icdf_name)
      invexprs = sympy.solve(self._exprs['cdf'].expr - self.__icdf,  self._cterm)
      n_exprs = len(invexprs)
      if n_exprs:
        invexpr = invexprs[0]
        if n_exprs > 1:
          for expr in invexprs[1:]:
            if len(expr.__repr__()) < len(invexpr.__repr__()):
              invexpr = expr
        self._exprs.update({'icdf': Expr(invexpr, remap={self._cterm.name: icdf_name})})

    # Set sampling function
    self._exprs.update({'sfun': functools.partial(sympy_sfun, self._distr)})

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
    """ Returns class member or partials dictionary object according to arg:

    If None or [:], the entire expression dictionary is returned.
    If [], then the original distribution is returned.
    If {}, then the distribution probability class object is returned.
    If '', then the keys of the partials dictionary are returned.
    Otherwise arg is treated as the key for the expressions dictionary.
    """
    if isinstance(arg, tuple):
      raise NotImplementedError("Tuple input for __getitem__ not supported")
    if arg is None or arg == slice(None):
      return self._exprs
    if isinstance(arg, str) and not arg:
      return list(self._exprs.keys())
    if isinstance(arg, (list, dict)):
      assert not len(arg), "Input argument of type {} must be empty".type(arg)
      if isinstance(arg, list):
        return self._distr
      if isinstance(arg, dict):
        return self._probj
    return self._exprs[arg]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
