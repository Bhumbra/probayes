"""
A probability class supporting probability distributions without specification 
of a variable set.
"""

#-------------------------------------------------------------------------------
import collections
import functools
import numpy as np
import scipy.stats
import sympy as sy
import sympy.stats
from probayes.icon import isiconic
from probayes.pscales import eval_pscale, rescale, iscomplex
from probayes.expression import Expression

#-------------------------------------------------------------------------------
SCIPY_STATS_CONT = {scipy.stats.rv_continuous}
SCIPY_STATS_DISC = {scipy.stats.rv_discrete}
SCIPY_STATS_MVAR = {scipy.stats._multivariate.multi_rv_generic}
SCIPY_STATS_DIST = SCIPY_STATS_MVAR.union(
                       SCIPY_STATS_CONT.union(SCIPY_STATS_DISC).\
                           union(SCIPY_STATS_MVAR))
SCIPY_DIST_METHODS = ['pdf', 'logpdf', 'pmf', 'logpmf', 'cdf', 'logcdf', 'ppf', 
                      'rvs', 'sf', 'logsf', 'isf', 'moment', 'stats', 'expect', 
                      'entropy', 'fit', 'median', 'mean', 'var', 'std', 'interval']

#-------------------------------------------------------------------------------
def is_scipy_stats_cont(arg, scipy_stats_cont=SCIPY_STATS_CONT):
  """ Returns if arg belongs to scipy.stats.continuous """
  return isinstance(arg, tuple(scipy_stats_cont))

def is_scipy_stats_disc(arg, scipy_stats_disc=SCIPY_STATS_DISC):
  """ Returns if arg belongs to scipy.stats.continuous """
  return isinstance(arg, tuple(scipy_stats_disc))

def is_scipy_stats_mvar(arg, scipy_stats_mvar=SCIPY_STATS_MVAR):
  """ Returns if arg belongs to scipy.stats._multivariate.multi_rv_generic """
  return isinstance(arg, tuple(scipy_stats_mvar))

def is_scipy_stats_dist(arg, scipy_stats_dist=SCIPY_STATS_DIST):
  """ Returns if arg belongs to scipy.stats.continuous or discrete """
  return isinstance(arg, tuple(scipy_stats_dist))

#-------------------------------------------------------------------------------
SYMPY_STATS_DIST = {sympy.stats.rv.RandomSymbol}
def is_sympy_stats_dist(arg, sympy_stats_dist=SYMPY_STATS_DIST):
  """ Returns if arguments belongs to sympy.stats.continuous or discrete """
  return isinstance(arg, tuple(sympy_stats_dist))

#-------------------------------------------------------------------------------
def sympy_prob(sympy_dist, vals, dtype=None, pfunc=sympy.stats.density, use_log=False):
  if isinstance(vals, np.ndarray) and vals.ndim:
    shape = vals.shape
    vals = np.ravel(vals).tolist()
    if dtype:
      prob = [dtype(pfunc(sympy_dist)(val)) for val in vals]
    else:
      prob = [pfunc(sympy_dist)(val) for val in vals]
    prob = np.array(prob).reshape(shape)
    if dtype and use_log:
      return np.log(prob)
    return prob
  prob = pfunc(sympy_dist)(vals)
  if dtype:
    prob = dtype(prob)
    if use_log:
      return np.log(prob)
    return prob
  return prob

#-------------------------------------------------------------------------------
def sympy_sfun(sympy_dist, size=0, dtype=None, sfunc=sympy.stats.sample):
  if not size:
    samples = sfunc(sympy_dist)
    if dtype:
      return (dtype)(samples)
    return samples
  if dtype:
    samples = [dtype(sfunc(sympy_dist)) for _ in range(size)]
    return np.array(samples)
  else:
    samples = [sfunc(sympy_dist) for _ in range(size)]
    return samples

#-------------------------------------------------------------------------------
class Prob (Expression): 
  """ A probability is quantification of degrees of belief concerning outcomes.
  It is therefore an expression that can be defined with respect to no, one,
  or more variable.

  :example:
  >>> import scipy.stats
  >>> import probayes as pb
  >>> normprob = pb.Prob(scipy.stats.norm)
  >>> print(normprob(0.))
  0.3989422804014327
  >>> normlogp = pb.Prob(scipy.stats.norm, pscale='log')
  >>> print(normlogp(0.))
  -0.9189385332046727

  Use of Sympy's probabilistic expressions can be set w.r.t. a symbol
  :example:
  >>> import sympy; import sympy.stats
  >>> import probayes as pb
  >>> x = sympy.Symbol('x')
  >>> normprob = pb.Prob(sympy.stats.Normal(x, 0, 1))
  """

  # Protected
  _prob = None       # Probability distribution function
  _logp = None       # Log probability distribution function (for SymPy only)
  _pscale = None     # Probability type (can be a scipy.stats.dist object)
  _pfun = None       # 2-length tuple of cdf/icdf
  _sfun = None       # Random-variate sampling function

  # Private
  __isscipy = None   # Boolean flag of whether expression is a scipy stats object
  __issympy = None   # Boolean flag of whether expression is a sympy stats object
  __issmvar = None   # Boolean flag of whether expression is a scipy mvar object

#-------------------------------------------------------------------------------
  def __init__(self, prob=None, *args, **kwds):
    """ Initialises the probability and pscale (see set_prob). """
    self.set_prob(prob, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def prob(self):
    return self._prob

  @property
  def isscipy(self):
    return self.__isscipy

  @property
  def issympy(self):
    return self.__issympy

  @property
  def issmvar(self):
    return self.__issmvar

  def set_prob(self, prob=None, *args, **kwds):
    """ Sets the probability and pscale with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.

    'pscale' is a reserved keyword. See set_pscale() for explanation of how 
    pscale is used.
    """
    pscale = None if 'pscale' not in kwds else kwds.pop('pscale')
    self.__isscipy = is_scipy_stats_dist(prob)
    self.__issympy = is_sympy_stats_dist(prob)
    self.__issmvar = is_scipy_stats_mvar(prob)

    # Probabilities can be defined as regular expressions
    if not self.__isscipy and not self.__issympy:
      self.set_expr(prob, *args, **kwds)
      self._prob = self._expr
      self.pscale = pscale
      return

    # Scipy/SymPy expressions
    if self.__issmvar: # Scipy self._expr must be instantiated as a frozen object
      self._expr = prob
    else:
      self.set_expr(prob, *args, **kwds)
    self._args = tuple(args)
    self._kwds = dict(kwds)
    self._prob = self._expr
    self._ismulti = True
    self._callable = True
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    if 'delta' in self._kwds:
      self.set_delta(self._kwds.pop('delta'))
    self.pscale = pscale
    self._set_partials()

    # Scipy dist
    if self.__isscipy:

      # Set pfun and sfun objects
      if 'cdf' in self._keys and 'ppf' in self._keys:
        self.set_pfun((self._expr.cdf, self._expr.ppf), *self._args, **self._kwds)
      if 'rvs' in self._keys and hasattr(self._expr, 'rvs'):
        self.set_sfun(self._expr.rvs, *self._args, **self._kwds)

    # Sympy sampler - CDFs require a Symbol and therefore are not set here
    elif self.__issympy:
      self.set_sfun(sympy_sfun, self._expr)

#-------------------------------------------------------------------------------
  @property
  def logp(self):
    return self._logp

  def _set_partials(self):

    # Regular expressions
    self._partials = collections.OrderedDict()
    if not self.__isscipy and not self.__issympy:
      super()._set_partials()
    
    # Extract SciPy object member functions
    elif self.__isscipy:

      # Instantiate multivariate scipy objects with pre-specified args
      if self.__issmvar and is_scipy_stats_mvar(self._expr):
        self._expr = self._expr(*self._args,  **self._kwds)
        self._args, self._kwds = (), {} # No longer used

      # Iterate available methods
      for method in SCIPY_DIST_METHODS:
        if hasattr(self._prob, method):
          call = functools.partial(Expression._partial_call, self, 
                                   getattr(self._expr, method),
                                   *self._args, **self._kwds)
          self._partials.update({method: call})

      # Provide a common interface for PDF/PMF and LOGPDF/LOGPMF
      if 'pdf' in self._partials.keys():
          self._partials.update({'prob': self._partials['pdf']})
          self._partials.update({'logp': self._partials['logpdf']})
      elif 'pmf' in self._partials.keys():
          self._partials.update({'prob': self._partials['pmf']})
          self._partials.update({'logp': self._partials['logpmf']})

    # Extract prob for sympy objects
    elif self.__issympy:
      kwds = dict(self._kwds)
      if 'dtype' not in kwds:
        kwds.update({'dtype':float})
      self._prob = Expression(sympy_prob, self._expr, *self._args, **kwds)
      if 'use_log' not in kwds:
        kwds.update({'use_log': True})
      self._logp = Expression(sympy_prob, self._expr, *self._args, **kwds)

    self._keys = list(self._partials.keys())

#-------------------------------------------------------------------------------
  @property
  def pscale(self):
    return self._pscale

  @pscale.setter
  def pscale(self, pscale=None):
    """ Sets the probability scaling constant used for probabilities.

    :param pscale: can be None, a real number, or a complex number, or 'log'

       if pscale is None (default) the normalisation constant is set as 1.
       if pscale is real, this defines the normalisation constant.
       if pscale is complex, this defines the offset for log probabilities.
       if pscale is 'log', this denotes a logarithmic scale with an offset of 0.

    :return: pscale (either as a real or complex number)
    """
    self._pscale = eval_pscale(pscale)
    return self._pscale

#-------------------------------------------------------------------------------
  @property
  def pfun(self):
    return self._pfun

  def set_pfun(self, pfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_prob(). It is necessary to set these
    functions if sampling variables non-randomly with non-flat distributions.

    :param pfun: two-length tuple of callable functions
    :param *args: arguments to pass to callable function
    :param **kwds: keywords to pass to callable function
    """
    self._pfun = pfun
    if self._pfun is None:
      return

    # Non-iconic ufun inputs must a two-length-tuple
    if not isiconic(self._pfun):
      message = "Input pfun must be a two-sized tuple of callable functions"
      assert isinstance(self._pfun, tuple), message
      assert len(self._pfun) == 2, message
      assert callable(self._pfun[0]), message
      assert callable(self._pfun[1]), message
    elif 'invertible' not in kwds:
      kwds.update({'invertible': True})
    self._pfun = Expression(self._pfun, *args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def sfun(self):
    return self._sfun

  def set_sfun(self, sfun=None, *args, **kwds):
    """ Sets the random variate sampling function with respect to the callable 
    function set by set_prob(). It is necessary to set this functions if 
    sampling variables within infinite limits.

    :param sfun: a callable function or a two-tuple pair of functions
    :param *args: arguments to pass to callable function
    :param **kwds: keywords to pass to callable function

    If two functions are entered, the first is assumed for contiguous 
    unrandomised sampling, and the second for randomised.
    """
    self._sfun = sfun
    if self._sfun is None:
      return

    # Non-iconic sfun inputs must a two-length-tuple
    if not isiconic(self._sfun):
      message = "Input sfun must be a single or pair of callable functions"
      if isinstance(self._sfun, tuple):
        assert len(self._sfun) == 2, message
        assert callable(self._sfun[0]), message
        assert callable(self._sfun[1]), message
      else:
        assert callable(self._sfun), message
    self._sfun = Expression(self._sfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def rescale(self, probs, **kwds):
    """ Returns a rescaling of probs from current pscale to the values according
    to the keyword pscale=new_pscale. """
    if 'pscale' not in kwds:
      return probs
    return rescale(probs, self._pscale, kwds['pscale'])

#-------------------------------------------------------------------------------
  def eval_prob(self, *args, **kwds):
    """ Evaluates the probability inputting optional args for callable cases

    :param *args: optional arguments for callable probability objects.
    :param **kwds: optional arguments to include pscale for rescaling.

    :return: evaluated probabilities
    """
    # Strip pscale if pscale
    kwds = dict(kwds)
    pscale = None if 'pscale' not in kwds else kwds.pop('pscale')

    # Callable and non-callable evaluations
    if self.callable:

      # Scipy
      if self.__isscipy:
        prob = self._partials['logp'] if iscomplex(self._pscale) \
                else self._partials['prob']
        if self.issmvar: # for mvar, convert dictionaries to arrays
          vals = np.array(list(args[0].values()))
          prob = prob(vals)
        else:
          prob = prob(*args)

      # Sympy
      elif self.__issympy:
        prob = self._logp if iscomplex(self._pscale) else self._prob
        prob = prob(*args)

      # Call via expression partials interface
      else:
        prob = self._partials[None](*args, **kwds)

    else:
      assert not len(args), \
          "Cannot evaluate from values from an uncallable probability function"
      prob = self._prob()

    # Optionally rescale
    if pscale:
      return self.rescale(prob, kwds['pscale'])
    return prob

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ See eval_prob() """
    return Prob.eval_prob(self, *args, **kwds)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
