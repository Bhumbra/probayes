'''
A simple functional wrapper without the big guns of func_tools,
making use of the 'order' keyword to map arguments and keywords.
This makes 'order' a disallowed keyword. The 'delta' keyword is
similarly used to may differences in form var_name'-var_name,
making also a disallowed keyword.

Func may be a tuple of callable/uncallable functions
'''
import numpy as np
import scipy.stats
from probayes.vtypes import isscalar, isunitary

#-------------------------------------------------------------------------------
""" 
Scipy calls is supported a special case for following call convention:"
func() pdf
func[0]() pdf
func[1]() logpdf
func[2]() cdf
func[3]() logcdf
func[4]() rvs
"""
SCIPY_STATS_MVAR = {scipy.stats._multivariate.multi_rv_generic}

#------------------------------------------------------------------------------- 
def is_scipy_stats_mvar(arg, scipy_stats_mvar=SCIPY_STATS_MVAR):
  """ Returns a boolean for whether arg is an instance of scipy multivariate """
  return isinstance(arg, tuple(scipy_stats_mvar))

#-------------------------------------------------------------------------------
class Func:
  """ A functional wrapper to enable object representations as an uncallable
  array, a callable function, or a tuple of callable functions

  :example:
  >>> from probayes.func import Func
  >>> hw = Func("Hello world!")
  >>> print(hw())
  Hello World!
  >>> inc = Func(lambda x: x+1)
  >>> print(inc(2.)
  3.0
  >>> inc_dec = Func( (lambda x:x+1, lambda x:x-1) )
  >>> print(inc_dec[0](3.))
  4.0
  >>> print(inc_dec[1](3.))
  2.0
  """

  # Protected
  _func = None
  _args = None
  _kwds = None

  # Private
  __istuple = None
  __isscalar = None # not of interest to Func but to objects that call Func
  __isscipy = None
  __callable = None
  __scipyobj = None
  __scipycalls = None
  __order = None
  __delta = None
  __index = None

#-------------------------------------------------------------------------------
  def __init__(self, func=None, *args, **kwds):
    """ Initialises Func according to object in func, which may be an uncallable
    object, a callable function, or a tuple of callable functions. See set_func()
    """
    self.set_func(func, *args, **kwds)
    
#-------------------------------------------------------------------------------
  def set_func(self, func=None, *args, **kwds):
    """ Set the Func instance's function object.

    :param func: an uncallable object, callable function, or tuple of functions
    :param *args: arguments to pass onto callables
    :param **kwds: keywords to pass onto callables

    Note that the following two reserved keywords are disallowed:

    'order': which instead denotes a dictionary of remappings.
    'delta': which instead denotes a mapping of differences.

    A special case is made for func instances that are Scipy multivariate objects.
    """

    self._func = func
    self._args = tuple(args)
    self._kwds = dict(kwds)
    self.__order = None
    self.__delta = None
    self.__callable = None
    self.__scipyobj = None
    self.__isscipy = False

    # Sanity check func
    if self._func is None:
      assert not args and not kwds, "No optional args without a function"
    self.__istuple = isinstance(self._func, tuple)
    self.__isscalar = False
    if not self.__istuple:
      self.__callable = callable(self._func)
      if not self.__callable:
        assert not args and not kwds, "No optional args with uncallable function"
        self.__isscalar = isscalar(self._func)
    else:
      func_callable = [callable(func) for func in self._func]
      func_isscalar = [isscalar(func) for func in self._func]
      assert len(set(func_callable)) < 2, \
          "Cannot mix callable and uncallable functions"
      assert len(set(func_isscalar)) < 2, \
          "Cannot mix scalars and nonscalars"
      if len(func_callable):
        self.__callable = func_callable[0]
        self.__isscalar = func_isscalar[0]
      if not self.__callable:
        assert not args and not kwds, "No optional args with uncallable function"
    if is_scipy_stats_mvar(self._func):
      self.__isscipy = True
      self.__scipyobj = self._func(*args, **kwds)
      self.__scipycalls = {
                           0: self.__scipyobj.pdf,
                           1: self.__scipyobj.logpdf,
                           2: self.__scipyobj.cdf,
                           3: self.__scipyobj.logcdf,
                           4: self.__scipyobj.rvs,
                          }
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    if 'delta' in self._kwds:
      self.set_delta(self._kwds.pop('delta'))
    
#-------------------------------------------------------------------------------
  def set_order(self, order=None):
    """ Sets an order remapping dictionary for functional calls in which
    keyword arguments are mapped to position (in numeric) or rekeyed (if str).
    """
    self.__order = order
    if self.__order is None:
      return
    assert self.__delta is None, "Cannot set both order and delta"
    assert self.__scipyobj is None, \
        "Optional 'order' keyword prohibited for scipy objectts"
    self._check_mapping(self.__order)

#-------------------------------------------------------------------------------
  def set_delta(self, order=None):
    """ Sets a difference remapping dictionary for functional calls in which
    keyword arguments are mapped to position (in numeric) or rekeyed (if str).
    """
    self.__delta = delta
    if self.__delta is None:
      return
    assert self.__order is None, "Cannot set both order and delta"
    assert self.__scipyobj is None, \
        "Optional 'delta' keyword prohibited for scipy objectts"
    self._check_mapping(self.__delta)

#-------------------------------------------------------------------------------
  def _check_mapping(self, mapping=None):
    """ Perform sanity checkings on mapping dictionary """
    if mapping is None:
      return
    # Used to sanity-check mapping dicts e.g. order and delta
    assert isinstance(mapping, dict), \
        "Mapping must be a dictionary type, not {}".format(type(mapping))
    key_list = list(mapping.keys())
    ind_list = list(mapping.values())
    keys = []
    inds = []
    for key, ind in zip(key_list, ind_list):
      keys.append(key)
      if type(ind) is int:
        inds.append(ind)
      elif ind is None:
        pass
      elif not isinstance(ind, str):
        raise TypeError("Cannot interpret index specification value: {}".ind)
    indset = set(inds)
    if len(indset):
      assert indset == set(range(min(indset), max(indset)+1)), \
          "Index specification non_sequitur: {}".format(indset)

#-------------------------------------------------------------------------------
  def ret_callable(self):
    """ Returns boolean flag as to whether func is callable """
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    """ Returns boolean flag as to whether func is a scalar """
    return self.__isscalar

#-------------------------------------------------------------------------------
  def ret_istuple(self):
    """ Returns boolean flag as to whether func comprises a tuple """
    return self.__istuple

#-------------------------------------------------------------------------------
  def ret_isscipy(self):
    """ Returns boolean flag as to whether func comprises a scipy object """
    return self.__isscipy

#-------------------------------------------------------------------------------
  def ret_scipyobj(self):
    """ Returns scipy object if specified """
    return self.__scipyobj

#-------------------------------------------------------------------------------
  def _call(self, *args, **kwds):
    """ Private call used by the wrapped Func interface.
    (see __call__ and __getitem__).
    """

    # Handle scipy objects separately
    if self.__isscipy:
      return self._call_scipy(*args, **kwds)

    # Check for indexing and reset if necessary
    func = self._func
    if self.__index is not None:
      func = func[self.__index]
      self.__index = None

    # Non-callables
    if not self.__callable:
      assert not args and not kwds, "No optional args with uncallable function"
      return func

    # Callables order-free
    if not kwds and len(args) == 1 and isinstance(args[0], dict):
      args, kwds = (), dict(args[0])
    if self._args:
      args = tuple(list(self._args) + list(args))
    if self._kwds:
      kwds = {**kwds, **self._kwds}
    if not self.__order and not self.__delta:
      return func(*args, **kwds)

    # Append None to args according to mapping index specification
    n_args = len(args)
    mapping = self.__order or self.__delta
    for val in mapping.values():
      if type(val) is int:
        n_args = max(n_args, val+1)
    args = list(args)
    while len(args) < n_args:
      args.append(None)

    # Callables with order wrapper
    if self.__order:
      for key, val in self.__order.items():
        if type(val) is int:
          args[val] = kwds.pop(key)
        elif val is None:
          kwds.pop(key)
        elif isinstance(val, str):
          kwds.update({val: kwds.pop(key)})
        else:
          raise TypeError("Unrecognised order key: val type: {}:{}".\
                          format(key, val))
      return func(*tuple(args), **kwds)

    # Callables with delta wrapper
    for key, val in self.__delta.items():
      if key[-1] != "'":
        value = kwds.pop(key)
      else:
        value = kwds.pop(key) - kwds.pop(key[:-1])
      if type(val) is int:
        args[val] = value
      elif val is None:
        pass
      elif isinstance(val, str):
        kwds.update({val: value})
      else:
        raise TypeError("Unrecognised delta key: val type: {}:{}".\
                        format(key, val))
    return func(*tuple(args), **kwds)

#-------------------------------------------------------------------------------
  def _call_scipy(self, *args, **kwds):
    """ Private call used by the wrapped Func interface for scipy objects.
    (see __call__ and __getitem__).
    """
    index = 0
    if self.__index is not None:
      index = self.__index
      self.__index = None
    vals = args[0]
    if index < 4:
      if len(args) == 1 and isinstance(args[0], dict):
        args = [np.ravel(val) for val in args[0].values()]
      elif not len(args) and len(kwds):
        args = list(collections.OrderedDict(**kwds).values())
        kwds = {}
      if isinstance(args, list) and not np.all([isunitary(arg) for arg in args]):
        args = args[::-1]
        if len(args) > 2:
          args = args[1:] + [args[0]]
        args = tuple(args)
      return self.__scipycalls[index](np.stack(np.meshgrid(*args), axis=-1), 
                                      **kwds)
    return self.__scipycalls[index](*args, **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   """ Wrapper call to the function with optional inclusion of additional
   args and kwds. """

   assert not self.__istuple or self.__isscipy, \
       "Cannot call with tuple func, use Func[]"
   return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, index=None):
   r""" Calls the $i$th function from the Func tuple where the index is $i$ """
   if index is None:
     return self._func
   assert self.__istuple or self.__isscipy, \
     "Cannot index without single func, use Func()"
   self.__index = index
   return self._call

#-------------------------------------------------------------------------------
  def __len__(self):
    """ Returns the number of functions in the tuple set by set_func() """
    if not self.__istuple:
      return None
    return len(self._func)

#-------------------------------------------------------------------------------
