'''
A simple functional wrapper without the big guns of func_tools,
making use of the 'order' keyword to map arguments and keywords.
This makes 'order' a disallowed keyword.

Func may be a tuple of callable/uncallable functions
'''
import numpy as np
from prob.vtypes import isscalar

#-------------------------------------------------------------------------------
class Func:

  # Protected
  _func = None
  _args = None
  _kwds = None

  # Private
  __istuple = None
  __isscalar = None # not of interest to Func but to objects that call Func
  __callable = None
  __order = None
  __index = None

#-------------------------------------------------------------------------------
  def __init__(self, func=None, *args, **kwds):
    self.set_func(func, *args, **kwds)
    
#-------------------------------------------------------------------------------
  def set_func(self, func=None, *args, **kwds):
    self._func = func
    self._args = tuple(args)
    self._kwds = dict(kwds)
    self.__order = None
    self.__callable = None

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
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    
#-------------------------------------------------------------------------------
  def set_order(self, order=None):
    self.__order = order
    if self.__order is None:
      return
    # Sanity check order
    key_list = list(self.__order.keys())
    ind_list = list(self.__order.values())
    keys = []
    inds = []
    for key, ind in zip(key_list, ind_list):
      keys.append(key)
      if type(ind) is int:
        inds.append(ind)
      elif ind is None:
        pass
      elif not isinstance(ind, str):
        raise TypeError("Cannot interpret order value: {}".ind)
    indset = set(inds)
    if len(indset):
      assert indset == set(range(min(indset), max(indset)+1)), \
          "Index specification non_sequitur: {}".format(indset)

#-------------------------------------------------------------------------------
  def ret_callable(self):
    return self.__callable

#-------------------------------------------------------------------------------
  def ret_isscalar(self):
    return self.__isscalar

#-------------------------------------------------------------------------------
  def ret_istuple(self):
    return self.__istuple

#-------------------------------------------------------------------------------
  def _call(self, *args, **kwds):
    func = self._func
    if self.__index is not None:
      func = func[self.__index]
      self.__index = None

    # Non-callables
    if not self.__callable:
      assert not args and not kwds, "No optional args with uncallable function"
      return func

    # Callables order-free
    args = tuple(args)
    kwds = dict(kwds)
    if not kwds and len(args) == 1 and isinstance(args[0], dict):
      args, kwds = (), dict(args[0])
    if not self.__order:
      return func(*args, **kwds)

    # Callables with order wrapper
    n_args = len(args)
    for val in self.__order.values():
      if type(val) is int:
        n_args = max(n_args, val+1)
    args = list(args)
    while len(args) < n_args:
      args.append(None)
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

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   assert not self.__istuple, "Cannot call with func tuple, use FuncWrap[]"
   return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, index=None):
   assert self.__istuple, "Cannot index without single func, use FuncWrap()"
   self.__index = index
   return self._call

#-------------------------------------------------------------------------------
