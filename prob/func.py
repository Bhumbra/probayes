'''
A simple functional wrapper without the big guns of func_tools,
making use of the 'order' keyword to map arguments and keywords.
This makes 'order' a disallowed keyword.

Func may be a tuple of callable/uncallable functions
'''

#-------------------------------------------------------------------------------
class Func:

  # Protected
  _func = None
  _args = None
  _kwds = None

  # Private
  __func_tuple = None
  __func_callable = None
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

    # Sanity check func
    if self._func is None:
      assert not args and not kwds, "No optional args without a function"
    self.__func_tuple = isinstance(self._func, tuple)
    if not self.__func_tuple:
      self.__func_callable = callable(self._func)
      if not self.__func_callable:
        assert not args and not kwds, "No optional args with uncallable function"
    else:
      self.__func_callable = [callable(func) for func in self._func]
      if not all(self.__func_callable):
        assert not args and not kwds, "No optional args with uncallable function"
      assert len(set(self.__func_callable)) < 2, \
          "Cannot mix callable and uncallable functions"
    if 'order' in self._kwds:
      self.set_order(self._kwds.pop('order'))
    return self.__func_callable
    
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
      elif not isinstance(ind, str):
        raise TypeError("Cannot interpret order value: {}".ind)
    indset = set(inds)
    assert indset == set(range(len(indset))), \
        "Index specification non_sequitur: {}".format(indset)

#-------------------------------------------------------------------------------
  def _call(self, *args, **kwds):
    val = tuple(args)
    func, func_callable = self._func, self.__func_callable 
    if self.__index is not None:
      func = func[self.__index]
      func_callable = func_callable[self.__index]
      self.__index = None
    args = tuple(args)
    kwds = dict(kwds)
    if not func_callable:
      assert not args and not kwds, "No optional args with uncallable function"
      return func
    if not self.__order:
      return func(*args, **kwds)
    if args:
      assert len(args) == 1 and not kwds and isinstance(args[0], dict), \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
      kwds = dict(args[0])
    elif kwds:
      assert not args, \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
    n_args = sum([type(val) is int for val in self.__order.values()])
    args = [None] * n_args
    for key, val in self.__order.items():
      if type(val) is int:
        args[val] = kwds.pop(key)
      else:
        kwds.update({val: kwds.pop(key)})
    return func(*tuple(args), **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
   assert not self.__func_tuple, "Cannot call with func tuple, use FuncWrap[]"
   return self._call(*args, **kwds)

#-------------------------------------------------------------------------------
  def __getitem__(self, index=None):
   assert self.__func_tuple, "Cannot index without single func, use FuncWrap()"
   self.__index = index
   return self._call

#-------------------------------------------------------------------------------
