# This module is to be deprecated along with Func.
'''A conditional function is a functional wrapper that describes the 
dependence of an about output RF with respect to an input RF or the
by the dependence of a subgroup of RVs with respect to the others. 
'''
import collections
from probayes.rv import RV
from probayes.func import Func

#-------------------------------------------------------------------------------
def _to_rf(obj=None):
  """ Converts obj (which may be an RV/RF) or list/tuple of such to an RF """
  from probayes.rf import RF
  if obj is None:
    return obj
  if isinstance(obj, RF):
    assert not obj.stems and not obj.roots, \
        "RFs must be rootless and stemless"
    return obj
  elif isinstance(obj, RV):
    return RF(obj)
  elif isinstance(obj, (list, tuple)):
    rvs = []
    for subobj in obj:
      if isinstance(subobj, RV):
        rvs += [subobj]
      elif isinstance(subobj, RF):
        assert not obj.stems and not obj.roots, \
            "RFs must be rootless and stemless"
        rvs += subobj.varlist
      else:
        raise TypeError("Unexpected type: {}".format(type(subobj)))
    return RF(*tuple(rvs))
  else:
    raise TypeError("Unexpected type: {}".format(type(subobj)))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class CF (Func):
  '''A conditional function is a functional wrapper that describes the 
  dependence of an about output RF with respect to an input RF or the
  by the dependence of a subgroup of RVs with respect to the others. 
  '''
  # Protected
  _out = None
  _inp = None
  _name = None
  _passprob = None
  _passdims = None

  # Private
  __inpdict = None
  __unknown = None

#-------------------------------------------------------------------------------
  @property
  def callable(self):
    return self._Func__callable

  @property
  def isscipy(self):
    return self._Func__isscipy

  @property
  def isscalar(self):
    return self._Func__isscalar

  @property
  def ismulti(self):
    return self._Func__ismulti

  def __init__(self, out=None, inp=None, func=None, *args, **kwds):
    """ Sets the output and input random fields out and inp and initialises 
    the function according to object in func, which may be an uncallable
    object, a callable function, or a tuple of callable functions. See Func.
    """
    self.set_out(out)
    self.set_inp(inp)
    self.set_func(func, *args, **kwds)
    
#-------------------------------------------------------------------------------
  def set_out(self, out=None):
    """ Sets the output and input random fields out, which may be an RF an RV or 
    tuple or RFs and RVs but not an SD. 
    """
    self._out = _to_rf(out)

#-------------------------------------------------------------------------------
  def set_inp(self, inp=None):
    """ Sets the output and input random fields out and inp, each of which may 
    be an RF an RV or tuple or RFs and RVs but not an SD. Optionally inp may
    be an OrderedDict with RV names in the form (only keys are checked):
    {'x_0,x_1': 'x_2,x_3,x_4', 'x_2': 'x_0', 'x_3,x_4': 'x_0,x_1,x_2'}. 
    """
    self.__inpdict = isinstance(inp, (tuple, list, dict))
    if self.__inpdict:
      if isinstance(inp, (tuple, list)):
        inp = {key: '' for key in inp}
    self.__unknown = None

    # Handle inp is a random field first
    if not self.__inpdict and inp is not None:
      assert self._out is not None, "Set outputs before inputs"
      self._inp = _to_rf(inp)
      self._name = '|'.join([self._out.name, self._inp.name])
      return 
    elif inp is not None and not isinstance(inp, collections.OrderedDict):
      raise TypeError("Dictionary specification for inp must be an OrderedDict")

    # At this stage, there must be at least two RVs
    rv_keys = self._out.keylist
    assert len(rv_keys) > 1, "Multiple RVs required for conditional functions"

    # Check inp if entered
    if inp:
      all_inp_keys = [] 
      for key, val in inp.items():
        inp_keys = key.split(',')
        val_keys = val.split(',')
        for val_key in val_keys:
          if len(inp) > 1:
            assert val_key not in inp_keys, \
                "Circular key reference {}:{}".format(key, val)
          assert val_key in rv_keys, \
              "Key {} not recognised in RF {}".format(val_key, self._out)
        all_inp_keys += inp_keys
      assert set(rv_keys) == set(all_inp_keys), \
          "Conditional function {} specification incommensurate with RF {}".\
          format(inp, self._out)
      self._inp = inp
    prime_keys = ["{}'".format(rv_key) for rv_key in rv_keys]
    self._name = '|'.join([','.join(prime_keys), ','.join(rv_keys)])

#-------------------------------------------------------------------------------
  def set_func(self, func=None, *args, **kwds):
    """ Set the CF instance's function object.

    :param func: an uncallable object, callable function, or tuple of functions
    :param *args: arguments to pass onto callables
    :param **kwds: keywords to pass onto callables

    Note that the following two reserved keywords are disallowed:

    'order': which instead denotes a dictionary of remappings.
    'delta': which instead denotes a mapping of differences.
    'passdims': which provides an auxiliary flag to SD to pass dims
    'passprob': which provides an auxiliary flag to SD to pass prob`

    """

    self._func = func
    self._args = tuple(args)
    self._kwds = dict(kwds)
    self._passdims = None if 'passdims' not in kwds else self._kwds.pop('passdims')
    self._passprob = None if 'passprob' not in kwds else self._kwds.pop('passprob')
    if isinstance(self._func, Func) and not self._args and not self._kwds:
      func = self._func.ret_func()
      args = self._func.ret_args()
      kwds = self._func.ret_kwds()
      order = self._func.ret_order()
      delta = self._func.ret_delta()
      argout = super().set_func(func, *args, **kwds)
      self.set_order(order)
      self.set_delta(delta)
      return argout
    super().set_func(self._func, *self._args, **self._kwds)

#-------------------------------------------------------------------------------
  def ret_name(self):
    """ Return the conditional function name determined from the RFs """
    return self._name

#-------------------------------------------------------------------------------
  def ret_out(self):
    """ Returns output RF """
    return self._out

#-------------------------------------------------------------------------------
  def ret_inp(self):
    """ Returns input RF """
    return self._inp

#-------------------------------------------------------------------------------
  def ret_passdims(self):
    """ Returns passdims flag if specified """
    return self._passdims

#-------------------------------------------------------------------------------
  def ret_passprob(self):
    """ Returns passprob flag if specified """
    return self._passprob

#-------------------------------------------------------------------------------
  def ret_inpdict(self):
    """ Returns flag inpdict indicating whether a dict """
    return self.__inpdict

#-------------------------------------------------------------------------------
  def __getitem__(self, spec=None):
   r""" Calls the $i$th function from the Func tuple where the spec is $i$.
   If spec is a string, then the specification is changed.
   """
   if not isinstance(spec, str):
     return super().__getitem__(spec)
   assert self.__inpdict, \
       "Input spec as a string only supported with a OrderedDict inp"
   self.__unknown = spec
   return self._call

#-------------------------------------------------------------------------------
  def _call(self, *args, **kwds):
    if self.__unknown is None:
      return super()._call(*args, **kwds)
    kwds = dict(kwds)
    kwds.update({'unknown': self.__unknown})
    self.__unknown = None
    return super()._call(*args, **kwds)

#-------------------------------------------------------------------------------
