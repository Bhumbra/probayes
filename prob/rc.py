"""
A random collection is a collection of random variables.
"""
#-------------------------------------------------------------------------------
import numpy as np
from prob.rv import RV
import collections

#-------------------------------------------------------------------------------
class RC:

  # Protected
  _rvs = None
  _index = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    if len(args) == 1:
      if isinstance(args[0], (tuple, list)):
        args = args[0]
      elif isinstance(args[0], RC):
        args = RC.ret_rvs()
      else:
        raise TypeError("Unknown RVs argument type: {}".format(type(args[0])))
    if isinstance(args, dict):
      for rv in args.values():
        self.add_rv(rv)
    else:
      for rv in args:
        self.add_rv(rv)
    return self.ret_rvs()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    if self._rvs is None:
      self._rvs = collections.OrderedDict()
    assert isinstance(rv, RV), \
        "RV not a RandVar instance but of type: {}".format(type(rv))
    self._rvs.update({rv.name: rv})

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = list(self._rvs.keys())[key]
    if isinstance(key, str):
      return self._rvs[key]
    raise TypeError("Unexpected key type: {}".format(key))

#-------------------------------------------------------------------------------
