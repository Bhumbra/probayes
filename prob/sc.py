"""
A stocastic collection is a family of random variables.
"""
#-------------------------------------------------------------------------------
import numpy as np
from prob.rv import RV
import collections

#-------------------------------------------------------------------------------
class SC:

  # Protected
  _rvs = None
  _nrvs = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    self.set_rvs(*args)

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    if len(args) == 1 and isinstance(args[0], (SC, dict, set, tuple, list)):
      args = args[0]
    self.add_rv(args)
    return self.ret_rvs()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    if self._rvs is None:
      self._rvs = collections.OrderedDict()
    if isinstance(rv, (SC, dict, set, tuple, list)):
      rvs = rv
      if isinstance(rvs, SC):
        rvs = rvs.ret_rvs()
      if isinstance(rvs, dict):
        rvs = rvs.values()
      [self.add_rv(rv) for rv in rvs]
    else:
      assert isinstance(rv, RV), \
          "Input not a RV instance but of type: {}".format(type(rv))
      assert rv.name not in self._rvs.keys(), \
          "Existing RV name {} already present in collection".format(rv.name)
      self._rvs.update({rv.name: rv})
    self._nrvs = len(self._rvs)
    return self._nrvs
#-------------------------------------------------------------------------------
  def __len__(self):
    return self._nvrs

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = list(self._rvs.keys())[key]
    if isinstance(key, str):
      return self._rvs[key]
    raise TypeError("Unexpected key type: {}".format(key))

#-------------------------------------------------------------------------------
