"""
A stocastic process is indexable sequence of realisations of a stochastic condition
"""
#-------------------------------------------------------------------------------
import numpy as np
from prob.sc import SC
import collections

#-------------------------------------------------------------------------------
class SP (SC):

  # Protected
  _dists = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    super().__init__(*args)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    if self._dists is None:
      self._dists = []
    self._dists.append(super().__call__(*args, **kwds))
    return self._dists[-1]

#-------------------------------------------------------------------------------
  def __getitem__(self, index=None):
    if self._dists is None or index is None:
      return self._dists
    return self._dists[index]

#-------------------------------------------------------------------------------
  def __len__(self):
    if self._dists is None:
      return None
    return len(self._dists)

#-------------------------------------------------------------------------------
