"""
A stocastic process is indexable sequence of realisations of a stochastic condition
"""
#-------------------------------------------------------------------------------
import numpy as np
from prob.sc import SC
import collections

#-------------------------------------------------------------------------------
class SP (SC):
  # Public
  opqrstu = None # opqr + scores + thresh + update

  # Private
  _scores = None # Scores function used for the basis of acceptance
  _thresh = None # Threshold function used 
  _update = None # Update function (output True, None, or False)

#-------------------------------------------------------------------------------
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)

#-------------------------------------------------------------------------------
  def _refresh(self):
    super()._refresh()
    if self._marg is None and self._cond is None:
      return
    self.opqrstu = collections.namedtuple(self._id, 
                       ['o', 'p', 'q', 'r', 's', 't', 'u'])

#-------------------------------------------------------------------------------
