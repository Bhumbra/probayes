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
  _index = None

#-------------------------------------------------------------------------------
  def __init__(self, *args):
    super().__init__(*args)

#-------------------------------------------------------------------------------
