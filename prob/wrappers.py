"""
A module to support dimension specification by string references from dictionaries
"""
import numpy 

#-------------------------------------------------------------------------------
def wrap_swapaxes(array, in_order, out_order):
  n = array.ndim
  assert n == len(in_order), "Input order incommensurate"
  assert n == len(out_order), "Output order incommensurate"
  in_axes = list(range(n))
  out_axes = [None] * n
  for i in range(n):
    for j in range(n):
      if in_order[i] == out_order[j]:
        assert out_axes[i] is not None, "Ambiguous in {} and out {} axes names".\
            format(in_order, out_order)
        out_axes[i] = j
  return np.swapaxes(array, in_axes, out_axes)

#-------------------------------------------------------------------------------
