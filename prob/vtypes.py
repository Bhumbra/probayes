"""
A module that handles variable data types.
"""

#-------------------------------------------------------------------------------
import numpy as np

#-------------------------------------------------------------------------------
NUMPY_DTYPES = {
                 np.dtype('bool'): bool,
                 np.dtype('int'): int,
                 np.dtype('int32'): int,
                 np.dtype('int64'): int,
                 np.dtype('float'): float,
                 np.dtype('float32'): float,
                 np.dtype('float64'): float,
               }

#-------------------------------------------------------------------------------
def eval_vtype(vtype):
  if isinstance(vtype, set):
    vtype = list(vtype)
  if isinstance(vtype, (list, tuple)):
    vtype = np.array(vtype)
  if isinstance(vtype, np.ndarray):
    vtype = vtype.dtype
  if vtype in NUMPY_DTYPES:
    vtype = NUMPY_DTYPES[vtype]
  if vtype in [bool, int, float]:
    return vtype
  return vtype

#-------------------------------------------------------------------------------
def isunitsetint(var):
  if isinstance(var, set):
    if len(var) == 1:
      element = list(var)[0]
      if type(element) is int:
        return True
  return False

#-------------------------------------------------------------------------------
def isscalar(var):
  # Integer singleton sets denote unspecified scalars 
  # as well as undimensioned Numpy arrays
  if isunitsetint(var):
    return True
  if isinstance(var, np.ndarray):
    if var.ndim == 0 and var.size == 1:
      return True
  return np.isscalar(var)

#-------------------------------------------------------------------------------
