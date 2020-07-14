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
  """ Usage depends on class:
  RVs, SJs, SCs: set(int) is a sample specification denoting number of samples:
                   positive values request samples using linear interpolation
                   negative values request samples using random generation.
  Dist: set(int): proxies as a 'value' for a variable as a set of size int.
  """

  if isinstance(var, set):
    if len(var) == 1:
      element_type = type(list(var)[0])
      if element_type is int:
        return True
  return False

#-------------------------------------------------------------------------------
def isunitsetfloat(var):
  """ Usage requests a sampling of value from a ICDF for then given P """
  if isinstance(var, set):
    if len(var) == 1:
      element_type = type(list(var)[0])
      if element_type is float:
        return True
  return False

#-------------------------------------------------------------------------------
def isunitset(var):
  if isinstance(var, set):
    if len(var) == 1:
      element_type = type(list(var)[0])
      if element_type in (int, float):
        return True
  return False

#-------------------------------------------------------------------------------
def isscalar(var):
  if isinstance(var, np.ndarray):
    if var.ndim == 0 and var.size == 1:
      return True
  return np.isscalar(var)

#-------------------------------------------------------------------------------
def issingleton(var):
  # Here we define singleton as a unit set or scalar
  if isunitset(var):
    return True
  return isscalar(var)

#-------------------------------------------------------------------------------
