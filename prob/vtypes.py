"""
A module that handles variable data types.
"""

#-------------------------------------------------------------------------------
import numpy as np
import functools
import operator

#-------------------------------------------------------------------------------
VTYPES = {
          bool: (bool, np.dtype('bool')),
          int:  (int, np.dtype('int'), np.dtype('int32'), np.dtype('int64')), 
          np.uint: (np.uint, np.uint8, np.uint16, np.uint32, np.uint64),
          float: (float, np.dtype('float'),  np.dtype('float32'), np.dtype('float64')),
         }

#-------------------------------------------------------------------------------
def eval_vtype(vtype):
  if isinstance(vtype, set):
    vtype = list(vtype)
  if isinstance(vtype, (list, tuple)):
    vtype = np.array(vtype)
  if isinstance(vtype, np.ndarray):
    vtype = vtype.dtype
  for key, val in VTYPES.items():
    if vtype in val:
      vtype = key
      break
  return vtype

#-------------------------------------------------------------------------------
def isunitset(var, vtype=None):
  if vtype is None:
    vtype = list(VTYPES.keys())
  elif not isinstance(vtype, (tuple,list)):
    vtype = [vtype]
  vtypes = functools.reduce(operator.concat, [VTYPES[key] for key in vtype])
  if isinstance(var, set):
    if len(var) == 1:
      element_type = type(list(var)[0])
      if element_type in vtypes:
        return True
  return False

#-------------------------------------------------------------------------------
def isunitsetuint(var):
  # Unsigned ints denote without bounds without margins
  return isunitset(var, np.uint)

#-------------------------------------------------------------------------------
def isunitsetsint(var):
  return isunitset(var, int)

#-------------------------------------------------------------------------------
def isunitsetint(var):
  """ Usage depends on class:
  RVs, SJs, SCs: set(int) is a sample specification denoting number of samples:
                   positive values request samples using linear interpolation
                   negative values request samples using random generation.
  Dist: set(int): proxies as a 'value' for a variable as a set of size int.
  """
  return isunitset(var, (int, np.uint)) 

#-------------------------------------------------------------------------------
def isunitsetfloat(var):
  """ Usage requests a sampling of value from a ICDF for then given P """
  return isunitset(var, float)

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
def uniform(v_0=0, v_1=1, number=1):
  if not number:
    return np.random.uniform(v_0, v_1)
  if number >= 0:
    if type(number) in VTYPES[np.uint]:
      return np.linspace(v_0, v_1, number)
    else:
      return np.linspace(v_0, v_1, number + 2)[1:-1]
  return np.random.uniform(v_0, v_1, size=-number)

#-------------------------------------------------------------------------------
