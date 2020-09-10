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
          int:  (int, np.dtype('int'), np.dtype('int32'), np.dtype('int64'),
                 np.uint, np.uint8, np.uint16, np.uint32, np.uint64),
          float: (float, np.dtype('float'),  np.dtype('float32'), np.dtype('float64')),
         }

#-------------------------------------------------------------------------------
def eval_vtype(vtype):
  """ Evaluates a variable type from input vtype.

  :param vtype: input object

  :return: either bool, int, float if correctly detected. 
           Otherwise the dtype is returned.
  """
  if isinstance(vtype, set):
    vtype = list(vtype)
  if isinstance(vtype, (list, tuple)):
    if any([isinstance(_vtype, tuple) for _vtype in vtype]):
        vtype = np.concatenate([np.array(_vtype).reshape([1]) \
                                for _vtype in vtype]).dtype
    else:
      vtype = np.array(vtype)
  if isinstance(vtype, np.ndarray):
    vtype = vtype.dtype
  elif hasattr(vtype, 'type'):
    vtype = vtype.type
  for key, val in VTYPES.items():
    try:
      found = vtype in val
    except TypeError:
      vtype = type(vtype)
      found = vtype in val
    if found:
      vtype = key
      break
  return vtype

#-------------------------------------------------------------------------------
def isunitset(var, vtype=None):
  """ Returns boolean flag as to whether var is a set of one element.

  :param var: variable to assess.
  :param vtype: optional filter for variable type (e.g. int or float)
  """
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
def isunitsetint(var):
  """ Returns boolean flag as to whether var is a set of one integer element.
  Note: Usage depends on class:
  RVs, RJs, RFs: set(int) is a sample specification denoting number of samples:
                   positive values request samples using linear interpolation
                   negative values request samples using random generation.
  Dist: set(int): proxies as a 'value' for a variable as a set of size int.
  """
  return isunitset(var, int) 

#-------------------------------------------------------------------------------
def isunitsetfloat(var):
  """ Returns boolean flag as to whether var is a set of one float element.
  Usage requests a sampling of value from a ICDF for then given P.
  """
  return isunitset(var, float)

#-------------------------------------------------------------------------------
def isscalar(var):
  """ If var is a dimensionless numpy array of size, this function returns True
  otherwise np.isscalar(var) is returned.
  """
  if isinstance(var, np.ndarray):
    if var.ndim == 0 and var.size == 1:
      return True
  return np.isscalar(var)

#-------------------------------------------------------------------------------
def issingleton(var):
  """ If isunitset(var) is True, this function returns True, 
  otherwise isscalar(var) is returned.
  """
  # Here we define singleton as a unit set or scalar
  if isunitset(var):
    return True
  return isscalar(var)

#-------------------------------------------------------------------------------
def isunitary(var):
  """ If isscalar(var) is True, this function returns True, 
  otherwise if var is numpy array of size 1, this function returns True, 
  False is returned.
  """
  if isscalar(var):
    return True
  if isinstance(var, np.ndarray):
    if var.size == 1:
      return True
  return False

#-------------------------------------------------------------------------------
def isdimensionless(var):
  """ If var is {0}, then True is returned, otherwise np.isscalar(var) is
  returned.
  """
  if isinstance(var, set):
    if len(var) == 1:
      var = list(var)[0]
      if var == 0:
        return True
      return False
  return np.isscalar(var)

#-------------------------------------------------------------------------------
def revtype(var, vtype=None):
  """ Converts var into Numpy array of variable type vtype """
  if vtype is None:
    return var
  vtype = eval_vtype(vtype)
  vartype = eval_vtype(var)
  if vtype == vartype:
    return var
  if isscalar(var):
    return vtype(var)
  return np.array(var, dtype=vtype)

#-------------------------------------------------------------------------------
def uniform(v_0=0, 
            v_1=1, 
            n=None,
            ex_0=False,
            ex_1=False):
  r""" Samples $n$ points in the range $v_0$ to $v_1$ with optional exclusion of
  respective limits according to boolean flags values (default False) given by
  ex_0 and ex_1. The sampled values are uniformly obtained in the range 
  according to $n$:

  If $n<0$, then $-n$ points are sampled randomly (ignoring limit exclusions).
  If $n=0$, then a single scalar value is sampled randomly.
  If $n>0$, then $n$ points are linearly sampled according to limit exclusions.
  """

  # Zero or negative denote random uniform
  if not n:
    return np.random.uniform(v_0, v_1)
  if n < 0:
    return np.random.uniform(v_0, v_1, size=-n)

  # Using linspace may require slicing
  if not ex_0 and not ex_1:
    if n == 1:
      return np.linspace(v_0, v_1, 3)[1:-1]
    else:
      return np.linspace(v_0, v_1, n)
  if ex_0 and ex_1:
    return np.linspace(v_0, v_1, n+2)[1:-1]
  if ex_0 and not ex_1:
    return np.linspace(v_0, v_1, n+1)[1:]
  if not ex_0 and ex_1:
    return np.linspace(v_0, v_1, n+1)[:-1]

  # We shouldn't reach here
  raise ValueError("Unrecognised exclusions {} and {}".format(ex_0, ex_1))

#-------------------------------------------------------------------------------
