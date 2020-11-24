"""
A variable is a named representation of a quantity.
"""

#-------------------------------------------------------------------------------
import numpy as np
import sympy as sy
from probayes.vtypes import revtype

#-------------------------------------------------------------------------------
DEFAULT_VNAME = 'var'
DEFAULT_VTYPE = bool
SY_SYMBOL = sy.Symbol

#-------------------------------------------------------------------------------
class Variable (SY_SYMBOL):
                    
  # Protected       
  _name = None      # Variable type
  _vtype = None      # Variable type

#-------------------------------------------------------------------------------
  def __new__(cls, name=DEFAULT_VNAME,
                   vtype=DEFAULT_VTYPE,
                   *args,
                   **kwds):
    return super(Variable, cls).__new__(cls, name, *args, **kwds)

#-------------------------------------------------------------------------------
  def __init__(self, name=DEFAULT_VNAME,
                     vtype=DEFAULT_VTYPE,
                     *args,
                     **kwds):
    """ Initialiser variable name and vtype:

    :param name: Name of the domain - string as valid identifier.
    :param vtype: variable type (bool, int, or float).
    :param *args: optional arguments to pass onto symbol representation.
    :param *kwds: optional keywords to pass onto symbol representation.
    """
    SY_SYMBOL.__init__(self) # takes no parameters
    self._name = name
    self._vtype = vtype

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name=DEFAULT_VNAME):
    self._name = name

#-------------------------------------------------------------------------------
  @property
  def vtype(self):
    return self._vtype

  @vtype.setter
  def vtype(self, vtype=DEFAULT_VTYPE):
    self._vtype = vtype

#-------------------------------------------------------------------------------
  def eval_vals(self, values=None):
    """ Outputs the values in accordance with variable type, returning either
    a scalar or NumPy array"""
    if values is None:
      return None
    return revtype(self._self_eval(values), self._vtype)

#-------------------------------------------------------------------------------
  def __call__(self, values=None):
    """ See eval_vals() """
    return self.eval_vals(values)

#-------------------------------------------------------------------------------
