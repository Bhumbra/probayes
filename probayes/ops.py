""" Provides operator functions for deterministic and stochastic classes """

#-------------------------------------------------------------------------------
def and_op(self, other):
  """ Unites (self, other) for Variables, Fields, Dependences, and their
  stochastic equivalents. """
  from probayes.variable import Variable
  from probayes.field import Field
  from probayes.rf import RF
  from probayes.dependence import Dependence
  from probayes.sd import SD

  # Determine whether stochastic or non-stochastic field
  assert hasattr(self, 'is_stochastic'), \
      "Unrecognised pre-operand type {}".format(type(self))
  assert hasattr(other, 'is_stochastic'), \
      "Unrecognised post-operand type {}".format(type(other))
  fld, dep = Field, Dependence
  if any([self.is_stochastic, other.is_stochastic]):
    fld, dep = RF, SD

  # Dependence as pre-operand
  if isinstance(self, Dependence):
    return dep(other, self)

  # Field as pre-operand
  if isinstance(self, Field):
    if isinstance(other, Dependence):
      leafs = list(self.leafs.varlist) + \
              list(other.leafs.varlist)
      stems = other.stems
      roots = other.roots
      args = [fld(*tuple(leafs))]
      if stems:
        args += list(stems.values())
      if roots:
        args += [roots]
      return dep(*tuple(args))

    if isinstance(other, Field):
      leafs = list(self.vars.values()) + list(other.vars.values())
      return fld(*tuple(leafs))

    if isinstance(other, Variable):
      leafs = list(self.vars.values()) + [other]
      return fld(*tuple(leafs))

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))


  # Variable as pre-operand
  if isinstance(self, Variable):
    if isinstance(other, Dependence):
      leafs = [self] + list(other.leafs.vars.values())
      stems = other.stems
      roots = other.roots
      args = fld(*tuple(leafs))
      if stems:
        args += list(stems.values())
      if roots:
        args += list(roots.values())
      return dep(*args)

    if isinstance(other, Field):
      var = [self] + list(other.vars.values())
      return fld(*tuple(var))

    if isinstance(other, Variable):
      return fld(self, other)
  
    raise TypeError("Unrecognised post-operand type: ".format(type(other)))

  raise TypeError("Unrecognised pre-operand type: ".format(type(self)))

#-------------------------------------------------------------------------------
def or_op(self, other):
  """ Conditions self w.r.y. other for Variables, Fields, Dependences, and their
  stochastic equivalents. """
  from probayes.dependence import Dependence
  from probayes.sd import SD

  # Determine whether stochastic or non-stochastic dependence
  assert hasattr(self, 'is_stochastic'), \
      "Unrecognised pre-operand type {}".format(type(self))
  assert hasattr(other, 'is_stochastic'), \
      "Unrecognised post-operand type {}".format(type(other))
  dep = SD if any([self.is_stochastic, other.is_stochastic]) else Dependence

  # Check for unviable cross conditionaliiation
  if isinstance(self, Dependence) and isinstance(other, Dependence):
    if self.roots and other.roots:
      raise ValueError("Cannot cross-conditionalise dependencies")
  return dep(self, other)

#-------------------------------------------------------------------------------

