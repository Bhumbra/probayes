import numpy as np
from prob.vtypes import isunitsetint, isunitsetfloat, uniform, VTYPES

"""
A module to provide functional support to rv.py
"""
#-------------------------------------------------------------------------------
def nominal_uniform_prob(*args, prob=1., vset=None):

  assert len(args) >= 1, "Minimum of a single positional argument"
  vals = args[0]

  # Default to prob if no values
  if vals is None:
    return prob
  vtype = eval_vtype(vset)

  # If scalar, check within variable set
  if isscalar(vals):
    if vtype in VTYPES[float]:
      prob = 0. if vals < min(vset) or vals > max(vset) else prob
    else:
      prob = prob if vals in vset else 0.
    return prob

  # Otherwise treat as arrays
  vals = np.atleast_1d(vals)
  prob = np.tile(prob, vals.shape)

  # Handle nominal probabilities
  if vtype in VTYPES[bool]:
    isfalse = np.logical_not(vals)
    prob[isfalse] = 1. - prob[isfalse]
    return prob

  # Otherwise treat as uniform within range
  if vtype in VTYPES[float]:
    outside = np.logical_or(vals < min(vset), vals > max(vset))
    prob[outside] = 0.
  else:
    outside = np.array([val not in vset for val in vals], dtype=bool)
    prob[outside] = 0.

  # Broadcast probabilities across args
  if len(args) > 1:
    for arg in args[1:]:
      prob = prob * nominal_uniform(arg, vset=vset)

  return prob

#-------------------------------------------------------------------------------
def nominal_uniform_cond(prev_vals, next_vals, prob=1., vset=None):
  assert not isunitset(prev_vals), "Preceding values cannot be a unit set"
  if isunitsetint(next_vals):
    pass


    
  # Handle non-sampling case
  if not isunitset(next_vals):
    return next_vals, nominal_uniform_prob(prev_vals, 
                                           next_vals, 
                                           prob=prob, 
                                           vset=vset)




#-------------------------------------------------------------------------------

