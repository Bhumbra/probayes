import numpy as np
from prob.vtypes import isunitsetint, isunitsetfloat, isunitset, \
                        uniform, VTYPES

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
def matrix_cond_sample(pred_vals, succ_vals, prob, vset=None):
  """ Returns succ_vals with sampling """
  if not isunitset(succ_vals):
    return succ_vals
  assert isscalar(pred_vals), \
      "Can only cumulatively sample from a single predecessor"
  assert prob.ndim==2 and len(set(prob.shape)) == 1, \
      "Transition matrix must be a square"
  support = range(prob.shape[0])
  if vset is None:
    vset = set(range(support))
  else:
    assert len(vset) == support, \
        "Transition matrix size {} incommensurate with set support {}".\
        format(support, len(vset))
  vset = np.array(vset)
  pred_idx = vset.find(pred_vals)
  cmf = np.cumsum(prob[:, pred_idx], axis=0)
  succ_cmf = list(succ_vals)[0]
  if type(succ_cmf) in VTYPES[int]:
    succ_cmf = uniform(0., 1., succ_cmf)
  else:
    succ_cmf = np.atleast_1d(succ_cmf)
  succ_idx = np.maximum(0, np.minimum(support-1, np.nonzero(succ_cmf, cmf)))
  return vset[succ_idx], pred_idx, succ_idx

#-------------------------------------------------------------------------------
def lookup_square_matrix(col_vals, row_vals, matrix, 
                         vset=None, col_idx=None, row_idx=None):
  assert prob.ndim==2 and len(set(prob.shape)) == 1, \
      "Transition matrix must be a square"
  support = range(matrix.shape[0])
  if vset is None:
    vset = set(range(support))
  else:
    assert len(vset) == support, \
        "Transition matrix size {} incommensurate with set support {}".\
        format(support, len(vset))
  if row_idx is None:
    if isscalar(row_vals):
      row_idx = vset.find(row_vals)
    else:
      row_idx = [vset.find(row_val) for row_val in row_vals]
  if col_idx is None:
    if isscalar(col_vals):
      col_idx = vset.find(col_vals)
    else:
      col_idx = [vset.find(col_val) for col_val in pred_vals]
  return prob[row_idx, col_idx]

#-------------------------------------------------------------------------------
