import numpy as np
from prob.vtypes import isunitsetint, isunitsetfloat, isunitset, isscalar, \
                        uniform, eval_vtype, VTYPES

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

  # Otherwise treat as arrays
  else:
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
      prob = prob * nominal_uniform_prob(arg, vset=vset)

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
  support = prob.shape[0]
  if vset is None:
    vset = set(range(support))
  else:
    assert len(vset) == support, \
        "Transition matrix size {} incommensurate with set support {}".\
        format(support, len(vset))
  vset = sorted(vset)
  pred_idx = vset.index(pred_vals)
  cmf = np.cumsum(prob[:, pred_idx], axis=0)
  succ_cmf = list(succ_vals)[0]
  if type(succ_cmf) in VTYPES[int]:
    succ_cmf = uniform(0., 1., succ_cmf)
  else:
    succ_cmf = np.atleast_1d(succ_cmf)
  succ_idx = np.maximum(0, np.minimum(support-1, np.digitize(succ_cmf, cmf)))
  return vset[succ_idx], pred_idx, succ_idx

#-------------------------------------------------------------------------------
def lookup_square_matrix(col_vals, row_vals, sq_matrix, 
                         vset=None, col_idx=None, row_idx=None):
  assert sq_matrix.ndim==2 and len(set(sq_matrix.shape)) == 1, \
      "Transition matrix must be a square"
  support = sq_matrix.shape[0]
  if vset is None:
    vset = list(range(support))
  else:
    assert len(vset) == support, \
        "Transition matrix size {} incommensurate with set support {}".\
        format(support, len(vset))
    vset = sorted(vset)
  if row_idx is None:
    if isscalar(row_vals):
      row_idx = vset.index(row_vals)
    else:
      row_idx = [vset.index(row_val) for row_val in row_vals]
  if col_idx is None:
    if isscalar(col_vals):
      col_idx = vset.index(col_vals)
    else:
      col_idx = [vset.index(col_val) for col_val in pred_vals]
  return sq_matrix[row_idx, col_idx]

#-------------------------------------------------------------------------------
