import numpy as np
from probayes.vtypes import isunitsetint, isunitsetfloat, isunitset, isscalar, \
                        uniform, eval_vtype, VTYPES

from probayes.pscales import eval_pscale, rescale, iscomplex, NEARLY_NEGATIVE_INF
"""
A module to provide functional support to rv.py
"""
#-------------------------------------------------------------------------------
def nominal_uniform_prob(*args, prob=None, inside=None, pscale=1.):
  """ Also handles categorical data types if inside is a list. Inside can
  (and should) be a callable function """

  # Detect ptype, default to prob if no values, otherwise detect vtype  
  assert len(args) >= 1, "Minimum of a single positional argument"
  pscale = eval_pscale(pscale)
  use_logs = iscomplex(pscale)
  if prob is None:
    prob = 0. if use_logs else 1.
  vals = args[0]
  if vals is None:
    return prob
  vtype = eval_vtype(vals) if callable(inside) else eval_type(inside)

  # Set inside function by vtype if not specified
  if not callable(inside):
    if vtype in VTYPES[bool]:
      pass
    elif vtype in VTYPES[float]:
      inside = lambda x: np.logical_and(x >= min(inside), x <= max(inside))
    else:
      inside = lambda x: np.isin(x, inside)

  # If scalar, check within variable set
  p_zero = NEARLY_NEGATIVE_INF if use_logs else 0.
  if isscalar(vals):
    if vtype in VTYPES[bool]:
      prob = prob if vals else \
             rescale(1. - rescale(prob, pscale, 1.), 1., pscale)
    else:
      prob = prob if inside(vals) else p_zero

  # Otherwise perform array operations
  else:

    # Handle nominal probabilities, that ignores inside
    if vtype in VTYPES[bool]:
      prob = rescale(prob, pscale, 1.)
      p_false = 1. - prob
      prob = np.tile(prob, vals.shape)
      prob[np.logical_not(vals)] = p_false
      prob = rescale(prob, 1., pscale)

    # Otherwise treat as uniform within range
    else:
      p_true = prob
      prob = np.tile(p_zero, vals.shape)
      prob[inside(vals)] = p_true

  # This section below is there just to play nicely with conditionals
  if len(args) > 1:
    for arg in args[1:]:
      if use_logs:
        prob = prob + nominal_uniform_prob(arg, inside=inside, pscale=0.j)
      else:
        prob = prob * nominal_uniform_prob(arg, inside=inside)
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
  rc_scalar = False
  if row_idx is None:
    if isscalar(row_vals):
      rc_scalar = True
      row_idx = vset.index(row_vals)
    else:
      row_shape = None
      if isinstance(row_vals, np.ndarray):
        row_vals = np.ravel(row_vals).tolist()
      row_idx = [vset.index(row_val) for row_val in row_vals]
  elif isscalar(row_idx):
      rc_scalar = True
  if col_idx is None:
    if isscalar(col_vals):
      rc_scalar = True
      col_idx = vset.index(col_vals)
    else:
      if isinstance(col_vals, np.ndarray):
        col_vals = np.ravel(col_vals).tolist()
      col_idx = [vset.index(col_val) for col_val in col_vals]
  elif isscalar(col_idx):
      rc_scalar = True
  if rc_scalar:
    return sq_matrix[row_idx, col_idx]
  mat = np.empty([len(row_idx), len(col_idx)], dtype=sq_matrix.dtype)
  for i, row in enumerate(row_idx):
    mat[i] = sq_matrix[row][col_idx]
  return mat
    
#-------------------------------------------------------------------------------
