"""
A module that handles probability calculations according different ptypes that
may represent probability coefficients (positive float scalars) or log 
probability offsets (complex float scalars).
"""

#-------------------------------------------------------------------------------
import numpy as np
""" Set limits to 32-bit precision """
NEARLY_POSITIVE_ZERO = 1.175494e-38
NEARLY_NEGATIVE_INF = -3.4028236e38
NEARLY_POSITIVE_INF =  3.4028236e38
LOG_NEARLY_POSITIVE_INF = np.log(NEARLY_POSITIVE_INF)
COMPLEX_ZERO = complex(0., 0.)

#-------------------------------------------------------------------------------
def iscomplex(ptype):
  return isinstance(ptype, complex)

#-------------------------------------------------------------------------------
def eval_ptype(ptype=None):
  """ Returns a float or complex ptype with the following conventions:
  if ptype is None, returns 1.
  if ptype is 'log' or 'ln' or 0, returns 0.j.
  otherwise ptype must be real or complex, then it is returned
  """
  if ptype is None:
    return 1.
  if ptype == 1:
    return 1.
  if ptype in ['log', 'ln', 0]:
    return COMPLEX_ZERO
  if isinstance(ptype, int):
    return float(ptype)
  if isinstance(ptype, float):
    if ptype == 0.:
      return COMPLEX_ZERO
    return ptype
  if iscomplex(ptype):
    return ptype
  raise ValueError("Cannot evaluate ptype={}".format(ptype))

#-------------------------------------------------------------------------------
def log_prob(prob):
  logp = np.tile(NEARLY_NEGATIVE_INF, prob.shape)
  ok = prob >= NEARLY_POSITIVE_ZERO
  logp[ok] = np.log(prob[ok])
  return logp

#-------------------------------------------------------------------------------
def exp_logp(logp):
  prob = np.tile(NEARLY_POSITIVE_INF, logp.shape)
  ok = logp <= LOG_NEARLY_POSITIVE_INF
  prob[ok] = np.exp(logp[ok])
  return prob

#-------------------------------------------------------------------------------
def logp_offs(ptype=None):
  ptype = eval_ptype(ptype)
  if not iscomplex(ptype):
    return float(np.log(ptype))
  if np.abs(np.imag(ptype)) < NEARLY_POSITIVE_ZERO:
    return float(np.real(ptype))
  return -float(np.real(ptype))

#-------------------------------------------------------------------------------
def prob_coef(ptype=None):
  ptype = eval_ptype(ptype)
  if not iscomplex(ptype):
    return float(ptype)
  if np.abs(np.imag(ptype)) < NEARLY_POSITIVE_ZERO:
    return np.exp(float(np.real(ptype)))
  return np.exp(-float(np.real(ptype)))

#-------------------------------------------------------------------------------
def rescale(prob, *args):
  """ Rescales prob according to ptypes given in args """
  ptype, rtype = None, None
  if len(args) == 0: 
    return prob
  elif len(args) ==  1: 
    rtype = args[0]
  else: 
    ptype, rtype = args[0], args[1]
  ptype, rtype = eval_ptype(ptype), eval_ptype(rtype)
  if ptype == rtype:
    return prob
  
  p_log, r_log = iscomplex(ptype), iscomplex(rtype)

  # Support non-logarithmic conversion (maybe used to avoid logging zeros)
  if not p_log and not r_log:
    coef = ptype / rtype
    if coef == 1.:
      return prob
    else:
      return coef * prob

  # For floating point precision, perform other operations in log-space
  if not p_log: prob = log_prob(prob)
  d_offs = logp_offs(ptype) - logp_offs(rtype)
  if np.abs(d_offs) >= NEARLY_POSITIVE_ZERO: prob = prob + d_offs
  if r_log:
    return prob
  return exp_logp(prob)

#-------------------------------------------------------------------------------
def prod_ptype(ptypes, use_logp=None):
  if not len(ptypes):
    return None
  if use_logp is None:
    use_logp = any([iscomplex(ptype) for ptype in ptypes])
  rtype = 0. if use_logp else 1.
  for _ptype in ptypes:
    ptype = eval_ptype(_ptype)
    if use_logp:
      rtype += logp_offs(ptype)
    else:
      rtype *= prob_coef(ptype)
  if use_logp:
    if abs(rtype) < NEARLY_POSITIVE_ZERO:
      return COMPLEX_ZERO
    elif rtype > 0:
      return complex(np.log(rtype), 0.)
    else:
      return complex(np.log(-rtype), np.pi)
  return rtype

#-------------------------------------------------------------------------------
def prod_rule(*args, **kwds):
  """ Returns prod, ptype. Reshaping is the responsibility of Dist. """
  kwds = dict(kwds)
  ptypes = kwds.get('ptypes', [1.] * len(args))
  use_logp = kwds.get('use_logp', any([iscomplex(_ptype) for _ptype in ptypes]))
  pptype = prod_ptype(ptypes, use_logp)
  ptype = kwds.get('ptype', pptype)
  n_args = len(args)
  assert len(ptypes) == n_args, \
      "Input ptypes length {} incommensurate with number of arguments {}".\
      format(len(ptypes), n_args)
  
  def _apply_prod(probs):
    # Numpy sum() and prod() produce inconsistent results with lists
    if len(probs) == 1:
      prob = np.copy(probs[0])
    else:
      prob = probs[0] + probs[1] if use_logp else probs[0] * probs[1]
      for _prob in probs[2:]:
        if use_logp:
          prob = prob + _prob
        else:
          prob = prob * _prob
    return prob

  # Possibly fast-track
  if use_logp != iscomplex(pptype):
    pptype = complex(np.log(pptype), 0.) if use_logp else float(np.exp(pptype))
  elif use_logp == iscomplex(ptype) and pptype == ptype and \
      len(set([iscomplex(_ptype) for _ptype in ptypes])) == 1:
    prob = _apply_prod(list(args))
    return prob, ptype

  # Otherwise exp/log before evaluating product
  probs = [None] * n_args
  for i, arg in enumerate(args):
    p_log = iscomplex(ptypes[i])
    probs[i] = args[i]
    if use_logp:
      if not p_log:
        probs[i] = log_prob(probs[i])
    else:
      if p_log:
        probs[i] = exp_logp(probs[i])
  prob = _apply_prod(probs)
  if use_logp != iscomplex(ptype):
    prob = rescale(prob, pptype, ptypes)

  return prob, ptype

#-------------------------------------------------------------------------------
