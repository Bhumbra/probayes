"""
A module that handles probability calculations according different pscales that
may represent probability coefficients (positive float scalars) or log 
probability offsets (complex float scalars).
"""

#-------------------------------------------------------------------------------
import numpy as np
""" Set limits to 32-bit precision """

DEFAULT_FP_PRECISION = 64
COMPLEX_ZERO = complex(0., 0.)
FP_CONSTANTS = {32: {'nearly_positive_zero': 1.175494351e-38,
                     'nearly_positive_inf': 3.4022823466e38},
                64: {'nearly_positive_zero': 2.2250738585072014e-308,
                     'nearly_positive_inf': 1.7976931348623158e+308},
               }

NEARLY_POSITIVE_ZERO = None
NEARLY_POSITIVE_INF = None
NEARLY_NEGATIVE_INF = None
LOG_NEARLY_POSITIVE_INF =  None

def SET_FP_CONSTANTS(precision=DEFAULT_FP_PRECISION):
  """ Sets global floating point constants whether using precision=32 or
  precision=64 (default)
  """
  global NEARLY_POSITIVE_ZERO
  global NEARLY_POSITIVE_INF
  global NEARLY_NEGATIVE_INF
  global NEARLY_POSITIVE_INF
  global LOG_NEARLY_POSITIVE_INF
  NEARLY_POSITIVE_ZERO = FP_CONSTANTS[precision]['nearly_positive_zero']
  NEARLY_POSITIVE_INF = FP_CONSTANTS[precision]['nearly_positive_inf']
  NEARLY_NEGATIVE_INF = -NEARLY_POSITIVE_INF
  LOG_NEARLY_POSITIVE_INF = np.log(NEARLY_POSITIVE_INF)

if NEARLY_NEGATIVE_INF is None or LOG_NEARLY_POSITIVE_INF is None:
  SET_FP_CONSTANTS()

#-------------------------------------------------------------------------------
def iscomplex(pscale):
  """ Returns whether pscale is an instance of a complex number """
  return isinstance(pscale, complex)

#-------------------------------------------------------------------------------
def eval_pscale(pscale=None):
  """ Returns a float or complex pscale with the following conventions:
  if pscale is None, returns 1.
  if pscale is 'log' or 'ln' or 0, returns 0.j.
  otherwise pscale must be real or complex, then it is returned
  """
  if pscale is None:
    return 1.
  if pscale == 1:
    return 1.
  if pscale in ['log', 'ln', 0]:
    return COMPLEX_ZERO
  if isinstance(pscale, int):
    return float(pscale)
  if isinstance(pscale, float):
    if pscale == 0.:
      return COMPLEX_ZERO
    return pscale
  if iscomplex(pscale):
    return pscale
  raise ValueError("Cannot evaluate pscale={}".format(pscale))

#-------------------------------------------------------------------------------
def log_prob(prob):
  """ Safely returns the logarithm of probability values in prob """
  if np.isscalar(prob):
    if prob >= NEARLY_POSITIVE_ZERO:
      return np.log(prob)
    return NEARLY_NEGATIVE_INF
  logp = np.tile(NEARLY_NEGATIVE_INF, prob.shape)
  ok = prob >= NEARLY_POSITIVE_ZERO
  logp[ok] = np.log(prob[ok])
  return logp

#-------------------------------------------------------------------------------
def exp_logp(logp):
  """ Safely returns the exponentional of log probability values in logp """
  if np.isscalar(logp):
    if logp <= LOG_NEARLY_POSITIVE_INF:
      return np.exp(logp)
    return NEARLY_POSITIVE_INF
  prob = np.tile(NEARLY_POSITIVE_INF, logp.shape)
  ok = logp <= LOG_NEARLY_POSITIVE_INF
  prob[ok] = np.exp(logp[ok])
  return prob

#-------------------------------------------------------------------------------
def logp_offs(pscale=None):
  """ Returns the offset in log probability represented by pscale """
  pscale = eval_pscale(pscale)
  if not iscomplex(pscale):
    return float(np.log(pscale))
  if np.abs(np.imag(pscale)) < NEARLY_POSITIVE_ZERO:
    return float(np.real(pscale))
  return -float(np.real(pscale))

#-------------------------------------------------------------------------------
def prob_coef(pscale=None):
  """ Returns the coefficient in probability represented by pscale """
  pscale = eval_pscale(pscale)
  if not iscomplex(pscale):
    return float(pscale)
  if np.abs(np.imag(pscale)) < NEARLY_POSITIVE_ZERO:
    return np.exp(float(np.real(pscale)))
  return np.exp(-float(np.real(pscale)))

#-------------------------------------------------------------------------------
def real_sqrt(real):
  """ Safely returns the square root of positive real numbers """
  if np.isscalar(real):
    if real >= NEARLY_POSITIVE_ZERO:
      return np.sqrt(real)
    return 0.
  root = np.zeros_like(real)
  ok = real >= NEARLY_POSITIVE_ZERO
  root[ok] = np.sqrt(real[ok])
  return root

#-------------------------------------------------------------------------------
def rescale(prob, *args):
  """ Rescales prob according to pscales given in args """
  pscale, rtype = None, None
  prob = prob if np.isscalar(prob) \
           or prob.dtype in [float, np.dtype('float32'), np.dtype('float64')] \
         else np.array(prob, dtype=float)
  if len(args) == 0: 
    return prob
  elif len(args) ==  1: 
    rtype = args[0]
  else: 
    pscale, rtype = args[0], args[1]
  pscale, rtype = eval_pscale(pscale), eval_pscale(rtype)
  if pscale == rtype:
    return prob
  
  p_log, r_log = iscomplex(pscale), iscomplex(rtype)

  # Support non-logarithmic conversion (maybe used to avoid logging zeros)
  if not p_log and not r_log:
    coef = pscale / rtype
    if coef == 1.:
      return prob
    else:
      return coef * prob

  # For floating point precision, perform other operations in log-space
  if not p_log: prob = log_prob(prob)
  d_offs = logp_offs(pscale) - logp_offs(rtype)
  if np.abs(d_offs) >= NEARLY_POSITIVE_ZERO: prob = prob + d_offs
  if r_log:
    return prob
  return exp_logp(prob)

#-------------------------------------------------------------------------------
def prod_pscale(pscales, use_logp=None):
  """ Returns the natural product pscale according all the values in pscales,
  adopting a log scale if use_logp is True (which defaults to True if any
  values in pscales are complex.
  """
  if not len(pscales):
    return None
  if use_logp is None:
    use_logp = any([iscomplex(pscale) for pscale in pscales])
  rtype = 0. if use_logp else 1.
  for _pscale in pscales:
    pscale = eval_pscale(_pscale)
    if use_logp:
      rtype += logp_offs(pscale)
    else:
      rtype *= prob_coef(pscale)
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
  """ Applies product rule to all probability arrays in arrays with optional
  keywords:

  'pscales': a list of corresponding pscales (defaulting to pscales all 1)
  'use_logp': a boolean flag to return log probabilities (defaults to true if
              any value in pscales is complex)

  :return: joint_probility, pscale (two element tuple)
  """
  kwds = dict(kwds)
  pscales = kwds.get('pscales', [1.] * len(args))
  use_logp = kwds.get('use_logp', any([iscomplex(_pscale) for _pscale in pscales]))
  ppscale = prod_pscale(pscales, use_logp)
  pscale = kwds.get('pscale', ppscale)
  n_args = len(args)
  assert len(pscales) == n_args, \
      "Input pscales length {} incommensurate with number of arguments {}".\
      format(len(pscales), n_args)
  
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
  if use_logp != iscomplex(ppscale):
    ppscale = complex(np.log(ppscale), 0.) if use_logp else float(np.exp(ppscale))
  elif use_logp == iscomplex(pscale) and ppscale == pscale and \
      len(set([iscomplex(_pscale) for _pscale in pscales])) == 1:
    prob = _apply_prod(list(args))
    return prob, pscale

  # Otherwise exp/log before evaluating product
  probs = [None] * n_args
  for i, arg in enumerate(args):
    p_log = iscomplex(pscales[i])
    probs[i] = args[i]
    if use_logp:
      if not p_log:
        probs[i] = log_prob(probs[i])
    else:
      if p_log:
        probs[i] = exp_logp(probs[i])
  prob = _apply_prod(probs)
  if use_logp != iscomplex(pscale):
    prob = rescale(prob, ppscale, pscales)

  return prob, pscale

#-------------------------------------------------------------------------------
def div_prob(dividend, divisor, *args, pscale=None):
  """ Safely divides two probability arrays.

  :param dividend: numerator
  :param divisor: denominator
  :param *args: pscales for divident, divisor (defaults to 1. in each case).
  :param pscale: output pscale (defaults to result first pscale).
  """
  pscales = [None, None]
  if len(args):
    assert len(args) == 2, "Both pscales must be specified if at all"
    pscales[0] = eval_pscale(args[0])
    pscales[1] = eval_pscale(args[1])
    pscale = pscale or pscales[0]
  dividend = rescale(dividend, pscales[0], None)
  divisor = rescale(divisor, pscales[1], None)
  quotient = dividend / np.maximum(NEARLY_POSITIVE_ZERO, divisor)
  return rescale(quotient, None, pscale)
  
#-------------------------------------------------------------------------------
