"""
A Constants module. 
"""
import numpy as np

#-------------------------------------------------------------------------------
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
