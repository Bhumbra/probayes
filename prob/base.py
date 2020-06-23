# A probability scaler

#-------------------------------------------------------------------------------
import numpy as np

#-------------------------------------------------------------------------------
# Single precision limits
NEARLY_POSITIVE_ZERO = 1.175494e-38
NEARLY_NEGATIVE_INF = -3.4028236e38
NEARLY_POSITIVE_INF =  3.4028236e38
LOG_NEARLY_POSITIVE_INF = np.log(NEARLY_POSITIVE_INF)
NUMPY_DTYPES = {
                 np.dtype('bool'): bool,
                 np.dtype('int'): int,
                 np.dtype('int32'): int,
                 np.dtype('int64'): int,
                 np.dtype('float'): float,
                 np.dtype('float32'): float,
                 np.dtype('float64'): float,
               }
NOMINAL_VSET = [False, True]

#-------------------------------------------------------------------------------
def nominal_prob(x, p):
  x, p = np.atleast_1d(x).astype(bool), float(p)
  prob = np.tile(1.-p, x.shape)
  prob[x] = p
  return prob

#-------------------------------------------------------------------------------
def log_prob(prob):
  logs = np.tile(NEARLY_NEGATIVE_INF, prob.shape)
  ok = prob >= NEARLY_POSITIVE_ZERO
  logs[ok] = np.log(prob[ok])
  return logs

#-------------------------------------------------------------------------------
def exp_logs(logs):
  prob = np.tile(NEARLY_POSITIVE_INF, logs.shape)
  ok = logs <= LOG_NEARLY_POSITIVE_INF
  prob[ok] = np.exp(logs[ok])
  return prob

#-------------------------------------------------------------------------------
def scale2logoffset(sc=None):
  if sc is None:
    return 0.
  if isinstance(sc, str):
    return float(sc)
  if sc <= 0.:
    return sc
  return np.log(sc)

#-------------------------------------------------------------------------------
class _P:

  # Protected
  _prsc = None

#-------------------------------------------------------------------------------
  def __init__(self, prsc=None):
    self.set_prsc(prsc)

#-------------------------------------------------------------------------------
  def set_prsc(self, prsc=None):
    """
    Positive denotes a normalising coefficient.
    Zero or negative denotes a log scale that must be added
    """
    self._prsc = prsc
    if type(self._prsc) is float and self._prsc <= 0.:
      self._prsc = str(abs(self._prsc))
    return self._prsc

#-------------------------------------------------------------------------------
  def ret_prsc(self):
    return self._prsc

#-------------------------------------------------------------------------------
  def rescale(self, probs, rsc=None):
    psc = self._prsc
    if type(psc) is float and psc <= 0.:
      psc = str(abs(psc))
    if type(rsc) is float and rsc <= 0.:
      rsc = str(abs(rsc))
    if psc == rsc:
      return probs
    
    plsc = isinstance(psc, str)
    rlsc = isinstance(rsc, str)

    # Support non-logarithmic conversion (maybe used to avoid logging zeros)
    if not plsc and not rlsc:
      coef = psc / rsc
      if coef == 1.:
        return prob
      else:
        return coef * prob

    # For floating point precision, perform other operations in log-space
    if not plsc: probs = log_probs(probs)
    offset = scale2logoffset(psc) - scale2logoffset(rsc)
    if offset != 0: probs = probs + offset
    if rlsc:
      return probs
    return exp_logs(probs)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class _V:

  # Protected
  _vset = None               # Variable set (array or 2-length tuple range)
  _vfun = None               # 2-length tuple of mutually inverting functions

  # Private
  __vtype = None
  __callable = None          # Flag to denote if prob is callable

#-------------------------------------------------------------------------------
  def __init__(self, name, 
                     vset=None, 
                     vfun=None,):
    self.set_vset(name, vset)

#-------------------------------------------------------------------------------
  def set_vset(self, vset=None, vfun=None):

    # Default vset to nominal, and convert a set,list to an array
    if vset is None: vset = NOMINAL_VSET
    self._vset = vset
    if isinstance(self._vset, (set, list)):
      self._vset = np.atleast_1d(self._vset) if isinstance(self._vset, list) \
                   else np.sort(np.atleast_1d(self._vset))
    elif isinstance(self._vset, tuple):
      assert len(self._vset) == 2,\
          "Tuple vset must be of length 2, not {}".format(len(self._vset))
    elif isinstance(self._vset, range):
      self._vset = np.arange(self._vset.start, self._vset,stop, self._vset.step,
                             dtype=int)

    # Set and check vfun
    self._vfun = vfun
    if self._vfun is not None:
      message = "Tuple setting of vfun be a two-sized tuple of callable functions"
      assert isinstance(self._vfun, tuple), message
      assert len(self._vfun) == 2, message
      assert callable(self._vfun[0]), message
      assert callable(self._vfun[1]), message

    # Detect vtype if not specified
    if self._vfun:
      self._vtype = float
    elif isinstance(self._vset, (bool, int, float)):
      self._vtype = type(self._vset)
    elif isinstance(self._vset, tuple):
      self._vtype = float
    elif isinstance(self._vset, np.ndarray):
      self._vtype = NUMPY_DTYPES.get(self._vset.dtype, None)
    else:
      self._vtype = None
    return self.ret_vtype()

#-------------------------------------------------------------------------------
  def ret_vtype(self):
    return self._vtype

#-------------------------------------------------------------------------------
  def ret_vfun(self):
    return self._vfun

#-------------------------------------------------------------------------------
  def get_bounds(self, use_vfun=None):
    if use_vfun is None:
      use_vfun = self._vfun is not None
    if self._vset is None:
      return None
    if not use_vfun or self._vfun is None \
      or self._vtype is not float or not isinstance(self._vset, tuple):
      return np.min(self._vset), np.max(self._vset)
    vset = self._vfun[0](np.array(self._vset, dtype=float))
    return float(vset[0]), float(vset[1])

#-------------------------------------------------------------------------------
  def eval_samp(self, samples=None):

    # Convert arrays
    if isinstance(samples, (tuple, list, np.ndarray)):
      immutable = isinstance(samples, tuple)
      samples = np.atleast_1d(samples) if not self._vtype else \
                np.atleast_1d(samples).astype(self._vtype)
      if not immutable and self._vfun:
        samples = self._vfun[0](samples)
      
    # Integer samples n values
    elif samples is None or type(samples) is int:
      if samples is None:
        assert not isinstance(self._vset, tuple),\
            "Samples must be specified for variable set: {}".format(self._vset)
        samples = len(self._vset)

      # Non-continuous support sets
      if not isinstance(self._vset, tuple):
        divisor = len(self._vset)
        if samples >= 0:
          indices = np.arange(samples, dtype=int) % divisor
        else:
          indices = np.random.permutation(-samples, dtype=int) % divisor
        samples = self._vset[indices]

      else:
        vset = np.array(self._vset, dtype = float)
        if self._vfun is not None:
          vset = self._vfun[0](vset)
        assert np.all(np.isfinite(vset)), \
            "Cannot evaluate {} samples for bounds: {}".format(samples, vset)
        lo, hi = self.get_bounds()
        if samples == 1:
          samples = np.atleast_1d(0.5 * (lo+hi))
        elif samples >= 0:
          samples = np.linspace(lo, hi, samples)
        else:
          samples = np.sort(np.random.uniform(lo, hi, size=-samples))

    else:
      raise TypeError("Ambiguous samples type: ".format(type(samples)))
    return samples
   
#-------------------------------------------------------------------------------
