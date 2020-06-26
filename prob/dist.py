# A distribution is triple comprising variable names, their values (vals), and respective probabilities.

#-------------------------------------------------------------------------------
import collections
import numpy as np
from prob.prob import rescale

STR_FLOAT_ZERO = str(float(0))

#-------------------------------------------------------------------------------
class Dist:

  # Public
  name = None    # Name of distribution
  vals = None    # Dictionary of values
  prob = None    # Numpy array

  # Protected
  _ptype = None  # Same convention as _Prob
  _scalar = None # if all scalars

#-------------------------------------------------------------------------------
  def __init__(self, name=None, vals=None, prob=None, ptype=None):
    self.set_name(name)
    self.set_vals(vals)
    self.set_prob(prob, ptype)

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    self.name = name

#-------------------------------------------------------------------------------
  def set_vals(self, vals=None):
    self.vals = vals
    self._scalar = None
    if self.vals is not None:
      assert isinstance(self.vals, dict), \
          "Dist vals must be variable-name keyed dictionary but given: {}". \
          format(self.vals)
      self._scalar = True
      for val in self.vals.values():
        if not np.isscalar(val):
          self._scalar = False
        break
    return self._scalar

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, ptype=None):
    self.prob = prob
    self._ptype = ptype
    if isinstance(self.prob, np.ndarray) and self._scalar:
      self.prob = float(self.prob)
    if self._ptype in ['log', 'ln']:
      self._ptype = STR_FLOAT_ZERO
    return self._ptype

#-------------------------------------------------------------------------------
  def ret_scalar(self):
    return self._scalar

#-------------------------------------------------------------------------------
  def ret_ptype(self):
    return self._ptype

#-------------------------------------------------------------------------------
  def rescale(self, ptype=None):
    self.set_prob(rescale(self.prob, self._ptype, ptype), ptype)
    return self.prob

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    return dist_prod(*tuple([self, other]))

#-------------------------------------------------------------------------------
  def __repr__(self):
    prefix = 'logp' if isinstance(self._ptype, str) else 'p'
    return super().__repr__() + ": " + prefix + "(" + self.name + ") [vals,prob]"

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def parse_name(name):
  """
  Returns a dictionary of two lists:
  {
   'marg': list of marginal rv names    (removing equals and beyond)
   'cond': list of conditional rv names (removing equals and beyond)
  }
  """
  marg_str = name
  cond_str = ''
  if '|' in marg_str:
    split_str = name.split('|')
    assert len(split_str) == 2, "Ambiguous name: {}".format(name)
    marg_str, cond_str = split_str
  marg = []
  cond = []
  if len(marg_str):
    marg = marg_str.split(',') if ',' in marg_str else [marg_str]
  if len(cond_str):
    cond = cond_str.split(',') if ',' in cond_str else [cond_str]
  return {'marg': marg, 'cond': cond}

#-------------------------------------------------------------------------------
def str2key(string):
  if isinstance(string, str):
    k = string.find('=')
    if k > 0:
      return string[:k]
    return string
  return [str2key(element) for element in string]

#-------------------------------------------------------------------------------
def marg_prod(*args, check=True):
  """ 
  Returns the marginal product of single-variable distributions with identical 
  ptypes. Assert checks cans be bypassed if set to zero
  """
  names = [parse_name(arg.name) for arg in args]
  if not names:
    return None
  ptypes = [arg.ret_ptype() for arg in args]
  marg_names = []
  for name in names: marg_names.extend(name['marg'])
  cond_names = names[0]['cond']
  ptype = args[0].ret_ptype()
  track_ptype = True
  if ptype == STR_FLOAT_ZERO:
    track_ptype = False
    for _ptype in ptypes[1:]:
      if _ptype != STR_FLOAT_ZERO:
        track_ptype = True
        break
  elif ptype is None or ptype == 1.:
    track_ptype = False
    for _ptype in ptypes[1:]:
      if _ptype != None or _ptype == 1:
        track_ptype = True
        break
  if check:
    assert len(marg_names) == len(set(marg_names)),\
      "Non-unique marginal variable name found in {}".format(marg_names)
    for name in names:
      assert cond_names == name['cond'], "Non-identical conditional variables"
    if len(cond_names):
       for marg_name in marg_names:
         assert marg_name not in cond_names,\
           "Overlap between variable {} found within condition {}".format(
               marg_name, cond_names)
    if isinstance(ptype, str):
      assert all([isinstance(_ptype, str) for _ptype in ptypes])
    else:
      assert all([_ptype is None or type(_ptype) is float for _ptype in ptypes])
  prod_vals = collections.OrderedDict()
  [prod_vals.update(arg.vals) for arg in args]
  prod_marg_name = ','.join(marg_names)
  prod_cond_name = ','.join(cond_names)
  prod_name = '|'.join([prod_marg_name, prod_cond_name])
  prod_keys = str2key(marg_names) + str2key(cond_names)

  # Bypass comprehensive approach for scalars
  if not track_ptype and not check: # bypass long-winded approach for scalars
    if all([arg.ret_scalar() for arg in args]):
      probs = [arg.prob for arg in args]
      prob = float(sum(probs)) if isinstance(ptype0, str) \
             else float(np.prod(probs))
      return Dist(prod_name, prod_vals, prob, ptype0)

  # Reshape values and marginal probabilities
  use_logs = isinstance(ptype, str)
  run_ptype = 0. if use_logs else 1.
  ndim_ones = np.ones(len(prod_keys), dtype=int)
  probs = [None] * len(args)
  for i, arg in enumerate(args):
    vals, prob, marg_names, ptype = arg.vals, arg.prob, names[i]['marg'], ptypes[i]
    reshape_prob = np.copy(ndim_ones)
    for key in vals.keys():
      reshape_vals = np.copy(ndim_ones)
      index = prod_keys.index(key)
      reshape_vals[index] = vals[key].size
      reshape_prob[index] = vals[key].size
      prod_vals.update({key: vals[key].reshape(reshape_vals)})
    if isinstance(prob, np.ndarray):
      prob = prob.reshape(reshape_prob)
    if not track_ptype:
      probs[i] = prob
    else:
      if not use_logs:
        if ptype is not None and ptype != 1.:
          run_ptype *= ptype
        probs[i] = prob
      else:
        logprob = ptype is None
        if isinstance(ptype, str):
          ptype = float(ptype)
          logprob = False
        elif type(ptype) is float:
          ptype = np.log(ptype)
          logprob = True
        run_ptype += ptype
        probs[i] = log_prob(prob) if logprob else np.copy(prob)

  # Evaluate joint probability
  prob = None
  for i in range(len(args)):
    if prob is None:
      prob = probs[i]
    elif use_logs:
      prob = prob + probs[i]
    else:
      prob = prob * probs[i]
  if track_ptype:
    if use_logs:
      if run_ptype != 0.:
        prob = prob - run_ptype
      prob = exp_logs(prob)
    else:
      if run_ptype != 1. and run_ptype != 0.:
        prob = prob / run_ptype

  return Dist(prod_name, prod_vals, prob, ptype)
 
#-------------------------------------------------------------------------------
def dist_prod(*args, **kwds):
  """ Multiplies two or more distributions subject to the following:
  They must not share the same marginal variables. If ptype is
  specified as a keyword, the resulting product distribution will
  conform to that ptype.
  """
  return marg_prod(*args, **kwds)

#-------------------------------------------------------------------------------
