# Utility module for SD objects

import collections
from probayes.pscales import iscomplex, prod_rule, prod_pscale

#-------------------------------------------------------------------------------
def desuffix(values, suffix="'"):
  assert isinstance(values, dict), \
      "Input must be a dictionary, not {}".format(type(values))
  suffix_found = any([key[-1] == suffix for key in values.keys()])
  vals = collections.OrderedDict()
  if not suffix_found:
    vals.update({values})
    return vals
  for key, val in values.items():
    vals_key = key if key[-1] != suffix else key[:-1]
    assert vals_key not in vals, "Repeated key: {}".format(vals_key)
    vals.update({vals_key: val})
  return vals
  
#-------------------------------------------------------------------------------
def get_suffixed(values, unsuffix=True, suffix="'"):
  assert isinstance(values, dict), \
      "Input must be a dictionary, not {}".format(type(values))
  vals = collections.OrderedDict()
  for key, val in values.items():
    if key[-1] == suffix:
      vals_key = key[:-1] if unsuffix else key
      vals.update({vals_key: val})
  return vals

#-------------------------------------------------------------------------------
def sd_prod_rule(*args, dims, sds, pscale=None):
  """ Returns the probability product treating all SDs as independent.
  Values (=args[0]) are keyed by RV name as are dimensions dims, and sds is a 
  list of SDs.
  """
  values = args[0]
  pscales = [sd.ret_pscale() for sd in sds]
  pscale = pscale or prod_pscale(pscales)
  use_logs = iscomplex(pscale)
  probs = [None] * len(sds)
  for i, sd in enumerate(sds):
    keys = sd.ret_keys()
    vals = {key: values[key] for key in keys}
    subdims = {key: dims[key] for key in keys}
    probs[i] = sd.eval_prob(vals, subdims)
  
  prob, pscale = prod_rule(*tuple(probs),
                           pscales=pscales,
                           pscale=pscale)

  # This section below is there just to play nicely with conditionals
  if len(args) > 1:
    if use_logs:
      prob = rescale(prob, pscale, 0.j)
    else:
      prob = rescale(prob, pscale, 1.)
    for arg in args[1:]:
      if use_logs:
        offs, _ = sd_prod_rule(arg, sds=sds, pscale=0.j)
        prob = prob + offs
      else:
        coef, _ = sd_prod_rule(arg, sds=sds, pscale=1.)
        prob = prob * coef
    if use_logs:
      prob = prob / float(len(args))
      prob = rescale(prob, 0.j, pscale)
    else:
      prob = prob ** (1. / float(len(args)))
      prob = rescale(prob, 1., pscale)
  return prob, pscale

#-------------------------------------------------------------------------------
