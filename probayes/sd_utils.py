# Utility module for SD objects

import collections
from probayes.pscales import prod_rule

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
def arch_prob(arch, dims, **kwds):
  """ Returns the combined probability of for arch given values """
 
  values = dict(kwds)
  dimkeys = list(dims.keys())
  assert isinstance(arch, (tuple, list)), "Archictecture must be tuple or list"
  serial = isinstance(arch, list)
  probs = [None] * len(arch)
  for i, subarch in enumerate(arch):
    keyset = subarch.keylist
    vals = {key: values[key] for key in dimkeys if key in keyset}
    subdims = {key: dims[key] for key in dimkeys if key in keyset}
    probs[i] = subarch.eval_prob(vals, subdims)
  if serial:
    return probs[-1]
  pscales = [subarch.pscale for subarch in arch]
  prob, pscale = prod_rule(*tuple(probs), pscales=pscales)
  return prob

#-------------------------------------------------------------------------------
