# Utility module for SC objects

import collections

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
    assert vals_key not in vals, "Repeated key: {}".format(val_key)
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

