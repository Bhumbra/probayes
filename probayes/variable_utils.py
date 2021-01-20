'''
Utility module for variable.Variable() (and therefore many other objects)
'''
import collections

#-------------------------------------------------------------------------------
def parse_as_str_dict(*args, **kwds):
  """ Aggregrates all arguments in args (which must be dictionary objects) and
  keywords to output as a single ordereddict(), ensuring all keys are strings,
  replacing all non-string keys with arg.name values. Duplicate keys are not
  checked. Note 'remap' is a reserved keyword to allow key substituion using
  a dictionary. """

  remap = None if 'remap' not in kwds else kwds.pop('remap')
  kwds = collections.OrderedDict(kwds)
  if not args:
    return kwds
  args_dict = collections.OrderedDict()
  for arg in args:
    assert isinstance(arg, dict), \
        "Each argument type must be dict, not {}".format(type(arg))
    if not remap and all([isinstance(key, str) for key in arg.keys()]):
      args_dict.update(arg)
    else:
      for key, val in arg.items():
        if isinstance(key, str):
          if remap and key in remap:
            args_dict.update({remap[key]: val})
          else:
            args_dict.update({key: val})
        else:
          keys = key if isinstance(key, (list,tuple)) else [key]
          key_name = [None] * len(keys)
          for i, subkey in enumerate(keys):
            if isinstance(subkey, str):
              key_name[i] = subkey
            elif hasattr(subkey, 'name'):
              key_name[i] = subkey.name
              assert isinstance(key_name[i], str), \
                  "Non-string name attribute {} for key object: {}".format(
                      key_name[i], key)
            else:
              raise KeyError("Uninterpretable key to parse: {}".format(key))
          key_name = ','.join(key_name)
          if remap and key_name in remap:
            args_dict.update({remap[key_name]: val})
          else:
            args_dict.update({key_name: val})
  if kwds:
    args_dict.update(kwds)
  return args_dict

#-------------------------------------------------------------------------------
