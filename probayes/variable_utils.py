'''
Utility module for variable.Variable() (and therefore many other objects)
'''
import collections

#-------------------------------------------------------------------------------
def parse_as_str_dict(*args, **kwds):
  """ Aggregrates all arguments in args (which must be dictionary objects) and
  keywords to output as a single ordereddict(), ensuring all keys are strings,
  replacing all non-string keys with arg.name values. Duplicate keys are not
  checked."""

  kwds = collections.OrderedDict(kwds)
  if not args:
    return kwds
  args_dict = collections.OrderedDict()
  for arg in args:
    assert isinstance(arg, dict), \
        "Each argument type must be dict, not {}".format(type(arg))
    if all([isinstance(key, str) for key in arg.keys()]):
      args_dict.update(arg)
    else:
      for key, val in arg.items():
        if isinstance(key, str):
          args_dict.update({key: val})
        else:
          assert hasattr(key, 'name'), \
              "Name attribute not found for key object: {}".format(key)
          assert isinstance(key.name, str), \
              "Non-string name attribute {} for key object: {}".format(
                  key.name, key)
          args_dict.update({key.name: val})
  if kwds:
    args_dict.update(kwds)
  return args_dict

#-------------------------------------------------------------------------------
