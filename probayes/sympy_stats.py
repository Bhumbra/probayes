"""
Sympy distributions seem to tend to a functional rather than object interface.
In order to have access to the distribution class instances, a wrapper is
needed. This module provides such a wrapper with a common interface for
continuous and discrete statistical distributions.
"""
import collections
import inspect
import functools
import operator
import sympy.stats
#-------------------------------------------------------------------------------
"""
Build [DISCRETE, CONTINUOUS] dictionary in the form:
    {key: {'obj': class object,
           'fun: distribution function}}
"""
SYMPY_STATS_DISTRIBUTION_TYPES = [sympy.stats.crv_types, sympy.stats.drv_types]
SYMPY_STATS_DISTRIBUTION_DICTIONARIES = [collections.OrderedDict() for _ in
    range(len(SYMPY_STATS_DISTRIBUTION_TYPES))]

for i, dist_types in enumerate(SYMPY_STATS_DISTRIBUTION_TYPES):
  # Extract class-defined distributions
  mem_names = []
  mem_objs = []
  for mem_name, mem_obj in inspect.getmembers(dist_types):
    mem_names.append(mem_name)
    mem_objs.append(mem_obj)
  for mem_name, mem_obj in zip(mem_names, mem_objs):
    if 'Distribution' in mem_name:
      dist_name = mem_name.replace('Distribution', '')
      if dist_name in mem_names:
        SYMPY_STATS_DISTRIBUTION_DICTIONARIES[i].update(
            {dist_name: {'obj': mem_obj}})
  for mem_name, mem_obj in zip(mem_names, mem_objs):
    if mem_name in SYMPY_STATS_DISTRIBUTION_DICTIONARIES[i].keys():
      SYMPY_STATS_DISTRIBUTION_DICTIONARIES[i][mem_name].update(
          {'fun': mem_obj})

#-------------------------------------------------------------------------------
""" Combine discrete and continuous distributions into a common interface """
class SympyStats:

  def __init__(self, name, obj, fun):
    self._name = name
    self._obj = obj
    self._fun = fun

#-------------------------------------------------------------------------------
  def __repr__(self):
    return self._name

#-------------------------------------------------------------------------------
  def __getitem__(self, arg=None):
    """ Member getter:

    :param arg: if None or slice operator, returns distribution class object
                if empty string: returns distribution name
                if empty list, tuple or dict: returns distribution function
    """
    if arg is None or arg == slice(None):
      return self._obj
    assert isinstance(arg, (str, list, tuple, dict)) and not len(arg),\
        "Non-none get item argument must be empty str, list, tuple or dict"
    if isinstance(arg, str): 
      return self._name
    return self._fun

#-------------------------------------------------------------------------------
SYMPY_STATS_DISTRIBUTION_NAMES = []
SYMPY_STATS_DISTRIBUTIONS = []

for stats_dist_dict in SYMPY_STATS_DISTRIBUTION_DICTIONARIES:
  for dist_name, dist_dict in stats_dist_dict.items():
    SYMPY_STATS_DISTRIBUTION_NAMES.append(dist_name)
    SYMPY_STATS_DISTRIBUTIONS.append(SympyStats(dist_name,
                                                dist_dict['obj'],
                                                dist_dict['fun']))
SYMPY_STATS_DISTRIBUTION_NAMEDTUPLE = collections.namedtuple(
    'SympyStats', SYMPY_STATS_DISTRIBUTION_NAMES)
SYMPY_DISTS = SYMPY_STATS_DISTRIBUTIONS
SYMPY_STATS = SYMPY_STATS_DISTRIBUTION_NAMEDTUPLE(
    *tuple(SYMPY_STATS_DISTRIBUTIONS))

#-------------------------------------------------------------------------------
