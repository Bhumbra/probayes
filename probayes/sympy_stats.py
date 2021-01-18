"""
Sympy distributions seem to tend to a functional rather than object interface.
In order to have access to the distribution class instances, a wrapper is needed.  
"""
import inspect
import sympy.stats

#-------------------------------------------------------------------------------
def collate_sympy_distributions(
    dist_types=[sympy.stats.drv_types, sympy.stats.crv_types]):
  """ Returns sympy distributions as a dict: {dist_name: dist_func} """
  sympy_dists = {}
  for dist_type in dist_types:
    members = {}
    for mem_name, mem_obj in inspect.getmembers(dist_type):
      members.update({mem_name: mem_obj})
    for mem_name, mem_obj in members.items():
      if 'Distribution' in mem_name:
        dist_name = mem_name.replace('Distribution', '')
        if dist_name in members.keys():
          sympy_dists.update({dist_name: members[dist_name]})
  return sympy_dists

SYMPY_DISTRIBUTIONS = collate_sympy_distributions()
SYMPY_DISTS = list(SYMPY_DISTRIBUTIONS.values())

#-------------------------------------------------------------------------------
def sympy_obj_from_dist(dist):
  """ Attempts to return the object class instance for sympy distribution.
  :example:
  >>> import sympy
  >>> import sympy.stats
  >>> import probayes as pb
  >>> x = sympy.Symbol('x')
  >>> p_x = sympy.stats.Normal(x, mean=0, std=1.)
  >>> p_x_obj = pb.sympy_obj_from_dist(p_x)
  >>> print(p_x_obj.pdf)
  0.5*sqrt(2)*exp(-0.5*x**2)/sqrt(pi)
  """
  obj = dist
  if hasattr(obj, 'pdf'):
    return obj
  if not hasattr(obj, 'args'):
    return None
  objs = [sympy_obj_from_dist(arg) for arg in obj.args]
  if any(objs):
    for obj in objs:
      if obj:
        return obj
  return None

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

#-------------------------------------------------------------------------------
