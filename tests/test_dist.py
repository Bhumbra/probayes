# Test for identical results for equivalent Sympy/Scipy Distributions

#-------------------------------------------------------------------------------
import pytest
import numpy as np
import scipy.stats
import sympy.stats
import probayes as pb

#-------------------------------------------------------------------------------
DIST_TESTS = [
    (scipy.stats.norm, {'loc': 1., 'scale': 0.5}, sympy.stats.Normal, (1., 0.5), 
     float, [-1, 3], {1000}),
    (scipy.stats.poisson, {'mu': 2.5}, sympy.stats.Poisson, (2.5,), 
     int, [0, 10], None),
    (scipy.stats.beta, {'a': 0.5, 'b':0.5}, sympy.stats.Beta, (0.5, 0.5), 
     float, [(0.,), (1.,)], {10}),
             ]
#-------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist_scipy, args_scipy, dist_sympy, args_sympy, vtype, vset, values", 
    DIST_TESTS)
def test_dists(dist_scipy, args_scipy, dist_sympy, args_sympy, 
               vtype, vset, values):

  # Scipy distribution
  x = pb.RV('x', vtype=vtype, vset=vset)
  if isinstance(args_scipy, dict):
    x.set_prob(dist_scipy, **args_scipy)
  else:
    x.set_prob(dist_scipy, *args_scipy)
  px = x({x: values})

  # Sympy distribution
  y = pb.RV('y', vtype=vtype, vset=vset)
  y.set_prob(dist_sympy(y[:], *args_sympy))
  py = y({y: values})

  assert np.allclose(px.prob, py.prob), \
      "Inconsistent results comparing {} vs {}".format(
          dist_scipy, dist_sympy)

#-------------------------------------------------------------------------------
