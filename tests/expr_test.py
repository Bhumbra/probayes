# Module to test Expressions

#-------------------------------------------------------------------------------
import numpy as np
import scipy.stats
import probayes as pb
import pytest

#-------------------------------------------------------------------------------
SCIPY_TESTS = [
    (scipy.stats.norm, np.linspace(-1, 3, 1000), {'loc': 1., 'scale': 0.5}),
    (scipy.stats.binom, np.arange(10, dtype=int), {'n': 9, 'p': 0.5})
              ]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("dist, values, kwds", SCIPY_TESTS)
def test_scipy(dist, values, kwds):
  print("Testing distribution {}".format(dist))
  expr = pb.Expression(dist, **kwds)
  prob = expr['prob'](values)
  logp = expr['logp'](values)
  assert np.allclose(prob, np.exp(logp)), \
      "{} probabilities not exponentials of associated logpdf".format(dist)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
  for test in SCIPY_TESTS:
    test_scipy(*test)

#-------------------------------------------------------------------------------
