# Module to test Expressions

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy as sy
import probayes as pb

#-------------------------------------------------------------------------------
SCIPY_PROB_TESTS = [
    (scipy.stats.norm, np.linspace(-1, 3, 1000), {'loc': 1., 'scale': 0.5}),
    (scipy.stats.binom, np.arange(10, dtype=int), {'n': 9, 'p': 0.5})
              ]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("dist, values, kwds", SCIPY_PROB_TESTS)
def test_prob_scipy(dist, values, kwds):
  expr = pb.Prob(dist, **kwds)
  prob = expr['prob'](values)
  logp = expr['logp'](values)
  assert np.allclose(prob, np.exp(logp)), \
      "{} probabilities not exponentials of associated logpdf".format(dist)
  cump = expr.pfun[0](values)
  invc = expr.pfun[-1](cump)
  assert np.allclose(values, invc), \
      "{} CDF and inverse do not reciprote".format(dist)

#-------------------------------------------------------------------------------
SCIPY_SAMP_TESTS = [
    (scipy.stats.norm, 100, {'loc': 1., 'scale': 0.5}),
              ]
@pytest.mark.parametrize("dist, size, kwds", SCIPY_SAMP_TESTS)
def test_samp_scipy(dist, size, kwds):
  expr = pb.Prob(dist, **kwds)
  samp = expr.sfun(size=size)
  assert len(samp) == size, "Mismatch in samples and size specification"

#-------------------------------------------------------------------------------

