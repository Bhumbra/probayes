# Module to test Expressions

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy.stats
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
SYMPY_PROB_TESTS = [
    (sympy.stats.Normal, np.linspace(-1, 3, 1000), (1., 0.5)),
              ]
#-------------------------------------------------------------------------------
@pytest.mark.parametrize("dist, values, args", SYMPY_PROB_TESTS)
def test_prob_sympy(dist, values, args):
  x = sympy.Symbol('x')
  expr = pb.Prob(dist(x, *args))
  prob = expr['prob'](values)
  logp = expr['logp'](values)
  assert len(prob) == len(values), "PDF not correct size"
  assert np.min(prob) >= 0., "Negative probabilities detected"
  assert np.allclose(prob, np.exp(logp)), \
      "{} probabilities not exponentials of associated logpdf".format(dist)

#-------------------------------------------------------------------------------
SYMPY_SAMP_TESTS = [
    (sympy.stats.Normal, 100, (1., 0.5)),
              ]
#-------------------------------------------------------------------------------
@pytest.mark.parametrize("dist, size, args", SYMPY_SAMP_TESTS)
def test_samp_sympy(dist, size, args):
  x = sympy.Symbol('x')
  expr = pb.Prob(dist(x, *args))
  samp = expr.sfun(size=size)
  assert len(samp) == size, "Mismatch in samples and size specification"

#-------------------------------------------------------------------------------
