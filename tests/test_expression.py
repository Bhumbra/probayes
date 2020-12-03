# Module to test Expressions

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy as sy
import probayes as pb

#-------------------------------------------------------------------------------
LOG_TESTS = [(math.exp(1.),1.)]
INC_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]
SYMBOL_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]
VALUE_TESTS = [(4,), ((2, 3),), ({'key_0': 0, 'key_1': 1})]
FUNCT_TESTS = [(np.negative, 2.), (np.reciprocal, 2.), 
    ((np.log, np.exp), 2.), ({'key_0': np.exp, 'key_1': np.log}, 2.)]
SCIPY_TESTS = [
    (scipy.stats.norm, np.linspace(-1, 3, 1000), {'loc': 1., 'scale': 0.5}),
    (scipy.stats.binom, np.arange(10, dtype=int), {'n': 9, 'p': 0.5})
              ]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", LOG_TESTS)
def test_log(inp, out):
  x = pb.Variable('x', vtype=float)
  expr = pb.Expr(sy.log(x[:]))
  f_x = expr({'x': inp})
  close = np.isclose(f_x, out)
  if isinstance(inp, np.ndarray):
    assert np.all(close), "Output values not as expected"
  else:
    assert close, "Output value {} not as expected {}".format(
        f_x, out)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", INC_TESTS)
def test_inc(inp, out):
  x = pb.Variable('x', vtype=float)
  expr = pb.Expr(x[:]+1)
  f_x = expr({'x': inp})
  close = np.isclose(f_x, out)
  if isinstance(inp, np.ndarray):
    assert np.all(close), "Output values not as expected"
  else:
    assert close, "Output value {} not as expected {}".format(
        f_x, out)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("values", VALUE_TESTS)
def test_values(values):
  expr = pb.Expression(values)
  if not isinstance(values, (dict, tuple)):
    assert expr() == values, "Scalar mismatch for non-multiple"
  elif isinstance(values, tuple):
    for i, val in enumerate(values):
      assert expr[i]() == val, "Scalar mismatch for tuple"
  else:
    for key, val in values.items():
      assert expr[key]() == val, "Scalar mismatch for dictionary"

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("func, vals", FUNCT_TESTS)
def test_functs(func, vals):
  expr = pb.Expression(func)
  if not isinstance(func, (dict, tuple)):
    forward = expr(vals)
    reverse = expr(forward)
  elif isinstance(func, tuple):
    forward = expr[0](vals)
    reverse = expr[1](forward)
  elif isinstance(func, dict):
    keys = list(func.keys())
    forward = expr[keys[0]](vals)
    reverse = expr[keys[1]](forward)
  assert forward != reverse, "No transformation"
  assert np.isclose(vals, reverse), "Reversal did not reverse"

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("dist, values, kwds", SCIPY_TESTS)
def test_scipy(dist, values, kwds):
  expr = pb.Expression(dist, **kwds)
  prob = expr['prob'](values)
  logp = expr['logp'](values)
  assert np.allclose(prob, np.exp(logp)), \
      "{} probabilities not exponentials of associated logpdf".format(dist)

#-------------------------------------------------------------------------------
