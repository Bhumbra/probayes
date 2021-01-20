# Module to test Expressions

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy
import probayes as pb

#-------------------------------------------------------------------------------
VALUE_TESTS = [(4,), ((2, 3),), ({'key_0': 0, 'key_1': 1})]
FUNCT_TESTS = [(np.negative, 2.), (np.reciprocal, 2.), 
    ((np.log, np.exp), 2.), ({'key_0': np.exp, 'key_1': np.log}, 2.)]
LAMBDA_TESTS = [(lambda x: np.negative(x), 2.), (lambda x: np.reciprocal(x), 2.)]
SYMPY_TESTS = [((sympy.log, sympy.exp), 2.), (sympy.log, 2)]

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
@pytest.mark.parametrize("func, vals", LAMBDA_TESTS)
def test_lambdas(func, vals):
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
@pytest.mark.parametrize("func, vals", SYMPY_TESTS)
def test_sympys(func, vals):
  x = sympy.Symbol('x')
  if isinstance(func, (list, tuple)):
    exprs = [fun(x) for fun in func]
    expr = pb.Expression(tuple(exprs))
  else:
    expr = pb.Expression(func(x), invertible=True)
  forward = expr[0](vals)
  reverse = expr[1](forward)
  assert forward != reverse, "No transformation"
  assert np.isclose(vals, reverse), "Reversal did not reverse"

#-------------------------------------------------------------------------------
