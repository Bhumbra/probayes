# Module to test Expr

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import sympy as sy
import probayes as pb

#-------------------------------------------------------------------------------
LOG_TESTS = [(math.exp(1.),1.)]
INC_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", LOG_TESTS)
def test_log(inp, out):
  for use_variable in False, True:
    if not use_variable:
      x = sy.Symbol('x')
      x_expr = pb.Expr(sy.log(x))
      output = x_expr({'x': inp})
    else:
      y = pb.Variable('y', vtype=float)
      y_expr = pb.Expr(sy.log(y[:]))
      output = y_expr({'y': inp})
    close = np.isclose(output, out)
    if isinstance(inp, np.ndarray):
      assert np.all(close), "Output values not as expected"
    else:
      assert close, "Output value {} not as expected {}".format(
          output, out)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", INC_TESTS)
def test_inc(inp, out):
  x = sy.Symbol('x')
  expr = pb.Expr(x+1)
  f_x = expr({'x': inp})
  close = np.isclose(f_x, out)
  if isinstance(inp, np.ndarray):
    assert np.all(close), "Output values not as expected"
  else:
    assert close, "Output value {} not as expected {}".format(
        f_x, out)

#-------------------------------------------------------------------------------
