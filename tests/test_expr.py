# Module to test Expr

#-------------------------------------------------------------------------------
import pytest
import sympy as sp
import numpy as np
import probayes as pb

#-------------------------------------------------------------------------------
VALUE_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", VALUE_TESTS)
def test_inc(inp, out):
  x = sp.Symbol('x')
  expr = pb.Expr(x+1)
  x_inc = expr({'x': inp})
  close = np.isclose(x_inc, out)
  if isinstance(inp, np.ndarray):
    assert np.all(close), "Output values not as expected"
  else:
    assert close, "Output value {} not as expected {}".format(
        x_inc, out)

#-------------------------------------------------------------------------------
