# Module to test Variables

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy as sy
import probayes as pb
from probayes import NEARLY_POSITIVE_INF as inf
from probayes import NEARLY_POSITIVE_ZERO as zero

#-------------------------------------------------------------------------------
LOG_TESTS = [(math.exp(1.),1.)]
INC_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]

#-------------------------------------------------------------------------------
def ismatch(x, y):
  close = np.isclose(x, y)
  if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
    return np.all(close)
  else:
    return close

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", LOG_TESTS)
def test_log(inp, out):
  x = pb.Variable('x', vtype=float, vset=[zero, inf])
  x.set_ufun(sy.log(x[:]))
  output = x.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = x.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", INC_TESTS)
def test_inc(inp, out):
  x = pb.Variable('x', vtype=float, vset=[zero, inf])
  x.set_ufun(x[:]+1.)
  output = x.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = x.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)

#-------------------------------------------------------------------------------
