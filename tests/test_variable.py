# Module to test Variables

#-------------------------------------------------------------------------------
import pytest
import math
import numpy as np
import scipy.stats
import sympy
import probayes as pb
from probayes import NEARLY_POSITIVE_INF as inf
from probayes import NEARLY_POSITIVE_ZERO as zero

#-------------------------------------------------------------------------------
LOG_TESTS = [(math.exp(1.),1.)]
INC_TESTS = [(3,4), (np.linspace(-3, 3, 7), np.linspace(-2, 4, 7))]
RAN_LOG_TESTS = [([(1,), (100,)], {100}), ([1, 100], {-100})]

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
  # Sympy
  x = pb.Variable('x', vtype=float, vset=[zero, inf])
  x.set_ufun(sympy.log(~x))
  output = x.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = x.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)
  # Numpy
  y = pb.Variable('y', vtype=float, vset=[zero, inf])
  y.set_ufun((np.log, np.exp))
  output = y.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = y.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,out", INC_TESTS)
def test_inc(inp, out):
  # Sympy
  x = pb.Variable('x', vtype=float, vset=[zero, inf])
  x.set_ufun(x[:]+1.)
  output = x.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = x.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)
  # Numpy
  y = pb.Variable('y', vtype=float, vset=[zero, inf])
  y.set_ufun((lambda z: z+1, lambda z: z-1))
  output = y.ufun[0](inp)
  assert ismatch(out, output), \
      "Observed/expected match {}/{}".format(output, out)
  output = y.ufun[-1](output)
  assert ismatch(inp, output), \
      "Observed/expected match {}/{}".format(output, inp)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("ran,val", RAN_LOG_TESTS)
def test_ran(ran, val):
  # Sympy
  x = pb.Variable('x', vtype=float, vset=ran)
  x.set_ufun(sympy.log(x[:]))
  vals = x(val)[x.name]
  assert np.max(vals) <= x.vlims[1] and np.min(vals) >= x.vlims[0]
  # Numpy
  y = pb.Variable('y', vtype=float, vset=ran)
  y.set_ufun((np.log, np.exp))
  vals = y(val)[y.name]
  assert np.max(vals) <= x.vlims[1] and np.min(vals) >= x.vlims[0]
  # Delta
  steps = int(abs(list(val)[0]))
  value = np.mean(vals)
  y.set_delta([1], bound=True)
  vals = np.empty(steps, dtype=float)
  for i in range(steps):
    vals[i] = y.apply_delta(value)
    value = vals[i]
  assert np.max(vals) <= x.vlims[1] and np.min(vals) >= x.vlims[0]

#-------------------------------------------------------------------------------
