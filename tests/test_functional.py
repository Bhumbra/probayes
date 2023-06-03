# Module to test Functional

#-------------------------------------------------------------------------------
import pytest
import sympy
import math
import numpy as np
import networkx as nx
import probayes as pb

#-------------------------------------------------------------------------------
SYMBOL_TESTS = [((sympy.log, sympy.exp), (2., math.log(2)))]
VAR_TESTS = [((sympy.log, sympy.exp), (2., math.log(2)))]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("func, vals", SYMBOL_TESTS)
def test_symbol(func, vals):
  x = sympy.Symbol('x')
  y = sympy.Symbol('y')
  a = sympy.Symbol('a')
  b = sympy.Symbol('b')
  xy = nx.Graph()
  ab = nx.Graph()
  xy.add_node(x)
  xy.add_node(y)
  ab.add_node(a)
  ab.add_node(b)
  functional = pb.Functional(ab, xy)
  functional.add_func(a, func[0](x))
  functional.add_func(b, func[1](y))
  eval_a = functional[a](vals[0])
  eval_b = functional[b](vals[1])
  assert np.isclose(eval_a, vals[1])
  assert np.isclose(eval_b, vals[0])

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("func, vals", VAR_TESTS)
def test_var(func, vals):
  x = pb.Variable('x', vtype=float)
  y = pb.Variable('y', vtype=float)
  a = pb.Variable('a', vtype=float)
  b = pb.Variable('b', vtype=float)
  xy = x & y
  ab = a & b
  functional = pb.Functional(ab, xy)
  functional.add_func(a, func[0](x[:]))
  functional.add_func(b, func[1](y[:]))
  eval_a = functional[a](vals[0])
  eval_b = functional[b](vals[1])
  assert np.isclose(eval_a, vals[1])
  assert np.isclose(eval_b, vals[0])

#-------------------------------------------------------------------------------
