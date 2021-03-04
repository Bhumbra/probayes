# Module to test Functional

#-------------------------------------------------------------------------------
import pytest
import sympy
import math
import numpy as np
import networkx as nx
import probayes as pb

#-------------------------------------------------------------------------------
FUNC_TESTS = [((sympy.log, sympy.exp), (2., math.log(2)))]

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("func, vals", FUNC_TESTS)
def test_func(func, vals):
  x = sympy.Symbol('x')
  y = sympy.Symbol('y')
  a = sympy.Symbol('a')
  b = sympy.Symbol('b')
  xy = nx.OrderedGraph()
  ab = nx.OrderedGraph()
  xy.add_node(x)
  xy.add_node(y)
  ab.add_node(a)
  ab.add_node(b)
  functional = pb.Functional(ab, xy)
  functional.add_map(a, func[0](x))
  functional.add_map(b, func[1](y))
  eval_a = functional[a](vals[0])
  eval_b = functional[b](vals[1])
  import pdb; pdb.set_trace()
  assert np.isclose(eval_a, vals[1])
  assert np.isclose(eval_b, vals[0])

#-------------------------------------------------------------------------------
