""" Provides Functional to contain multiple expressions """

import collections
import networkx as nx
from probayes.expression import Expression

#-------------------------------------------------------------------------------
NX_UNDIRECTED_GRAPH = nx.OrderedGraph

#-------------------------------------------------------------------------------
def collate_vertices(graph=None):
  vertices = collections.OrderedDict()
  if not graph:
    return vertices
  if not isinstance(graph, NX_UNDIRECTED_GRAPH):
    raise TypeError("Expecting type {} but inputted {}".format(
      NX_UNDIRECTED_GRAPH, type(graph)))
  for vertex in list(graph.nodes):
    assert hasattr(vertex, 'name'), "Every vertex must include name attribute"
    vertex_name = vertex['name']
    assert isinstance(vertex_name, str), "Vertex name must be a string"
    assert vertex_name.isidentifier(), "Vertex name must be an identifier"
    vertices.update({vertex_name: vertex})
  return vertices

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Functional:
  '''A Functional is container for multiple expressions that provides a
  deterministic mapping between one set of variables and another.
  '''
  _out  = None # Output graph
  _outs = None # Output vertices
  _inp  = None # Input graph
  _inps = None # Input vertices
  _maps = None # Mapping expressions linking inputs to outputs

#-------------------------------------------------------------------------------
  def __init__(self, out=None, inp=None):
    """ Initialises output and input objects for the functional, where out and
    inp are undirected graphs made of vertices comprising object instances with
    members node.name comprising an identifier string."""
    self.out = out
    self.inp = inp or self._out

#-------------------------------------------------------------------------------
  @property
  def out(self):
    return self._out
  
  @out.setter
  def out(self, out):
    self._out = out
    self._outs = collate_vertices(self._out)

#-------------------------------------------------------------------------------
  @property
  def inp(self):
    return self._inp
  
  @out.setter
  def inp(self, inp):
    self._inp = inp
    self._inps = collate_vertices(self._inp)

#-------------------------------------------------------------------------------
