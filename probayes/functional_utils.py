""" Provides utility functions for Functional """
import collections
import networkx as nx

#-------------------------------------------------------------------------------
NX_UNDIRECTED_GRAPH = nx.OrderedGraph

#-------------------------------------------------------------------------------
def collate_vertices(graph=None):
  """ Processes through all nodes of graph returning a dictionary in the form
  {node_name: node_instance}
  """
  vertices = collections.OrderedDict()
  if not graph:
    return vertices
  if not isinstance(graph, NX_UNDIRECTED_GRAPH):
    raise TypeError("Expecting type {} but inputted {}".format(
      NX_UNDIRECTED_GRAPH, type(graph)))
  for vertex in list(graph.nodes):
    assert hasattr(vertex, 'name'), "Every vertex must include name attribute"
    vertex_name = vertex.name
    assert isinstance(vertex_name, str), "Vertex name must be a string"
    assert vertex_name.isidentifier(), "Vertex name must be an identifier"
    vertices.update({vertex_name: vertex})
  return vertices

#-------------------------------------------------------------------------------
def parse_identifiers(identifiers):
  """ Reads identifiers, which may be a string or list/tuple/set of objects 
  instances with name instances as string, returning a frozen set of names.
  """
  if isinstance(identifiers, str):
    return frozenset(identifiers.split(','))
  if not isinstance(hashable, (list, tuple, set)):
    raise TypeError("Input identigiers be a string or tuple/list/set, not {}".\
        format(type(hashable)))
  keys = list(hashable)
  for i, key in enumerate(keys):
    if not isinstance(key, str):
      assert hasattr(key, 'name'), \
        "Each element in hashable tuple must be a string or have name attribute"
      key_name = key.name
      assert isinstance(key_name, str), \
        "Each non-string hashable tuple element must a string name attribute"
      keys[i] = key_name
  return frozenset(keys)

#-------------------------------------------------------------------------------
