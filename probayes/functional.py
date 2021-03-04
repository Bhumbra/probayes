""" Provides Functional to contain multiple expressions """

import collections
import functools
from probayes.functional_utils import NX_UNDIRECTED_GRAPH, collate_vertices,\
                                      parse_identifiers
from probayes.expression import Expression

#-------------------------------------------------------------------------------
class Functional:
  '''A Functional is container for multiple expressions that provides a
  deterministic mapping between one graph of variables and another.
  '''
  _out  = None     # Output graph
  _out_dict = None # Output vertices
  _out_set = None  # Set of output vertices _inp  = None     # Input graph
  _inp_dict = None # Input vertices
  _inp_set = None  # Set of input vertices
  _args = None     # Default args
  _kwds = None     # Default kwds
  _maps = None     # Mapping linking inputs to outputs
  _exprs = None    # Expression instances with same keys as maps
  _isiconic = None # Flag for being iconic
  _partials = None # Dict of partial functions

#-------------------------------------------------------------------------------
  def __init__(self, out=None, inp=None, *args, **kwds):
    """ Initialises output and input objects for the functional, where out and
    inp are undirected graphs made of vertices comprising object instances with
    members node.name comprising an identifier string."""
    self.out = out
    self.inp = inp or self._out
    self._args = tuple(args) 
    self._kwds = dict(kwds)

#-------------------------------------------------------------------------------
  @property
  def out(self):
    """ Output graph """
    return self._out

  @property
  def out_dict(self):
    """ Output dictionary """
    return self._out_dict

  @property
  def out_set(self):
    """ Output set """
    return self._out_set
  
  @out.setter
  def out(self, out):
    self._out = out
    self._out_dict = collate_vertices(self._out)
    self._out_set = frozenset(self._out_dict.keys())

#-------------------------------------------------------------------------------
  @property
  def inp(self):
    """ Input graph """
    return self._inp
  
  @property
  def inp_dict(self):
    """ Input dictionary """
    return self._inp_dict

  @property
  def inp_set(self):
    """ Input set """
    return self._inp_set
  
  @inp.setter
  def inp(self, inp):
    self._inp = inp
    self._inp_dict = collate_vertices(self._inp)
    self._inp_set = frozenset(self._inp_dict.keys())

#-------------------------------------------------------------------------------
  @property
  def maps(self):
    """ Input/output mappings dictionary """
    return self._maps

  @property
  def exprs(self):
    """ Expressions dictionary """
    return self._exprs
  
  @property
  def isiconic(self):
    """ Iconic flag """
    return self._iconic

  def add_map(self, spec, expr, *args, **kwds):
    """ Adds an expression map for a specificied inp/out relationship where
    expr, *args, **kwds are the inputs to the corresponding Expression instance
    (see Expression) and spec specifies the dependent variable(s) (for iconic
    expressions) or (for non-iconic) both dependent and variables.
    """

    # Initialise maps and exprs if not set
    if self._maps is None:
      self._maps = collections.OrderedDict()
    if self._exprs is None:
      self._exprs = collections.OrderedDict()

    # Update maps
    spec_out = None
    spec_inp = None
    if isinstance(spec, dict):
      spec_out = parse_identifiers(tuple(spec.keys()))
      spec_inp = parse_identifiers(tuple(spec.values()))
    else:
      spec_out = parse_identifiers(spec)

    if self._maps:
      assert spec_out not in self._maps.keys(), \
          "Output mapping for {} already previously set".format(spec_out)
    self._maps.update({spec_out: spec_inp})

    # Update exprs and set isiconic flag if not previously set
    self._exprs.update({spec_out: Expression(expr, *args, **kwds)})
    if self._isiconic is None:
      self._isiconic = self._exprs[spec_out].isiconic
    else:
       assert self._isiconic == self._exprs[spec_out].isiconic, \
           "Cannot mix iconic and non-iconic expressions within functional"

    # Detect inputs for iconics if not specified
    if spec_inp is None:
      assert self._isiconic, \
          "Input specification mandatory for non-iconic functionals"
      spec_inp = tuple(self._exprs[spec_out].symbols.keys())
      spec_inp = parse_identifiers(spec_inp)
      self._maps[spec_out] = spec_inp

    # Detect subsets are covered
    assert spec_out.issubset(self._out_set), \
        "Specification output must be a subset of output graph"
    assert spec_inp.issubset(self._inp_set), \
        "Specification input must be a subset of input graph"

    self._set_partials()

#-------------------------------------------------------------------------------
  @property
  def partials(self):
    return self._partials

  def _set_partials(self):
    # Protected function to update partial function dictionary of calls
    self._partials = collections.OrderedDict()
    for key, val in self._exprs.items():
      call = functools.partial(val.__call__, *self._args, **self._kwds)
      self._partials.update({key: call})

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
   """ Returns the partial according to arg  """ 
   key = parse_identifiers(arg)
   return self._partials[key]

#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Print representation """
    if self._isiconic:
      return object.__repr__(self)+ ": '{}'".format(self._exprs)
    if self._maps is None:
      return object.__repr__(self) 
    return object.__repr__(self)+ ": '{}'".format(self._maps)

#-------------------------------------------------------------------------------
