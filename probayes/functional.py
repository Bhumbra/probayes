""" Provides Functional to contain multiple expressions """

import collections
import functools
from probayes.functional_utils import NX_UNDIRECTED_GRAPH, collate_vertices,\
                                      parse_identifiers
from probayes.expression import Expression

#-------------------------------------------------------------------------------
class Functional:
  '''A Functional is container for multiple functions that define a
  deterministic relationship between one graph of vertices and another.
  '''
  _out  = None     # Output graph
  _out_dict = None # Output vertices
  _out_set = None  # Set of output vertices _inp  = None     # Input graph
  _inp_dict = None # Input vertices
  _inp_set = None  # Set of input vertices
  _args = None     # Default args
  _kwds = None     # Default kwds
  _deps = None     # Dependences linking inputs to outputs
  _funcs = None    # Expression instances with same keys as deps
  _ndeps = None    # len(deps)
  _isiconic = None # Flag for being iconic
  _partials = None # Dict of partial functions

#-------------------------------------------------------------------------------
  def __init__(self, out=None, inp=None, *args, **kwds):
    """ Initialises output and input objects for the functional, where out and
    inp are undirected graphs made of vertices comprising object instances with
    members node.name comprising an identifier string."""
    self.out = out
    self.inp = inp or self._out
    self._ndeps = 0
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
  def deps(self):
    """ Input/output dependences dictionary """
    return self._deps

  def __len__(self):
    return self._ndeps

  def __bool__(self):
    if self._ndeps:
      return True
    return False

  @property
  def ndeps(self):
    return self._ndeps

  @property
  def funcs(self):
    """ Expressions dictionary """
    return self._funcs
  
  @property
  def isiconic(self):
    """ Iconic flag """
    return self._iconic

  def add_func(self, spec, func, *args, **kwds):
    """ Adds an function dependency for a specificied inp/out relationship where
    func, *args, **kwds are the inputs to the corresponding Expression instance
    (see Expression) and spec specifies the dependent variable(s) (for iconic
    expressions) or (for non-iconic) both dependent and variables.
    """

    # Initialise deps and funcs if not set
    if self._deps is None:
      self._deps = collections.OrderedDict()
    if self._funcs is None:
      self._funcs = collections.OrderedDict()

    # Update deps
    spec_out = None
    spec_inp = None
    if isinstance(spec, dict):
      spec_out = parse_identifiers(tuple(spec.keys()))
      spec_inp = parse_identifiers(tuple(spec.values()))
    else:
      spec_out = parse_identifiers(spec)

    if self._deps:
      assert spec_out not in self._deps.keys(), \
          "Output dependence for {} already previously set".format(spec_out)
    self._deps.update({spec_out: spec_inp})
    self._ndeps = len(self._deps)

    # Update funcs and set isiconic flag if not previously set
    self._funcs.update({spec_out: Expression(func, *args, **kwds)})
    if self._isiconic is None:
      self._isiconic = self._funcs[spec_out].isiconic
    else:
       assert self._isiconic == self._funcs[spec_out].isiconic, \
           "Cannot mix iconic and non-iconic expressions within functional"

    # If iconic, check for single spec_out and update dexpr if available
    if self._isiconic:
      assert not len(args) and not len(kwds), \
          "No optional arguments or keywords supported for iconic functions"
      assert len(spec_out) == 1, \
          "Only single output functions supported for iconic expressions"
      key = list(spec_out)[0]
      assert key in self._out_set, \
          "Key {} not found in amount output set {}".format(key, self._out_set)
      var = self._out_dict[key]
      if hasattr(var, 'dexpr'):
        var.dexpr = func

    # Detect inputs for iconics if not specified
    if spec_inp is None:
      assert self._isiconic, \
          "Input specification mandatory for non-iconic functionals"
      spec_inp = tuple(self._funcs[spec_out].symbols.keys())
      spec_inp = parse_identifiers(spec_inp)
      self._deps[spec_out] = spec_inp

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
    for key, val in self._funcs.items():
      call = functools.partial(val.__call__, *self._args, **self._kwds)
      self._partials.update({key: call})
      if not val.order:
        order = dict()
        deps = self._deps[key]
        for element in self._inp_set:
          if element not in deps:
            order.update({element: None})
        val.set_order(order)

#-------------------------------------------------------------------------------
  def __getitem__(self, arg):
   """ Returns the partial according to arg  """ 
   key = parse_identifiers(arg)
   return self._partials[key]

#-------------------------------------------------------------------------------
  def __repr__(self):
    """ Print representation """
    if self._isiconic:
      return object.__repr__(self)+ ": '{}'".format(self._funcs)
    if self._deps is None:
      return object.__repr__(self) 
    return object.__repr__(self)+ ": '{}'".format(self._deps)

#-------------------------------------------------------------------------------
