""" Provides Dependence class """

#-------------------------------------------------------------------------------
import collections
import networkx as nx
from probayes.variable import Variable
from probayes.field import Field
from probayes.cf import CF
from probayes.functional import Functional
NX_DIRECTED_GRAPH = nx.DiGraph

#-------------------------------------------------------------------------------
class Dependence (NX_DIRECTED_GRAPH, Field):
  """ A dependence is field with an explicit directed conditionality. The 
  dependences are represented as a graph, representing variables as vertices, 
  and edges for their corresponding inter-relations. 
  
  Direct conditional dependences across groups of variables are set using 
  conditional functions that inter-relate fields. This can be performed via 
  the implicit architectural interface or explicit dependency interface 
  (self.add_deps()).

  Modification of a dependence functional can be made from an instance, only
  affecting its correspondence leaf vertices. Intermediate dependence functions
  can be set only from dependence predecessors.

  For convenience to define deltas, Dependence supports defaulting of proposal 
  objects without actually supporting proposals.
  """

  # Protected
  _arch = None             # Implicit archectectural configuration
  _leafs = None            # Field of variables that do not condition others
  _roots = None            # Field of variables not dependent on others
  _stems = None            # OrderedDict of latent RVs
  _preds = None            # Field of variables that are predecessors to leafs
  _def_prop_obj = None     # Default value for prop_obj
  _prop_obj = None         # Object referencing propositional conditions
  _variable_cls = Variable # Variable class
  _field_cls = Field       # Field class
  _subfields = None        # Convenience dictionary for the roots and leafs RFs
  _is_stochastic = False   # Flag of stochasticity status

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    """ Initialises the Dependence with Variables, Fields, or Dependences. 
    See def_deps().; """
    NX_DIRECTED_GRAPH.__init__(self)
    self.def_deps(*args)

#-------------------------------------------------------------------------------
  @property
  def preds(self):
    return self._preds

  @property
  def deps(self):
    return self._deps

  @property
  def arch(self):
    return self._arch

  def def_deps(self, *args):
    """ Defaults the dependence with variables, fields, or dependence arguments.

    :param args: each arg may be a variable, field, or dependence with the 
                 dependence chain with running right to left. If one argument is
                 a dependence, then all arguments must comprise of dependence 
                 instances.
    """
    self._deps = None
    self._arch = None
    if not args:
      return
    arg_aredep = [isinstance(arg, type(self)) for arg in args]

    # Absence of Dependence instances means at most a single direct dependency
    if not any(arg_aredep):
      self._arch = self
      fields = [None] * len(args)
      for i, arg in enumerate(args):
        if isinstance(arg, self._variable_cls):
          fields[i] = self._field_cls(arg)
        elif isinstance(arg, self._field_cls):
          fields[i] = arg
        else:
          raise TypeError("Unrecognised input argument type: {}".format(
              type(arg)))

      # Declare vertices, adding edges if there are multiple arguments
      if len(fields) == 1:
        variables = list(fields[0].varlist)
        for var in variables:
          NX_DIRECTED_GRAPH.add_node(self, var)
        return self._refresh(fields[0])
      leafs_vars = fields[0].varlist
      roots_vars = fields[-1].varlist
      if len(fields) > 2:
        for field in fields[1:-1]:
          leafs_vars += list(field.varlist)

      # Add leafs then roots
      for var in leafs_vars:
        NX_DIRECTED_GRAPH.add_node(self, var)
      for var in roots_vars:
        NX_DIRECTED_GRAPH.add_node(self, var)

      # Add edges
      for roots_var in roots_vars:
        for leafs_var in leafs_vars:
          NX_DIRECTED_GRAPH.add_edge(self, roots_var, leafs_var)
      if len(fields) == 2:
        return self._refresh(fields[0], fields[1])
      return self._refresh(self._field_cls(*tuple(leafs_var)), fields[-1])

    # At this point, all arguments must be dependences
    assert all(arg_aredep), "Cannot mix Dependences with other input types"

    # Adding all vertices in forward order, starting with leafs then the rest
    for arg in args:
      leafs = list(arg.leafs.varlist)
      for leaf in leafs:
        NX_DIRECTED_GRAPH.add_node(self, leaf)
    for arg in args:
      variables = list(arg.nodes)
      for var in variables:
        NX_DIRECTED_GRAPH.add_node(self, var)

    # Adding all edges in reverse order
    for arg in args[::-1]:
      NX_DIRECTED_GRAPH.add_edges_from(self, arg.edges())

    # Explicit dependences to be added in reverse order with all implicit bets off
    deps = [arg.deps for arg in args[::-1]]
    if any(deps):
      self._arch = None
      [self.add_deps(dep) for dep in deps]
      return self._refresh()

    # Implicit dependences may either be in parallel or in series, but not both

    # Iterate args, add RV vertices, detect running roots/leafs and explicit
    run_leafs = [None] * len(args)
    run_roots = [None] * len(args)
    for i, arg in enumerate(args):
      run_leafs[i] = list(arg.leafs.varlist)
      run_roots[i] = list(arg.roots.varlist)

    # Detect for implicit serial dependences
    serial = len(args) > 1
    for i in range(len(args)-1):
      if run_roots[i] is None or set(run_roots[i]) != set(run_leafs[i+1]):
        serial = False
        break
    if serial:
      self._arch = list(args[::-1])
      leafs, roots = None, None
      for i, arg in enumerate(args):
        if i == 0:
          leafs = arg.leafs
          roots = arg.roots
        elif i == len(args) - 1:
          roots = arg.roots
      return self._refresh(leafs, roots)

    # Detect for implicit parallel dependences
    parallel = len(args) > 1 
    leafs = set()
    roots = []
    if parallel:
      for i, arg in enumerate(args):
        roots += run_roots[i]
        if not leafs:
          leafs = run_leafs[i]
        elif not len(run_roots[i]) or leafs != run_leafs[i]:
          parallel = False
          break
    if parallel:
      self._arch = tuple(args[::-1])
      leafs = args[0].leafs
      roots = self._field_cls(*tuple(roots))
      return self._refresh(leafs, roots)

    return self._refresh()

#-------------------------------------------------------------------------------
  @property
  def subfields(self):
    return self._subfields

  def _refresh(self, leafs=None, roots=None):
    """ Refreshes tree summaries, Dependence name and identity, and default 
    states. While roots and leafs are represented as Fields, stems are contained 
    within a single ordered dictionary to be flexible enough to accommodate 
    dependence  arborisations.

    :param leafs: sets default for leafs
    :param roots: sets default for roots
    """
    self._leafs = None
    self._stems = collections.OrderedDict()
    self._roots = None

    # If defaulting leafs, then assume a simple RF specification
    if leafs:
      assert type(leafs) is self._field_cls, \
          "Input leafs cannot be type {}".format(type(leafs))
      self._leafs = leafs
      self._roots = roots
      if self._roots:
        assert type(self._roots) is self._field_cls, \
          "Input roots cannot be type {}".format(type(roots))

    # Otherwise distinguish Variables belonging to leafs, roots, and stems
    else:
      leafs = [] # Field of vertices with no children/successors
      roots = [] # Field of vertices with no parents/predecessors (and not a leaf)
      self._stems = collections.OrderedDict()
      variables = list(self.nodes)
      for var in variables:
        parents = list(self.predecessors(var))
        children = list(self.successors(var))
        if parents and children:
          self._stems.update({var.name: var})
        elif children: # roots must have children
          roots += [var]
        else: # leafs can be parentless
          leafs += [var]
      self._leafs = self._field_cls(*tuple(leafs))
      if roots:
        self._roots = self._field_cls(*tuple(roots))

    # Evaluate leafs predecessors 
    self._preds = None
    if not self._stems:
      self._preds = None if not self._roots else self._roots
    else:
      preds = set()
      for leaf_var in self._leafs.varlist:
        [preds.add(pred_var) for pred_var in self.predecessors(leaf_var)]
      self._preds = self._field_cls(*tuple(preds))

    # Default functions for implicit architectures
    if self._arch: # TODO self._arch must always be defined by this point
      self._func = Functional(self._leafs, self._preds)

    # Set convenience subfields, and evaluate name and id from leafs and roots only
    super()._refresh()
    self._name = self._leafs.name
    self._id = self._leafs.id
    self._subfields = {'leafs': self._leafs}
    if self._roots:
      self._name += "|{}".format(self._roots.name)
      self._id += "_with_{}".format(self._roots.id)
      self._subfields.update({'roots': self._roots})
    self.eval_length()

    # Set the default proposal object and default the delta accordingly
    self._def_prop_obj = self._roots if self._roots is not None else self._leafs
    self._Delta = self._def_prop_obj.Delta
    self._delta_type = self._def_prop_obj._delta_type
    self.set_prop_obj(None) # this is for the instantiater to decide

#-------------------------------------------------------------------------------
  @property
  def func(self):
    return self._func

  def add_func(self, spec, func, *args, **kwds):
    assert self._func, \
        "Architectual specification not able to accommodate functionals"
    self._func.add_func(spec, func, *args, **kwds)

#-------------------------------------------------------------------------------
  def add_deps(self, out, inp=None, func=None, *args, **kwds):
    """ Adds a conditional dependence that conditions conditioning with respect
    to out being conditioned by inp by function func with *args and **kwds.
    """
    if self._arch:
      self.remove_edges_from(self.edges)
      self._arch = None
    if self._deps is None:
      self._deps = collections.OrderedDict()
    assert not self._prob, \
        "Cannot assign conditional dependencies alongside specified probability"
    if inp is None and func is None:
      for key, val in out.items():
        self._deps.update({key: val})
        self.add_edges_from(val)

    dep = CF(out, inp, func, *args, **kwds)
    dep_key = dep.ret_name()
    self._deps.update({dep_key: dep})
    out_keys = list(dep.ret_out().ret_keys(as_list=True))
    inp_keys = list(dep.ret_inp().ret_keys(as_list=True))
    for out_key in out_keys:
      for inp_key in inp_keys:
        self.add_edge(inp_key, out_key)
    return collections.OrderedDict({dep_key: self._deps[dep_key]})

#-------------------------------------------------------------------------------
  @property
  def prop_obj(self):
    return self._prop_obj

  def set_prop_obj(self, prop_obj=None):
    """ Sets the object used for assigning proposal distributions """
    self._prop_obj = prop_obj
    if self._prop_obj is None:
      return
    self._Delta = self._prop_obj.Delta
    self._delta_type = self._prop_obj._delta_type

#-------------------------------------------------------------------------------
  def leafs_roots(self, spec=None):
    """ Returns a proxy object from _subfields. """
    if spec is None:
      return self._subfields
    if not isinstance(spec, str) and spec not in self._subfields.values(): 
      return False
    if isinstance(spec, str):
      assert spec in self._subfields, \
          '{} absent from {}'.format(spec, self._name)
      return self._subfields[spec]
    return spec

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    _delta = self.leafs_roots(delta)
    if not _delta:
      return super().set_delta(delta, *args, **kwds)
    self.set_prop_obj(_delta)
    self._delta = _delta.delta
    self._delta_args = _delta._delta_args
    self._delta_kwds = _delta._delta_kwds
    self._delta_type = _delta._delta_type
    self._spherise = _delta._spherise
    return self._delta

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values, suffix=None):
    if suffix is not None:
      return super().eval_dist_name(values, suffix)
    keys = self._keylist 
    vals = collections.OrderedDict()
    if not isinstance(vals, dict):
      vals.update({key: vals for key in keys})
    else:
      for key, val in values.items():
        if ',' in key:
          subkeys = key.split(',')
          for i, subkey in enumerate(subkeys):
            vals.update({subkey: val[i]})
        else:
          vals.update({key: val})
      for key in self._keylist:
        if key not in vals.keys():
          vals.update({key: None})
    marg_vals = collections.OrderedDict()
    if self._leafs:
      for key in self._leafs.keylist:
        if key in keys:
          marg_vals.update({key: vals[key]})
    cond_vals = collections.OrderedDict()
    if self._roots:
      for key in self._roots.keylist:
        if key in keys:
          cond_vals.update({key: vals[key]})
    marg_dist_name = self._leafs.eval_dist_name(marg_vals)
    cond_dist_name = '' if not self._roots else \
                     self._roots.eval_dist_name(cond_vals)
    dist_name = marg_dist_name
    if len(cond_dist_name):
      dist_name += "|{}".format(cond_dist_name)
    return dist_name

#-------------------------------------------------------------------------------
  def set_vars(self, *args):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def add_var(self, rv):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_marg_prod(self, samples):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def eval_deps(self, *args, _skip_parsing=False, **kwds):
    if self._deps is None:
      return None
    vals = self.parse_args(*args, **kwds) if not _skip_parsing else args[0]
    for key, val in self._deps.items():
      output = val(vals)
      evals, dims = None, None
      if isinstance(output, dict):
        evals = output
      elif not isinstance(output, tuple):
        raise TypeError("Unrecognised type {} for output for dependency {}".\
                        format(type(output), key))
      else: 
        evals = output[0] 
        assert isinstance(evals, dict),\
            "Unrecognised dependency evaluation output type: {}".format(
                type(evals))
        assert len(output) < 3, \
            "Maximum for 3 outputs from dependency evaluation"
        for argout in output:
          if isinstance(argout, dict):
            assert dims is None, "Output ambiguous for dimensionality"
            dims = argout
    return vals

#-------------------------------------------------------------------------------
  def eval_func(self, *args, _skip_parsing=False, **kwds):
    """ Evaluates leaf values from immediate predecessors """
    assert len(self._func), \
        "No functions defined for functional {}".format(self._func)
    vals = self.parse_args(*args, **kwds) if not _skip_parsing else args[0]

    # Can only evaluate architectures
    if not self._arch: 
      return vals

    # Iconics should be evaluable directly:
    if self._func.isiconic:
      pass

#-------------------------------------------------------------------------------
  def evaluate(self, *args, _skip_parsing=False, **kwds):
    """ Returns evaluation for Field() if there are no dependencies otherwise
    evaluation is based on functionals defined for each Dependences.
    """
    assert self._leafs, "No leaf stochastic random variables defined"
    if not self._arch or not self._func or not len(self._func):
      return super().evaluate(*args, _skip_parsing=_skip_parsing, **kwds)
    

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Currently calls Field.__call__ """
    return super().__call__(*args, **kwds)

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    prop_obj = self._prop_obj
    if prop_obj is None:
      return super().step(*args, **kwds)
    prop_obj = prop_obj or self._def_prop_obj
    return prop_obj.step(*args, **kwds)

#-------------------------------------------------------------------------------
