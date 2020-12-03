""" Contains stochastic dependence class SD. SD-related utility functions are
provided in sd_utils.py """
#-------------------------------------------------------------------------------
import numpy as np
import collections
import networkx as nx
from probayes.rv import RV
from probayes.rf import RF
from probayes.func import Func
from probayes.dist import Dist
from probayes.dist_utils import product
from probayes.sd_utils import desuffix, get_suffixed, arch_prob
from probayes.cf import CF

NX_DIRECTED_GRAPH = nx.OrderedDiGraph
DEFAULT_CONVERGENCE_FUNCTION = 'mul'

#-------------------------------------------------------------------------------
class SD (NX_DIRECTED_GRAPH, RF):
  """ A stochastic dependence is a random field that accommodates directed 
  conditionality according to one or more conditional probability distribution 
  functions. The dependences are represented as a graph, representing
  RVs as vertices and edges for their corresponding inter-relations. 
  
  Direct conditional dependences across groups of RVs are set using conditional 
  functions that inter-relate RFs. This can be performed via the implicit
  architectural interface (using SD.set_prob()) or explicit dependency
  interface (self.add_deps()).
  """
  # Public
  opqr = None          # (p(pred), p(succ), q(succ|pred), q(pred|succ))

  # Protected
  _arch = None         # Implicit archectectural configuration
  _leafs = None        # RF of RVs that do not condition others
  _roots = None        # RF of RVs not dependent on others
  _stems = None        # OrderedDict of latent RVs
  _def_prop_obj = None # Default value for prop_obj
  _prop_obj = None     # Object referencing propositional conditions
  _tran_obj = None     # Object referencing transitional conditions
  _unit_prob = None    # Flag for single RV probability
  _unit_tran = None    # Flag for single RV transitional

  # Private
  __sub_rfs = None     # Convenience dictionary for the roots and leafs RFs
  __sub_cfs = None     # Dictionary of conditional functions
  __sym_tran = None    # Flag to denote symmetrical conditionals

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    """ Initialises the SD with RVs, RFs, or SDs. See def_deps() """
    NX_DIRECTED_GRAPH.__init__(self)
    self.def_deps(*args)
    self.set_prob()

#-------------------------------------------------------------------------------
  def def_deps(self, *args):
    """ Defaults the dependence of SD with RVs, RFs. or SD arguments.

    :param args: each arg may be an RV, RF, or SD with the dependence chain
                 compatible with running right to left. If one argument in an 
                 SD then all arguments must comprise of SD instances.
    """
    self._deps = None
    self._arch = None
    self.__sym_tran = False
    if not args:
      return
    arg_issd = [isinstance(arg, SD) for arg in args]

    # Absence of SD instances means at most a single direct dependency
    if not any(arg_issd):
      self._arch = self
      rfs = [None] * len(args)
      for i, arg in enumerate(args):
        if isinstance(arg, RV):
          rfs[i] = RF(arg)
        elif isinstance(arg, RF):
          rfs[i] = arg
        else:
          raise TypeError("Unrecognised input argument type: {}".format(
              type(arg)))

      # Declare vertices, adding edges if there are multiple arguments
      if len(rfs) == 1:
        rvs = rfs[0].ret_rvs(aslist=True)
        for rv in rvs:
          NX_DIRECTED_GRAPH.add_node(self, rv.name, **{'rv': rv})
        return self._refresh(rfs[0])
      leafs_rvs = rfs[0].ret_rvs(aslist=True)
      roots_rvs = rfs[-1].ret_rvs(aslist=True)
      if len(rfs) > 2:
        for rf in rfs[1:-1]:
          leafs_rvs += rf.ret_rvs(aslist=True)
      for rv in roots_rvs:
        NX_DIRECTED_GRAPH.add_node(self, rv.name, **{'rv': rv})
      for rv in leafs_rvs:
        NX_DIRECTED_GRAPH.add_node(self, rv.name, **{'rv': rv})
      roots_keys = [rv.name for rv in roots_rvs]
      leafs_keys = [rv.name for rv in leafs_rvs]
      for roots_key in roots_keys:
        for leafs_key in leafs_keys:
          NX_DIRECTED_GRAPH.add_edge(self, roots_key, leafs_key)
      if len(rfs) == 2:
        return self._refresh(rfs[0], rfs[1])
      return self._refresh(RF(*tuple(leafs_rvs)), rfs[-1])

    assert all(arg_issd), "Cannot mix SD with other input types"

    # Adding all vertices in forward order, starting with leafs then the rest
    for arg in args:
      leafs_data = collections.OrderedDict(arg.ret_leafs().nodes.data())
      for key, val in leafs_data.items():
        NX_DIRECTED_GRAPH.add_node(self, val['rv'].name(), **{'rv': val['rv']})
    for arg in args:
      rvs_data = collections.OrderedDict(arg.nodes.data())
      for key, val in rvs_data.items():
        NX_DIRECTED_GRAPH.add_node(self, val['rv'].name(), **{'rv': val['rv']})

    # Adding all edges in reverse order
    for arg in args[::-1]:
      NX_DIRECTED_GRAPH.add_edges_from(self, arg.edges())

    # Explicit dependences to be added in reverse order with all implicit bets off
    deps = [arg.ret_deps() for arg in args[::-1]]
    if any(deps):
      self._arch = None
      [self.add_deps(dep) for dep in deps]
      return self._refresh()

    # Implicit dependences may either be in parallel or in series, but not both

    # Iterate args, add RV vertices, detect running roots/leafs and explicit
    run_leafs = [None] * len(args)
    run_roots = [None] * len(args)
    for i, arg in enumerate(args):
      run_leafs[i] = arg.ret_leafs().ret_rvs(aslist=True)
      run_roots[i] = arg.ret_roots().ret_rvs(aslist=True)

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
          leafs = arg.ret_leafs()
          roots = arg.ret_roots()
        elif i == len(args) - 1:
          roots = arg.ret_roots()
      return self._refresh(leafs, roots)

    # Detect for implicit parallel dependences
    parallel = len(args) > 1 
    leafs = set()
    roots = []
    if parallel:
      sub_deps = []
      for i, arg in enumerate(args):
        roots += run_roots[i]
        if not leafs:
          leafs = run_leafs[i]
        elif not len(run_roots[i]) or leafs != run_leafs[i]:
          parallel = False
          break
    if parallel:
      self._arch = tuple(args[::-1])
      leafs = args[0].ret_leafs()
      roots = RF(*tuple(roots))
      return self._refresh(leafs, roots)

    return self._refresh()

#-------------------------------------------------------------------------------
  def _refresh(self, leafs=None, roots=None):
    """ Refreshes tree summaries, SD name and identity, and default states. 
    While roots and leafs are represented as RFs, stems are contained within a
    single ordered dictionary to be flexible enough to accommodate dependence 
    arborisations.

    :param leafs: sets default for leafs
    :param roots: sets default for roots
    """
    super()._refresh()
    self._leafs = None
    self._stems = collections.OrderedDict()
    self._roots = None

    # If defaulting leafs, then assume a simple RF specification
    if leafs:
      assert type(leafs) is RF, "Input leafs must be an RF"
      self._leafs = leafs
      self._roots = roots
      if self._roots:
        assert type(self._roots) is RF, "Input roots must be an RF"

    # Otherwise distinguish RVs belonging to leafs, roots, and stems
    else:
      leafs = [] # RF of vertices with no children/successors
      roots = [] # RF of vertices with no parents/predecessors (and not a leaf)
      self._stems = collections.OrderedDict()
      rvs = collections.OrderedDict(self.nodes.data())
      for key, val in rvs.items():
        parents = list(self.predecessors(key))
        children = list(self.successors(key))
        if parents and children:
          self._stems.update({key: val})
        elif children: # roots must have children
          roots += [val]
        else: # leafs can be parentless
          leafs += [val]
      self._leafs = RF(*tuple(leafs))
      if roots:
        self._roots = RF(*tuple(roots))

    # Default IID keys and evaluate name and id from leafs and roots only
    self._keys = list(self._leafs.nodes)
    for key in list(self.nodes):
      if key not in self._keys:
        self._keys.append(key)
    self._keyset = set(self._keys)
    self._defiid = self._leafs.ret_keys(False)
    self._name = self._leafs.ret_name()
    self._id = self._leafs.ret_id()
    self.__sub_rfs = {'leafs': self._leafs}
    if self._roots:
      self._name += "|{}".format(self._roots.ret_name())
      self._id += "_with_{}".format(self._roots.ret_id())
      self.__sub_rfs.update({'roots': self._roots})
    self.set_pscale()
    self.eval_length()
    self.opqr = collections.namedtuple(self._id, ['o', 'p', 'q', 'r'])

    # Set the default proposal object and default the delta accordingly
    self._def_prop_obj = self._roots if self._roots is not None else self._leafs
    self.delta = self._def_prop_obj.delta
    self._delta_type = self._def_prop_obj._delta_type
    self.set_prop_obj(None) # this is for the instantiater to decide

    # Determine unit RVRF
    self._unit_prob = False
    self._unit_tran = False
    if self._nrvs == 1:
      rv = self.ret_rvs()[0]
      self._unit_prob = self._prob is None and rv.ret_prob() is not None
      self._unit_tran = self._tran is None and rv.ret_tran() is not None

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    """ Sets the joint probability with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param pscale: represents the scale used to represent probabilities.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.
    """
    if prob is not None:
      assert self._deps is None, \
          "Cannot specify probabilities alongside deps conditional dependencies"
    prob = super().set_prob(prob, *args, **kwds)
    if prob is not None or not isinstance(self._arch, (list, tuple)):
      return prob
    return super().set_prob(arch_prob, arch=self._arch, passdims=True)

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
  def ret_deps(self, key=None):
    if self._deps is None:
      return None
    if key is not None:
      return self._deps[key]
    return self._deps

#-------------------------------------------------------------------------------
  def set_prop_obj(self, prop_obj=None):
    """ Sets the object used for assigning proposal distributions """
    self._prop_obj = prop_obj
    if self._prop_obj is None:
      return
    self.delta = self._prop_obj.delta
    self._delta_type = self._prop_obj._delta_type

#-------------------------------------------------------------------------------
  def ret_leafs_roots(self, spec=None):
    """ Returns a proxy object from __sub_rfs. """
    if spec is None:
      return self.__sub_rfs
    if not isinstance(spec, str) and spec not in self.__sub_rfs.values(): 
      return False
    if isinstance(spec, str):
      assert spec in self.__sub_rfs, \
          '{} absent from {}'.format(spec, self._name)
      return self.__sub_rfs[spec]
    return spec

#-------------------------------------------------------------------------------
  def ret_arch(self):
    return self._arch 
 
#-------------------------------------------------------------------------------
  def set_prop(self, prop=None, *args, **kwds):
    _prop = self.ret_leafs_roots(prop)
    if not _prop:
      return super().set_prop(prop, *args, **kwds)
    self.set_prop_obj(_prop)
    self._prop = _prop._prop
    return self._prop

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    _delta = self.ret_leafs_roots(delta)
    if not _delta:
      return super().set_delta(delta, *args, **kwds)
    self.set_prop_obj(_delta)
    self._delta = _delta._delta
    self._delta_args = _delta._delta_args
    self._delta_kwds = _delta._delta_kwds
    self._delta_type = _delta._delta_type
    self._spherise = _delta._spherise
    return self._delta

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    _tran = self.ret_leafs_roots(tran)
    if not _tran:
      self._tran_obj = self
      return super().set_tran(tran, *args, **kwds)
    self._tran_obj = _tran
    self.set_prop_obj(self._tran_obj)
    self._tran = _tran._tran
    return self._tran

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    _tfun = self.ret_leafs_roots(tfun)
    if not _tfun:
      self._tran_obj = self
      return super().set_tfun(tfun, *args, **kwds)
    self._tran_obj = _tfun
    self.set_prop_obj(_tfun)
    self._tfun = _tfun._tfun
    return self._tfun

#-------------------------------------------------------------------------------
  def set_cfun(self, cfun=None, *args, **kwds):
    _cfun = self.ret_leafs_roots(cfun)
    if not _cfun:
      return super().set_cfun(cfun, *args, **kwds)
    self.set_prop_obj(_cfun)
    self._cfun = _cfun._cfun
    self._cfun_lud = _cfun._cfun_lud
    return self._cfun

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values, suffix=None):
    if suffix is not None:
      return super().eval_dist_name(values, suffix)
    keys = self._keys 
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
      for key in self._keys:
        if key not in vals.keys():
          vals.update({key: None})
    marg_vals = collections.OrderedDict()
    if self._leafs:
      for key in self._leafs.ret_keys():
        if key in keys:
          marg_vals.update({key: vals[key]})
    cond_vals = collections.OrderedDict()
    if self._roots:
      for key in self._roots.ret_keys():
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
  def ret_leafs(self):
    return self._leafs

#-------------------------------------------------------------------------------
  def ret_roots(self):
    return self._roots

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    raise NotImplementedError()

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
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
      evals, dims, prob = None, None, None
      if isinstance(output, dict):
        evals = output
      elif not isinstance(output, tuple):
        raise TypeError("Unrecognised type {} for output for dependency {}".\
                        format(type(output), key))
      else: 
        evals = output[0] 
        assert isinstance(evals, dict),\
          "Unrecognised dependency evaluation output type".format(type(evals))
        assert len(output) < 3, \
            "Maximum for 3 outputs from dependency evaluation"
        for argout in output:
          if isinstance(argout, dict):
            assert dims is None, "Output ambiguous for dimensionality"
            dims = argout
          elif isinstance(argout, np.ndarray):
            assert prob is None, "Output ambiguous for probabilities"
            prob = argout
    return vals

#-------------------------------------------------------------------------------
  def eval_vals(self, *args, _skip_parsing=False, **kwds):
    assert self._leafs, "No leaf stochastic random variables defined"
    return super().eval_vals(*args, _skip_parsing=_skip_parsing, **kwds)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Like RF.__call__ but optionally takes 'joint' keyword """

    if not self._nrvs:
      return None
    joint = False if 'joint' not in kwds else kwds.pop('joint')
    dist = super().__call__(*args, **kwds)
    if not joint:
      return dist
    vals = dist.ret_cond_vals()
    cond_dist = self._roots(vals)
    joint_dist = product(cond_dist, dist)
    return joint_dist

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    prop_obj = self._prop_obj
    if prop_obj is None and (self._tran is not None or self._prop is not None):
      return super().step(*args, **kwds)
    prop_obj = prop_obj or self._def_prop_obj
    return prop_obj.step(*args, **kwds)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    prop_obj = self._prop_obj
    if prop_obj is None and (self._tran is not None or self._prop is not None):
      return super().propose(*args, **kwds)
    prop_obj = prop_obj or self._def_prop_obj
    return prop_obj.propose(*args, **kwds)

#-------------------------------------------------------------------------------
  def parse_pred_args(self, *args):
    if self._tran_obj == self:
      return self.parse_args(*args)
    if len(args) == 1 and isinstance(args[0], dict):
      arg = args[0]
      keyset = self._tran_obj.ret_keys(False)
      pred = collections.OrderedDict({key: val for key, val in arg.items() 
                                               if key in keyset})
      return self._tran_obj.parse_args(pred)
    return self._tran_obj.parse_args(*args)

#-------------------------------------------------------------------------------
  def sample(self, *args, **kwds):
    """ A function for unconditional and conditional sampling. For conditional
    sampling, use RF.set_delta() to set the delta specification. if neither
    set_prob() nor set_tran() are set, then opqr inputs are disallowed and this
    function outputs a normal __call__(). Otherwise this function returns a 
    namedtuple-generated opqr object that can be accessed using opqr.p or 
    opqr[1] for the probability distribution and opqr.q or opqr[2] for the 
    proposal. Unavailable values are set to None. 
    
    If using set_prop() the output opqr comprises:

    opqr.o: None
    opqr.p: Probability distribution 
    opqr.q: Proposition distribution
    opqr.r: None

    If using set_tran() the output opqr comprises:

    opqr.o: Probability distribution for predecessor
    opqr.p: Probability distribution for successor
    opqr.q: Proposition distribution (successor | predecessor)
    opqr.r: None [for now, reserved for proposition (predecessor | successor)]

    If inputting and opqr object using set_prop(), the values for performing any
    delta operations are taken from the entered proposition distribution. If using
    set_prop(), optional keyword flag suffix=False may be used to remove prime
    notation in keys.

    An optional argument args[1] can included in order to input a dictionary
    of values beyond outside the proposition distribution required to evaluate
    the probability distribution.
    """
    if not args:
      args = {0},
    assert len(args) < 3, "Maximum of two positional arguments"
    if self._tran is None and self._tfun is None and not self._unit_tran:
      if self._prop is None:
        assert not isinstance(args[0], self.opqr),\
            "Cannot input opqr object with neither set_prob() nor set_tran() set"
        return self.__call__(*args, **kwds)
      return self._sample_prop(*args, **kwds)
    return self._sample_tran(*args, **kwds)

#-------------------------------------------------------------------------------
  def _sample_prop(self, *args, **kwds):

    # Extract suffix status; it is latter popped by propose()
    suffix = "'" if 'suffix' not in kwds else kwds['suffix'] 

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self.opqr):
      prop = self.propose(args[0], **kwds)

    # Otherwise parse:
    else:
      assert args[0].q is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = desuffix(args[0].q.vals)
      prop = self.propose(vals, **kwds)

    # Evaluation of probability
    vals = desuffix(prop.vals)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    call = self.__call__(vals, **kwds)

    return self.opqr(None, call, prop, None)

#-------------------------------------------------------------------------------
  def _sample_tran(self, *args, **kwds):
    assert 'suffix' not in kwds, \
        "Disallowed keyword 'suffix' when using set_tran()"

    # Original probability distribution, proposal, and revp defaults to None
    orig = None
    prop = None
    revp = None

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self.opqr):
      prop = self.step(args[0], **kwds)

    # Otherwise parse successor:
    else:
      dist = args[0].q
      orig = args[0].p
      assert dist is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = get_suffixed(dist.vals)
      prop = self.step(vals, **kwds)

    # Evaluate reverse proposal if transition function not symmetric
    if not self._sym_tran and not self._unit_tran:
      revp = self.reval_tran(prop)

    # Extract values evaluating probability
    vals = get_suffixed(prop.vals)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    prob = self.__call__(vals, **kwds)

    return self.opqr(orig, prob, prop, revp)

#-------------------------------------------------------------------------------
  def __and__(self, other):
    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, SD):
      marg = marg + other.ret_marg().ret_rvs()
      cond = cond + other.ret_cond().ret_rvs()
      return SD(marg, cond)

    if isinstance(other, RF):
      marg = marg + other.ret_rvs()
      return SD(marg, cond)

    if isinstance(other, RV):
      marg = marg + [other]
      return SD(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __or__(self, other):
    marg = self.ret_marg().ret_rvs()
    cond = self.ret_cond().ret_rvs()
    if isinstance(other, SD):
      marg = marg + other.ret_cond().ret_rvs()
      cond = cond + other.ret_marg().ret_rvs()
      return SD(marg, cond)

    if isinstance(other, RF):
      cond = cond + other.ret_rvs()
      return SD(marg, cond)

    if isinstance(other, RV):
      cond = cond + [self]
      return SD(marg, cond)

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
