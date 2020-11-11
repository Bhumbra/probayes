"""
A stochastic dependency is a random field that accommodates directed 
conditionality according to one or more conditional probability distribution 
functions. The dependency is represented as a graph of nodes, for the random 
variables, and edges for their corresponding inter-relations.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
import networkx as nx
from probayes.rv import RV
from probayes.rf import RF
from probayes.func import Func
from probayes.dist import Dist
from probayes.dist_utils import product
from probayes.rf_utils import rf_prod_rule
from probayes.sd_utils import desuffix, get_suffixed

NX_DIRECTED_GRAPH = nx.OrderedDiGraph
DEFAULT_CONVERGENCE_FUNCTION = 'mul'

#-------------------------------------------------------------------------------
class SD (NX_DIRECTED_GRAPH, RF):
  """ A stochastic dependency is a random field that accommodates directed 
  conditionality according to one or more conditional probability distribution 
  functions. The dependency is represented as a graph of nodes, for the random 
  variables, and edges for their corresponding inter-relations.
  """
  # Public
  opqr = None          # (p(pred), p(succ), q(succ|pred), q(pred|succ))

  # Protected
  _leafs = None        # RF of RVs that do not condition others
  _roots = None        # RF of RVs not dependent on others
  _stems = None        # OrderedDict of latent RVs
  _cverg = None        # Convergence function (defaults to multiplication)
  _cdeps = None        # OrderedDict of conditional dependency graphs
  _def_prop_obj = None # Default value for prop_obj
  _prop_obj = None     # Object referencing propositional conditions
  _tran_obj = None     # Object referencing transitional conditions
  _unit_prob = None    # Flag for single RV probability
  _unit_tran = None    # Flag for single RV transitional

  # Private
  __def_leafs = None   # Default leafs - provides convenience interface
  __def_roots = None   # Default roots - provides convenience interface
  __sub_rfs = None     # Convenience dictionary for the roots and leafs RFs
  __sub_deps = None    # Convenience list if SDs in parallel/series
  __sym_tran = None    # Flag to denote symmetrical conditionals
  __implicit = None    # Implicit configuration: None, "parallel" or "series"

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    """ Initialises the SD with RVs, RFs, or SDs. See def_deps() """
    NX_DIRECTED_GRAPH.__init__(self)
    self.set_prob()
    self.def_deps(*args)
    self.set_cverg()

#-------------------------------------------------------------------------------
  def def_deps(self, *args):
    """ Defaults the dependency of SD with RVs, RFs. or SD arguments.

    :param args: each arg may be an RV, RF, or SD with the dependency chain
                 running from right to left.
    """
    self._cdeps = None
    self.__implicit = None
    self.__sym_tran = False
    self.__def_leafs = None
    self.__def_roots = None
    if not args:
      return

    # Iterate through arguments to add RV vertices detect running roots
    run_roots = [None] * len(args)
    for i, arg in enumerate(args):
      if isinstance(arg, (SD, RF)):
        rvs = arg.ret_rvs(aslist=True)
        for rv in rvs:
          NX_DIRECTED_GRAPH.add_node(self, rv.ret_name(), **{'rv': rv})
        if isinstance(arg, SD):
          run_roots[i] = arg.ret_roots()
          NX_DIRECTED_GRAPH.add_edges_from(self, arg.edges)
        else:
          run_roots[i] = rvs
          if i == 0:
            self.__def_leafs = arg
          elif i == len(args) - 1:
            self.__def_roots = arg
      elif isinstance(arg, RV):
        NX_DIRECTED_GRAPH.add_node(self, arg.ret_name(), **{'rv': arg})
        run_roots[i] = [arg]
    
    # Identify parallel dependencies
    self.__implicit = len(args) > 1 and \
                          all([isinstance(arg, SD) for arg in args])
    leafs = set()
    roots = []
    self.__sub_deps = None
    if self.__implicit:
      self.__implicit = 'parallel'
      self.__sub_deps = []
      for arg in args:
        arg_roots = arg.ret_roots()
        roots += arg_roots.ret_rvs(aslist=True)
        if self.__sub_deps is not None:
          prob = arg.ret_prob()
          if prob is None:
            self.__sub_deps = None
          else:
            self.__sub_deps += [arg]
        if not len(arg_roots):
          self.__implicit = None
        elif not len(leafs):
          leafs = arg.ret_leafs().ret_rvs(aslist=False)
        elif leafs != arg.ret_leafs().ret_rvs(aslist=False):
          self.__implicit = False
        if not self.__implicit:
          self.__sub_deps = None
          self.__implicit = None
          break

    # Default leafs and roots in parallel or series
    if self.__implicit == 'parallel':
      assert len(roots) == len(set(roots)), \
          "For arguments with common leafs, roots must not share RVs"
      self.__def_leafs = args[0].ret_leafs()
      self.__def_roots = RF(*tuple(roots))
      root_keys = [rv.ret_name() for rv in self.__def_leafs.ret_rvs()]
      leaf_keys = [rv.ret_name() for rv in self.__def_roots.ret_rvs()]
      for root_key in root_keys:
        for leaf_key in leaf_keys:
          NX_DIRECTED_GRAPH.add_edge(self, root_key, leaf_key)
    else:
      run_leafs = None
      for i, arg in enumerate(args):
        if type(arg) is RF:
          if i == 0:
            self.__def_leafs = arg
          elif i == len(args) - 1:
            self.__def_roots = arg
        if i > 0:
          root_keys = [rv.ret_name() for rv in run_roots[i]]
          leaf_keys = [rv.ret_name() for rv in run_leafs]
          for root_key in root_keys:
            for leaf_key in leaf_keys:
              NX_DIRECTED_GRAPH.add_edge(self, root_key, leaf_key)
        run_leafs = run_roots[i]

    self._refresh()

#-------------------------------------------------------------------------------
  def _refresh(self):
    """ Refreshes tree summaries, SD name and identity, and default states. 
    While roots and leafs are represented as RFs, stems are contained within a
    single ordered dictionary to be flexible enough to accommodate dependency 
    arborisations.
    """
    super()._refresh()

    # Distinguish RVs belonging to leafs, roots, and stems
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

    # Setup leafs and roots RF objects
    self._leafs, self._roots = None, None
    if not self._cdeps:
      self._leafs = self.__def_leafs
      self._roots = self.__def_roots
    if not self._leafs:
      self._leafs = RF(*tuple(leafs))
    if not self._roots and roots:
      self._roots = RF(*tuple(roots))
    self._defiid = self._leafs.ret_keys(False)

    # Evaluate name and id from leafs and roots only
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
      assert self._cdeps is None, \
          "Cannot specify probabilities alongside cdeps conditional dependencies"
    return super().set_prob(prob, *args, **kwds)

#-------------------------------------------------------------------------------
  def add_cdep(self, conditioned, conditioning, func, *args, **kwds):
    """ Adds a conditional dependence that conditions conditioning with respect
    to condiitioning with functional inputs *args and **kwds.
    """
    if self.__implicit:
      self.remove_edges_from(self.edges)
      self.__implicit = False
    if self._cdeps is None:
      self._cdeps = collections.OrderedDict()
    assert not self._prob, \
        "Cannot assign conditional dependencies alongside specified probability"
    conditioned = self.subfield(conditioned)
    conditioning = self.subfield(conditioning)
    conditioned_rvs = conditioned.ret_rvs(aslist=False)
    conditioning_rvs = conditioning.ret_rvs(aslist=False)
    conditioned_keys = list(conditioned_rvs.keys())
    conditioning_keys = list(conditioning_rvs.keys())
    assert len(set(conditioned_keys).union(set(conditioning_keys))) == \
           len(conditioned_keys) + len(conditioning_keys), \
           "Duplicate RV detected among conditioned and conditioned inputs"
    cdep_key = '|'.join([','.join(conditioned_keys), ','.join(conditioning_keys)])
    self._cdeps.update({cdep_key: 
                        NX_DIRECTED_GRAPH(None, func=Func(func, *args, **kwds))})
    for conditioned_key in conditioned_keys:
      for conditioning_key in conditioning_keys:
        self._cdeps[cdep_key].add_edge(conditioning_key, conditioned_key)
    self.add_edges_from(self._cdeps[cdep_key])
    return self._cdeps[cdep_key]

#-------------------------------------------------------------------------------
  def ret_cdeps(self, key=None):
    if self._cdeps is None:
      return None
    if key is not None:
      return self._cdeps[key]
    return self._cdeps

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
  def ret_leafs(self):
    return self._leafs

#-------------------------------------------------------------------------------
  def ret_roots(self):
    return self._roots

#-------------------------------------------------------------------------------
  def ret_stems(self):
    return self._stems

#-------------------------------------------------------------------------------
  def ret_implicit(self):
    return self.__implicit 
 
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
      return super().set_tfun(tfun, *args, **kwds)
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
  def set_cverg(self, cverg=None, *args, **kwds):
    self._cverg = cverg or DEFAULT_CONVERGENCE_FUNCTION
    if isinstance(self._cverg, str):
      return
    self._cverg = Func(self._cverg, *args, **kwds)

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
  def eval_vals(self, *args, _skip_parsing=False, **kwds):
    assert self._leafs, "No marginal stochastic random variables defined"
    return super().eval_vals(*args, _skip_parsing=_skip_parsing, **kwds)

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None, dims=None):
    if self._prob is not None or self.__sub_deps is None or \
        not (self._cdeps or self.__implicit):
      return super().eval_prob(values, dims)

    # Implicit vergence
    if self.__implicit:
      if self.__implicit == 'parallel':
        if self._cverg == 'mul':
          prob, _ = rf_prod_rule(values, dims=dims, rfs=self.__sub_deps, 
                                 pscale=self._pscale)
        else:
          raise ValueError("Unrecognised vergence function {} ".format(
            self._cverg) + "for implicit parallel dependencies")
      else:
        raise ValueError("Unknown implicit specification: {}".format(
          self.__implicit))
      return prob

    # Explicit vergence
    

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
    return product(cond_dist, dist)

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
    if self._tran is None and not self._unit_tran:
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
  def __mul__(self, other):
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
  def __truediv__(self, other):
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
