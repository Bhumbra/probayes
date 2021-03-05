""" Provides stochastic dependence class SD. SD-related utility functions are
provided in sd_utils.py """
#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.dependence import Dependence
from probayes.rv import RV
from probayes.rf import RF
from probayes.pd_utils import product
from probayes.sd_utils import desuffix, get_suffixed, arch_prob

#-------------------------------------------------------------------------------
class SD (Dependence, RF):
  """ A stochastic dependence is a random field that accommodates directed 
  conditionality according to one or more conditional probability distribution 
  functions. The dependences are represented as a graph, representing
  RVs as vertices and edges for their corresponding inter-relations. 
  
  Direct conditional dependences across groups of RVs are set using conditional 
  functions that inter-relate RFs. This can be performed via the implicit
  architectural interface (using SD.set_prob()) or explicit dependency
  interface (self.add_deps()).
  """

  # Protected
  _opqr = None         # (p(pred), p(succ), q(succ|pred), q(pred|succ))
  _tran_obj = None     # Object referencing transitional conditions
  _variable_cls = RV   # Variable class
  _field_cls = RF      # Field class
  _unit_prob = None    # Flag for single RV probability
  _unit_tran = None    # Flag for single RV transitional
  _is_stochastic = True  # Flag of stochasticity status

  # Private
  __sub_cfs = None     # Dictionary of conditional functions
  __sym_tran = None    # Flag to denote symmetrical conditionals

#------------------------------------------------------------------------------- 
  def __init__(self, *args):
    """ Initialises the SD with RVs, RFs, or SDs. See def_deps() """
    Dependence.__init__(self, *args)
    self.set_prob()

#-------------------------------------------------------------------------------
  @property
  def opqr(self):
    return self._opqr

  def _refresh(self, leafs=None, roots=None):
    """ Refreshes tree summaries, SD name and identity, and default states. 
    While roots and leafs are represented as RFs, stems are contained within a
    single ordered dictionary to be flexible enough to accommodate dependence 
    arborisations.

    :param leafs: sets default for leafs
    :param roots: sets default for roots
    """
    self.__sym_tran = False
    super()._refresh(leafs, roots)

    # Determine unit RVRF
    self._opqr = collections.namedtuple(self._id, ['o', 'p', 'q', 'r'])
    self._unit_prob = False
    self._unit_tran = False
    if self._nvars == 1:
      var = self._varlist[0]
      self._unit_prob = self._prob is None and var.prob is not None
      self._unit_tran = self._tran is None and var.tran is not None

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
  def set_prop(self, prop=None, *args, **kwds):
    _prop = self.leafs_roots(prop)
    if not _prop:
      return super().set_prop(prop, *args, **kwds)
    self.set_prop_obj(_prop)
    self._prop = _prop._prop
    return self._prop

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    _tran = self.leafs_roots(tran)
    if not _tran:
      self._tran_obj = self
      return super().set_tran(tran, *args, **kwds)
    self._tran_obj = _tran
    self.set_prop_obj(self._tran_obj)
    self._tran = _tran.tran
    return self._tran

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    _tfun = self.leafs_roots(tfun)
    if not _tfun:
      self._tran_obj = self
      return super().set_tfun(tfun, *args, **kwds)
    self._tran_obj = _tfun
    self.set_prop_obj(_tfun)
    self._tfun = _tfun.tfun
    return self._tfun

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
            "Unrecognised dependency evaluation output type: {}".format(
                type(evals))
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
  def __call__(self, *args, **kwds):
    """ Like RF.__call__ but optionally takes 'joint' keyword """
    if not self.anystoch:
      return Dependence.__call__(self, *args, **kwds)
    if not self._nvars:
      return None
    joint = False if 'joint' not in kwds else kwds.pop('joint')
    dist = super().__call__(*args, **kwds)
    if not joint:
      return dist
    vals = dist('cond')
    cond_dist = self._roots(vals)
    joint_dist = product(cond_dist, dist)
    return joint_dist

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
      keyset = self._tran_obj.keyset
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
    if not args: # Default to randomly sampling variable scalars
      args = {0},
    assert len(args) < 3, "Maximum of two positional arguments"
    if self._tran is None and self._tfun is None and not self._unit_tran:
      if self._prop is None:
        assert not isinstance(args[0], self._opqr),\
            "Cannot input opqr object with neither set_prob() nor set_tran() set"
        return self.__call__(*args, **kwds)
      return self._sample_prop(*args, **kwds)
    return self._sample_tran(*args, **kwds)

#-------------------------------------------------------------------------------
  def _sample_prop(self, *args, **kwds):

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self._opqr):
      prop = self.propose(args[0], **kwds)

    # Otherwise parse:
    else:
      assert args[0].q is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = desuffix(args[0].q)
      prop = self.propose(vals, **kwds)

    # Evaluation of probability
    vals = desuffix(prop)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    call = self.__call__(vals, **kwds)

    return self._opqr(None, call, prop, None)

#-------------------------------------------------------------------------------
  def _sample_tran(self, *args, **kwds):
    assert 'suffix' not in kwds, \
        "Disallowed keyword 'suffix' when using set_tran()"

    # Original probability distribution, proposal, and revp defaults to None
    orig = None
    prop = None
    revp = None

    # Non-opqr argument requires no parsing
    if not isinstance(args[0], self._opqr):
      prop = self.step(args[0], **kwds)

    # Otherwise parse successor:
    else:
      dist = args[0].q
      orig = args[0].p
      assert dist is not None, \
          "An input opqr argument must contain a non-None value for opqr.q"
      vals = get_suffixed(dist)
      prop = self.step(vals, **kwds)

    # Evaluate reverse proposal if transition function not symmetric
    if not self._sym_tran and not self._unit_tran:
      revp = self.reval_tran(prop)

    # Extract values evaluating probability
    vals = get_suffixed(prop)
    if len(args) > 1:
      assert isinstance(args[1], dict),\
          "Second argument must be dictionary type, not {}".format(
              type(args[1]))
      vals.update(args[1])
    prob = self.__call__(vals, **kwds)

    return self._opqr(orig, prob, prop, revp)

#-------------------------------------------------------------------------------
