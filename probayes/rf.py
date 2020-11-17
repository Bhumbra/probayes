"""
A random field is a collection of a random variables that participate in a joint 
probability distribution function without conditioning directions.
"""
#-------------------------------------------------------------------------------
import warnings
import collections
import numpy as np
import scipy.stats
import networkx as nx

from probayes.rv import RV
from probayes.dist import Dist
from probayes.dist_utils import margcond_str
from probayes.vtypes import isscalar, isunitsetint, issingleton, isdimensionless, \
                            revtype, uniform
from probayes.pscales import iscomplex, real_sqrt, prod_rule, \
                         rescale, eval_pscale, prod_pscale
from probayes.rf_utils import rv_prod_rule, call_scipy_prob, sample_cond_cov
from probayes.func import Func, is_scipy_stats_mvar
from probayes.cf import CF
from probayes.cond_cov import CondCov

NX_UNDIRECTED_GRAPH = nx.OrderedGraph

#-------------------------------------------------------------------------------
class RF (NX_UNDIRECTED_GRAPH):
  """
  A random field is a collection of a random variables that participate in a 
  joint probability distribution function without explicit directional 
  conditionality. 
  
  Since this class is intended as a building block for SD instances and networkx 
  cannot mix undirected and directed graphs, edges cannot be defined explicitly 
  within this class. Use SD if directed edges are required. Implicit support for
  undirected edges is provided by the set_prob(), set_prop(), and set_tran()
  methods.
  """

  # Public
  delta = None       # Publicly available delta factory

  # Protected
  _name = None       # Random field name cannot be set externally
  _nrvs = None       # Number of random variables
  _keys = None       # Ordered list of keys of random variable names
  _keyset = None     # Unordered set of keys of random variable names
  _defiid = None     # Default IID random variables for calling distributions
  _prob = None       # Joint probability distribution function 
  _pscale = None     # Probability scaling (see RV)  
  _prop = None       # Non-transitional proposition function
  _prop_deps = None  # Set of proposition dependencies
  _delta = None      # Delta function (to replace step)
  _delta_args = None # Optional delta args (must be dictionaries)
  _delta_kwds = None # Optional delta kwds
  _delta_type = None # Proxy for delta used for casting
  _tran = None       # Transitional proposition function
  _tfun = None       # CDF/IDF of transition function 
  _tsteps = None     # Number of steps per transitional modificiation
  _crvs = None       # Conditional random variable sampling specification
  _length = None     # Length of junction
  _lengths = None    # Lengths of RVs
  _sym_tran = None   # Flag for symmetrical transitional conditional functions
  _spherise = None   # Flag to spherise samples

  # Private
  __isscalar = None  # isscalar(_prob)
  __callable = None  # callable(_prob)
  __cond_mod = None  # conditional RV index modulus
  __cond_cov = None  # conditional covariance matrix

#-------------------------------------------------------------------------------
  def __init__(self, *args): # over-rides NX_GRAPH.__init__()
    """ Initialises a random field with RVs for in args. See set_rvs(). """
    super().__init__()
    self.set_rvs(*args)
    self.set_prob()
    self.set_prop()
    self.set_delta()
    self.set_tran()

#-------------------------------------------------------------------------------
  def add_node(self, *args, **kwds):
    """ Direct adding of nodes disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def add_node_from(self, *args, **kwds):
    """ Direct adding of nodes disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def remove_node(self, *args, **kwds):
    """ Removal of nodes disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def add_edge(self, *args, **kwds):
    """ Direct adding of edges disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def add_edges_from(self, *args, **kwds):
    """ Direct adding of edges disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def add_cd(self, *args, **kwds):
    """ Adding of conditional dependences disallowed """
    raise NotImplementedError("Not implemented for this class")

#-------------------------------------------------------------------------------
  def set_rvs(self, *args):
    """ Initialises a random field with RVs for each arg in args.

    :param *args: each arg may be an RV instance or the first arg may be a RF.
    
    """
    if len(args) == 1 and isinstance(args[0], (RF, dict, set, tuple, list)):
      args = args[0]
    else:
      args = tuple(args)
    self.add_rv(args)

#-------------------------------------------------------------------------------
  def add_rv(self, rv):
    """ Adds one or more random variables to the random field.

    :param rv: a RV or RF instance, or a list/tuple of RV instances.
    """
    assert self._prob is None, \
      "Cannot assign new randon variables after specifying joint/condition prob"
    if isinstance(rv, (RF, dict, set, tuple, list)):
      rvs = rv
      if isinstance(rvs, RF):
        rvs = rvs.ret_rvs()
      if isinstance(rvs, dict):
        rvs = rvs.values()
      [self.add_rv(rv) for rv in rvs]
    else:
      assert isinstance(rv, RV), \
          "Input not a RV instance but of type: {}".format(type(rv))
      key = rv.ret_name()
      if self._nrvs:
        assert key not in list(self.nodes), \
            "Existing RV name {} already present in collection".format(key)
      super().add_node(key, **{'rv': rv})
    self._refresh()

#-------------------------------------------------------------------------------
  def _refresh(self):
    """ Updates RV summary objects, RF name and id, and delta factory. """
    self._nrvs = self.number_of_nodes()
    self._keys = list(self.nodes)
    self._keyset = set(self._keys)
    self._defiid = self._keyset
    self._name = ','.join(self._keys)
    self._id = '_and_'.join(self._keys)
    if self._id:
      self.delta = collections.namedtuple('ð', self._keys)
      self._delta_type = self.delta
    self.set_pscale()
    self.eval_length()

#-------------------------------------------------------------------------------
  def set_prob(self, prob=None, *args, **kwds):
    """ Sets the joint probability with optional arguments and keywords.

    :param prob: may be a scalar, array, or callable function.
    :param pscale: represents the scale used to represent probabilities.
    :param *args: optional arguments to pass if prob is callable.
    :param **kwds: optional keywords to pass if prob is callable.
    """
    kwds = dict(kwds)
    self.__passdims = False
    if 'pscale' in kwds:
      pscale = kwds.pop('pscale')
      self.set_pscale(pscale)
    if 'passdims' in kwds:
      self.__passdims = kwds.pop('passdims')
    self.__callable = None
    self.__isscalar = None
    self._prob = prob
    if self._prob is None:
      return
    self._prob = Func(self._prob, *args, **kwds)
    self.__callable = self._prob.ret_callable()
    self.__isscalar = self._prob.ret_isscalar()

#-------------------------------------------------------------------------------
  def set_pscale(self, pscale=None):
    """ Sets the probability scaling constant used for probabilities.

    :param pscale: can be None, a real number, or a complex number, or 'log'

       if pscale is None (default) the normalisation constant is set as 1.
       if pscale is real, this defines the normalisation constant.
       if pscale is complex, this defines the offset for log probabilities.
       if pscale is 'log', this denotes a logarithmic scale with an offset of 0.

    :return: pscale (either as a real or complex number)
    """
    if pscale is not None or not self._nrvs:
      self._pscale = eval_pscale(pscale)
      return self._pscale
    rvs = self.ret_rvs(aslist=True)
    pscales = [rv.ret_pscale() for rv in rvs]
    self._pscale = prod_pscale(pscales)
    return self._pscale

#-------------------------------------------------------------------------------
  def set_prop(self, prop=None, *args, **kwds):
    """ Sets the joint proposition function with optional arguments and keywords.

    :param prop: may be a scalar, array, or callable function.
    :param *args: optional arguments to pass if prop is callable.
    :param **kwds: optional keywords to pass if prop is callable.
    """
    self._prop = prop
    self._prop_deps = self._keys if 'deps' not in kwds else kwds.pop['deps']
    if self._prop is None:
      return
    assert self._tran is None, \
        "Cannot assign both proposition and transition probabilities"
    self._prop = Func(self._prop, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_delta(self, delta=None, *args, **kwds):
    """ Sets the default delta function or operation.

    :param delta: the delta function or operation (see below)
    :param *args: optional arguments to pass if delta is callable.
    :param **kwds: optional keywords to pass if delta is callable.

    The input delta may be:

    1. A callable function (for which args and kwds are passed on as usual).
    2. An RV.delta instance (this defaults all RV deltas).
    3. A dictionary for RVs, this is converted to an RF.delta.
    4. A scalar that may contained in a list or tuple:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly and independently sampled across RVs.
      c) Tuple - delta is spherically sampled across RVs.

      For non-tuples, an optional argument (args[0]) can be included as a 
      dictionary to specify by RV-name deltas following the above conventions
      except their values are not subject to scaling even if 'scale' is given,
      but they are subject to bounding if 'bound' is specified.

    For setting types 2-4, optional keywords are (default False):
      'scale': Flag to denote scaling deltas to RV lengths
      'bound': Flag to constrain delta effects to RV bounds (None bounces)
      
    """
    self._delta = delta
    self._delta_args = args
    self._delta_kwds = dict(kwds)
    self._spherise = {}
    if self._delta is None:
      return
    elif callable(self._delta):
      self._delta = Func(self._delta, *args, **kwds)
      return

    # Default scale and bound
    if 'scale' not in self._delta_kwds:
      self._delta_kwds.update({'scale': False})
    if 'bound' not in self._delta_kwds:
      self._delta_kwds.update({'bound': False})
    scale = self._delta_kwds['scale']
    bound = self._delta_kwds['bound']

    # Handle deltas and dictionaries
    if isinstance(self._delta, dict):
      self._delta = self._delta_type(**self._delta)
    if isinstance(delta, self._delta_type):
      assert not args, \
        "Optional args prohibited for dict/delta instance inputs"
      rvs = self.ret_rvs(aslist=True)
      for i, rv in enumerate(rvs):
        rv.set_delta(self._delta[i], scale=scale, bound=bound)
      return

    # Default scale and bound and check args
    if self._delta_args:
      assert len(self._delta_args) == 1, \
          "Optional positional arguments must comprises a single dict"
      unscale = self._delta_args[0]
      assert isinstance(unscale, dict), \
          "Optional positional arguments must comprises a single dict"

    # Non tuples can be converted to deltas; can pre-scale here
    if not isinstance(self._delta, tuple):
      scaling = self._lengths
      delta = self._delta 
      urand = isinstance(delta, list)
      if urand:
        assert len(delta) == 1, "List delta requires a single element"
        delta = delta[0]
      deltas = {key: delta for key in self._keys}
      unscale = {} if not self._delta_args else self._delta_args[0]
      deltas.update(unscale)
      delta_dict = collections.OrderedDict(deltas)
      for i, (key, val) in enumerate(deltas.items()):
        delta = val
        if scale and key not in unscale:
          assert np.isfinite(self._lengths[i]), \
              "Cannot scale by infinite length for RV {}".format(key)
          delta = val * self._lengths[i]
        if urand:
          delta = [delta]
        delta_dict.update({key: delta})
      self._delta = self._delta_type(**delta_dict)
      rvs = self.ret_rvs(aslist=True)
      for i, rv in enumerate(rvs):
        rv.set_delta(self._delta[i], scale=False, bound=bound)

    # Tuple deltas must be evaluated on-the-fly and cannot be pre-scaled
    else:
      unscale = {} if not self._delta_args else self._delta_args[0]
      self._spherise = {}
      for i, key in enumerate(self._keys):
        if key not in unscale.keys():
          length = self._lengths[i]
          assert np.isfinite(length), \
              "Cannot spherise RV {} with infinite length".format(key)
          self._spherise.update({key: length})

#-------------------------------------------------------------------------------
  def set_tran(self, tran=None, *args, **kwds):
    """ Sets the conditional transition function with optional arguments and 
    keywords.

    :param tran: may be a scalar, covariance array, or callable function.
    :param *args: optional arguments to pass if tran is callable.
    :param **kwds: optional keywords to pass if tran is callable.
    """
    # Set transition function
    self._tran = tran
    self._sym_tran = False
    self._tsteps = None
    if self._tran is None:
      return
    assert self._prop is None, \
        "Cannot assign both proposition and transition probabilities"
    kwds = dict(kwds)
    if 'tsteps' in kwds:
      self._tsteps = kwds.pop('tsteps')
      if self._tsteps:
        assert type(self._tsteps) is int, "Input tsteps must be int"

    # Pass CF objects directly
    if isinstance(tran, CF):
      assert not args and not kwds, \
        "Setting args and kwds not supported if inputting a CF instance"
      self._tran = CF(self, 
                      tran.ret_inp(), 
                      tran.ret_func(), 
                      *tran.ret_args(), 
                      **tran.ret_kwds())
      return

    # Handle scalars explicitly
    elif isscalar(tran):
      self._tran = Func(tran, *args, **kwds)
      self._sym_tran = not self._tran.ret_ismulti()
      return

    # If a covariance matrix, set the LU decomposition as the tfun
    elif not callable(tran):
      message = "Non-callable non-scalar tran objects must be a square 2D " + \
                "Numpy array of size corresponding to number of variables {}".\
                 format(self._nrvs)
      assert isinstance(tran, np.ndarray), message
      assert tran.ndim == 2, message
      assert np.all(np.array(tran.shape) == self._nrvs), message
      self._tran = Func(tran, *args, **kwds)
      self._sym_tran = not self._tran.ret_ismulti()
      self.set_tfun(np.linalg.cholesky(tran))
      return

    # If a scipy object, set the tfun
    elif is_scipy_stats_mvar(tran):
      self._tran = Func(tran, *args, **kwds)
      self._sym_tran = not self._tran.ret_ismulti()
      scipyobj = self._tran.ret_scipyobj()
      self.set_tfun(self._tran, scipyobj=self._tran.ret_scipyobj())
      return

    # Otherwise instantiate a formal conditional function
    inp = None
    if len(args):
      if isinstance(args[0], collections.OrderedDict):
        args = tuple(args)
        inp, args = args[0], args[1:]
    self._tran = CF(self, inp, tran, *args, **kwds)
    self._sym_tran = not self._tran.ret_ismulti()

#-------------------------------------------------------------------------------
  def set_tfun(self, tfun=None, *args, **kwds):
    """ Sets a two-length tuple of functions that should correspond to the
    (cumulative probability function, inverse cumulative function) with respect
    to the callable function set by set_tran(). It is necessary to set these
    functions for conditional sampling variables with non-flat distributions.

    :param tfun: two-length tuple of callable functions or an LU decomposition
    :param *args: arguments to pass to tfun functions
    :param **kwds: keywords to pass to tfun functions
    """
    scipyobj = None if 'scipyobj' not in kwds else kwds['scipyobj']
    self._tfun = tfun 
    if self._tfun is None:
      return

    # Pass CF objects directly
    if isinstance(tfun, CF):
      assert not args and not kwds, \
        "Setting args and kwds not supported if inputting a CF instance"
      self._tfun = CF(self, 
                      tfun.ret_inp(), 
                      tfun.ret_func(), 
                      *tfun.ret_args(), 
                      **tfun.ret_kwds())
      return

    # Handle SciPy objects specifically
    elif scipyobj is not None:      
      rvs = self.ret_rvs(aslist=True)
      lims = np.array([rv.ret_lims() for rv in rvs])
      mean = scipyobj.mean
      cov = scipyobj.cov
      self._cond_cov = CondCov(mean, cov, lims)
      scipy_cond = sample_cond_cov if self._sym_tran else \
                   (sample_cond_cov, sample_cond_cov)
      self._tfun = Func(scipy_cond, cond_cov=self._cond_cov)
      return

    # Non-callables are treated as LUD matrices
    elif not callable(self._tfun): 
        self._tfun = Func(self._tfun, *args, **kwds)
        message = "Non-callable tran objects must be a triangular 2D Numpy array " + \
                  "of size corresponding to number of variables {}".format(self._nrvs)
        assert isinstance(tfun, np.ndarray), message
        assert tfun.ndim == 2, message
        assert np.all(np.array(tfun.shape) == self._nrvs), message
        assert np.allclose(tfun, np.tril(tfun)) or \
               np.allclose(tfun, np.triu(tfun)), message
        return

    # Otherwise instantiate a formal conditional function
    inp = None
    if len(args):
      if isinstance(args[0], collections.OrderedDict):
        args = tuple(args)
        inp, args = args[0], args[1:]
    self._tfun = CF(self, inp, self._tfun, *args, **kwds)

#-------------------------------------------------------------------------------
  def ret_rvs(self, aslist=True):
    """ Returns the RVs belonging to the random field either as a list
    (by default) or as a dictionary {rv_name: rv_instance}. 

    :param aslist: Boolean flag (default True) to return RVs as a list

    :return: RVs an OrderedDict or a list.
    """
    rvs_data = collections.OrderedDict(self.nodes.data())
    for key, val in rvs_data.items():
      if 'rv' not in val: # quick debug check
        import pdb; pdb.set_trace()
    rvs = collections.OrderedDict({key:val['rv'] 
                                       for key,val in rvs_data.items()})
    if aslist:
      if isinstance(rvs, dict):
        rvs = list(rvs.values())
      assert isinstance(rvs, list), "RVs not a recognised variable type: {}".\
                                    format(type(rvs))
    return rvs

#-------------------------------------------------------------------------------
  def eval_length(self):
    """ Evaluates and returns the joint length of the random junction. """
    rvs = self.ret_rvs(aslist=True)
    self._lengths = np.array([rv.ret_length() for rv in rvs], dtype=float)
    self._length = np.sqrt(np.sum(self._lengths))
    return self._length

#-------------------------------------------------------------------------------
  def ret_length(self):
    """ Returns the length of the random junction """
    return self._length

#-------------------------------------------------------------------------------
  def ret_name(self):
    """ Returns the name of the random junction """
    return self._name

#-------------------------------------------------------------------------------
  def ret_id(self):
    """ Returns the id of the random junction """
    return self._id
#-------------------------------------------------------------------------------
  def ret_nrvs(self):
    """ Returns the number of random variables belonging to the random junction.
    """
    return self._nrvs

#-------------------------------------------------------------------------------
  def ret_keys(self, aslist=True):
    """ Returns the RV keys as a list (by default) otherwise as a set """
    if aslist:
      return self._keys
    return self._keyset

#-------------------------------------------------------------------------------
  def ret_pscale(self):
    """ Returns the real or complex scaling constant set for pscale """
    return self._pscale

#-------------------------------------------------------------------------------
  def parse_args(self, *args, **kwds):
    """ Returns (values, iid) from *args and **kwds """
    args = tuple(args)
    kwds = dict(kwds)
    pass_all = False if 'pass_all' not in kwds else kwds.pop('pass_all')
    
    if not args and not kwds:
      args = (None,)
    if args:
      assert len(args) == 1 and not kwds, \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
      kwds = dict(args[0]) if isinstance(args[0], dict) else \
             ({key: args[0] for key in self._keys})

    elif kwds:
      assert not args, \
        "With order specified, calls argument must be a single " + \
              "dictionary or keywords only"
    values = dict(kwds)
    seen_keys = []
    for key, val in values.items():
      count_comma = key.count(',')
      if count_comma:
        seen_keys.extend(key.split(','))
        if isinstance(val, (tuple, list)):
          assert len(val) == count_comma+1, \
              "Mismatch in key specification {} and number of values {}".\
              format(key, len(val))
        else:
          values.update({key: [val] * (count_comma+1)})
      else:
        seen_keys.append(key)
      if not pass_all:
        assert seen_keys[-1] in self._keys, \
            "Unrecognised key {} among available RVs {}".format(
                seen_keys[-1], self._keys)
    for key in self._keys:
      if key not in seen_keys:
        values.update({key: None})
    if pass_all:
      list_keys = list(values.keys())
      for key in list_keys:
        if key not in self._keys:
          values.pop(key)

    return values

#-------------------------------------------------------------------------------
  def eval_dist_name(self, values=None, suffix=None):
    # Evaluates the string used to set the distribution name
    vals = collections.OrderedDict()
    if isinstance(values, dict):
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
    else:
      vals.update({key: values for key in self._keys})
    rvs = self.ret_rvs()
    rv_dist_names = [rv.eval_dist_name(vals[rv.ret_name()], suffix) \
                     for rv in rvs]
    dist_name = ','.join(rv_dist_names)
    return dist_name

#-------------------------------------------------------------------------------
  def eval_vals(self, *args, _skip_parsing=False, min_dim=0, **kwds):
    """ 
    Keep args and kwds since could be called externally. This ignores self._prob.
    """
    values = self.parse_args(*args, **kwds) if not _skip_parsing else args[0]
    dims = {}
    
    # Don't reshape if all scalars (and therefore by definition no shared keys)
    if all([np.isscalar(value) for value in values.values()]): # use np.scalar
      return values, dims

    # Create reference mapping for shared keys across rvs
    values_ref = collections.OrderedDict({key: [key, None] for key in self._keys})
    for key in values.keys():
      if ',' in key:
        subkeys = key.split(',')
        for i, subkey in enumerate(subkeys):
          values_ref[subkey] = [key, i]

    # Share dimensions for joint variables and do not dimension scalars
    ndim = min_dim
    dims = collections.OrderedDict({key: None for key in self._keys})
    seen_keys = set()
    for i, key in enumerate(self._keys):
      new_dim = False
      if values_ref[key][1] is None: # i.e. not shared
        if not isdimensionless(values[key]):
          dims[key] = ndim
          new_dim = True
        seen_keys.add(key)
      elif key not in seen_keys:
        val_ref = values_ref[key]
        subkeys = val_ref[0].split(',')
        for subkey in subkeys:
          dims[subkey] = ndim
          seen_keys.add(subkey)
        if not isdimensionless(values[val_ref[0]][val_ref[1]]):
          new_dim = True
      if new_dim:
        ndim += 1

    # Reshape
    vdims = [dim for dim in dims.values() if dim is not None]
    ndims = max(vdims) + 1 if len(vdims) else 0
    ones_ndims = np.ones(ndims, dtype=int)
    vals = collections.OrderedDict()
    rvs = self.ret_rvs(aslist=True)
    for i, rv in enumerate(rvs):
      key = rv.ret_name()
      reshape = True
      if key in values.keys():
        vals.update({key: values[key]})
        reshape = not np.isscalar(vals[key])
        if vals[key] is None or isinstance(vals[key], set):
          vals[key] = rv.eval_vals(vals[key])
      else:
        val_ref = values_ref[key]
        vals_val = values[val_ref[0]][val_ref[1]]
        if vals_val is None or isinstance(vals_val, set):
          vals_val = rv.eval_vals(vals_val)
        vals.update({key: vals_val})
      if reshape and not isscalar(vals[key]):
        re_shape = np.copy(ones_ndims)
        re_dim = dims[key]
        re_shape[re_dim] = vals[key].size
        vals[key] = vals[key].reshape(re_shape)
    
    # Remove dimensionality for singletons
    for key in self._keys:
      if issingleton(vals[key]):
        dims[key] = None
    return vals, dims

#-------------------------------------------------------------------------------
  def eval_prob(self, values=None, dims=None):
    if values is None:
      values = {}
    else:
      assert isinstance(values, dict), \
          "Input to eval_prob() requires values dict"
      assert set(values.keys()) == self._keyset, \
        "Sample dictionary keys {} mismatch with RV names {}".format(
          values.keys(), self._keys)

    # If not specified, treat as independent variables
    if self._prob is None:
      rvs = self.ret_rvs(aslist=True)
      if len(rvs) == 1 and rvs[0]._prob is not None:
        prob = rvs[0].eval_prob(values[rvs[0].ret_name()])
      else:
        prob, _ = rv_prod_rule(values, rvs=rvs, pscale=self._pscale)
      return prob

    # Otherwise distinguish between uncallable and callables
    if not self.__callable:
      return self._prob()
    elif isinstance(self._prob, Func) and self._prob.ret_isscipy():
      return call_scipy_prob(self._prob, self._pscale, values)
    if self.__passdims:
      return self._prob(values, dims=dims)
    return self._prob(values)

#-------------------------------------------------------------------------------
  def eval_delta(self, delta=None):

    # Handle native delta types within RV deltas
    if delta is None: 
      if self._delta is None:
        rvs = self.ret_rvs(aslist=True)
        if len(rvs) == 1 and rvs[0]._delta is not None:
          return rvs[0]._delta
        return None
      elif isinstance(self._delta, Func):
        delta = self._delta()
      elif isinstance(self._delta, self._delta_type):
        delta_dict = collections.OrderedDict()
        rvs = self.ret_rvs(aslist=True)
        for i, key in enumerate(self._keys):
          delta_dict.update({key: rvs[i].eval_delta()})
        delta = self._delta_type(**delta_dict)
      else:
        delta = self._delta
    elif isinstance(delta, Func):
      delta = delta()
    elif isinstance(delta, self._delta_type):
      delta_dict = collections.OrderedDict()
      rvs = self.ret_rvs(aslist=True)
      for i, key in enumerate(self._keys):
        delta_dict.update({key: rvs[i].eval_delta(delta[i])})
      delta = self._delta_type(**delta_dict)

    # Non spherical case
    if not isinstance(self._delta_type, Func) and \
         isinstance(delta, self._delta_type): # i.e. non-spherical
      if self._tfun is None or self._tfun.ret_isscalar():
        return delta
      elif not self._tfun.ret_callable():
        delta = np.ravel(delta)
        delta = self._tfun().dot(delta)
        return self._delta_type(*delta)
      else:
        delta = self._tfun(delta)
        assert isinstance(delta, self._delta_type), \
            "User supplied tfun did not output delta type {}".\
            format(self._delta_type)
        return delta

    # Rule out possibility of all RVs contained in unscaling argument
    assert isinstance(delta, tuple), \
        "Unknown delta type: {}".format(delta)
    unscale = {} if not self._delta_args else self._delta_args
    if not len(self._spherise):
      return self._delta_type(**unscale)

    # Spherical version
    delta = delta[0]
    spherise = self._spherise
    keys = self._spherise.keys()
    rss = real_sqrt(np.sum(np.array(list(spherise.values()))**2))
    if self._delta_kwds['scale']:
      delta *= rss
    deltas = np.random.uniform(-delta, delta, size=len(spherise))
    rss_deltas = real_sqrt(np.sum(deltas ** 2.))
    deltas = (deltas * delta) / rss_deltas
    delta_dict = collections.OrderedDict()
    rvs = [self[key] for key in keys]
    idx = 0
    for i, key in enumerate(keys):
      if key in unscale:
        val = unscale[key]
      else:
        val = deltas[idx]
        idx += 1
        if self._delta_kwds['scale']:
          val *= self._lengths[i]
      delta_dict.update({key: val})
    delta = self._delta_type(**delta_dict)
    if self._tfun is None or self._tfun.ret_isscalar():
      return delta
    elif not self._tfun.ret_callable():
      delta = self._tfun().dot(np.array(delta, dtype=float))
      return self._delta_type(*delta)
    else:
      delta = self._tfun(delta)
      assert isinstance(delta, self._delta_type), \
          "User supplied tfun did not output delta type {}".\
          format(self._delta_type)
      return delta

#-------------------------------------------------------------------------------
  def apply_delta(self, values, delta=None):
    delta = delta or self._delta
    if delta is None:
      return values
    if not isinstance(delta, self._delta_type):
      rvs = self.ret_rvs(aslist=True)
      if len(rvs) == 1 and isinstance(delta, rvs[0].delta):
        return rvs[0].apply_delta(values, delta)
      raise TypeError("Cannot apply delta without providing delta type {}".\
        format(self._delta_type))
    bound = False if 'bound' not in self._delta_kwds \
           else self._delta_kwds['bound']
    vals = collections.OrderedDict(values)
    keys = delta._fields
    rvs = [self[key] for key in keys]
    for i, key in enumerate(keys):
      vals.update({key: rvs[i].apply_delta(values[key], delta[i], bound=bound)})
    return vals

#-------------------------------------------------------------------------------
  def eval_prop(self, values, **kwargs):
    if self._tran is not None:
      return self.eval_tran(values, **kwargs)
    if values is None:
      values = {}
    if self._prop is None:
      return self.eval_prob(values, **kwargs)
    if not self._prop.ret_callable():
      return self._prop()
    return self._prop(values)

#-------------------------------------------------------------------------------
  def eval_step(self, pred_vals, succ_vals, reverse=False):
    """ Returns adjusted succ_vals """

    # Evaluate deltas if required
    if succ_vals is None:
      if self._delta is None:
        pred_values = list(pred_vals.values())
        if all([isscalar(pred_value) for pred_value in pred_values]):
          succ_vals = {0}
        else:
          succ_vals = pred_vals
      else:
        succ_vals = self.eval_delta()
    elif isinstance(succ_vals, Func) or \
        isinstance(succ_vals, (tuple, self._delta_type)):
      succ_vals = self.eval_delta(succ_vals)

    # Apply deltas
    cond = None
    if isinstance(succ_vals, self._delta_type):
      succ_vals = self.apply_delta(pred_vals, succ_vals)
    elif isunitsetint(succ_vals):
      if self._tfun is not None and self._tfun.ret_callable():
        number = list(succ_vals)[0]
        succ_vals = collections.OrderedDict()
        succ_lims = collections.OrderedDict()
        succ_vset = collections.OrderedDict()
        for key in self._keys:
          if key in pred_vals:
            succ_vals.update({key: pred_vals[key]})
            lims = self[key].ret_lims()
            assert np.all(np.isfinite(lims)), \
                "Cannot sample RV {} exhibiting limits {}".\
                format(key, lims)
            succ_vals.update({key: pred_vals[key]})
            succ_lims.update({key: lims})
            succ_vset.update({key: self[key].ret_vset()})
      elif self._nrvs == 1:
        rv = self.ret_rvs(aslist=True)[0]
        tran = rv.ret_tran()
        tfun = rv.ret_tfun()
        if (tran is not None and not tran.ret_callable()) or \
            (tfun is not None and tfun.ret_callable()):
          vals, dims, kwargs = rv.eval_step(pred_vals[rv.ret_name()], succ_vals, reverse=reverse)
          return vals, dims, kwargs
        raise ValueError("Transitional CDF calling requires callable tfun")
      else:
        raise ValueError("Transitional CDF calling requires callable tfun")
      keys = list(succ_vals.keys())

      # One variable at a time modification
      if self._tsteps:
        if self.__cond_mod is None:
          self.__cond_mod = 0
        keys = keys[self.__cond_mod:(self.__cond_mod+self._tsteps)]
        self.__cond_mod += self._tsteps
        if self.__cond_mod >= len(succ_vals):
          self.__cond_mod = 0

      # Iterate through values
      cond = np.nan
      if self._tfun.ret_ismulti():
        for key in keys:
          succ_vals[key] = {0}
          succ_vals[key] = self._tfun[int(reverse)](succ_vals)
      else:
        for key in keys:
          succ_vals[key] = {0}
          succ_vals[key] = self._tfun(succ_vals)

    # Initialise outputs with predecessor values
    dims = {}
    kwargs = {'reverse': reverse}
    if cond is not None:
      kwargs = {'cond': cond}
    vals = collections.OrderedDict()
    for key in self._keys:
      vals.update({key: pred_vals[key]})
    if succ_vals is None and self._tran is None:
      return vals, dims, kwargs

    # If stepping or have a transition function, add successor values
    for key in self._keys:
      mod_key = key+"'"
      succ_key = key if mod_key not in succ_vals else mod_key
      vals.update({key+"'": succ_vals[succ_key]})

    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def eval_tran(self, values, **kwargs):
    if 'cond' in kwargs:
      return kwargs['cond']
    reverse = False if 'reverse' not in kwargs else kwargs['reverse']
    if self._tran is None:
      rvs = self.ret_rvs(aslist=True)
      if len(rvs) == 1 and rvs[0]._tran is not None:
        return rvs[0].eval_tran(values, **kwargs)
      pred_vals = dict()
      succ_vals = dict()
      for key_, val in values.items():
        prime = key_[-1] == "'"
        key = key_[:-1] if prime else key_
        if key in self._keys:
          if prime:
            succ_vals.update({key: val})
          else:
            pred_vals.update({key: val})
      cond, _ = rv_prod_rule(pred_vals, succ_vals, rvs=rvs, pscale=self._pscale)
    elif self._tran.ret_isscalar():
      cond = self._tran()
    elif not self._tran.ret_callable():
      cond = 0. if iscomplex(self._pscale) else 1.
    else:
      cond = self._tran(values) if self._sym_tran else \
             self._tran[int(reverse)](values)
    return cond

#-------------------------------------------------------------------------------
  def reval_tran(self, dist):
    """ Evaluates the conditional reverse-transition function for corresponding 
    transition conditional distribution dist. This requires a tuple input for
    self.set_tran() to evaluate a new conditional.
    """
    assert isinstance(dist, Dist), \
        "Input must be a distribution, not {} type.".format(type(dist))
    marg, cond = dist.cond, dist.marg
    name = margcond_str(marg, cond)
    vals = dist.vals
    dims = dist.dims
    prob = dist.prob if self._sym_tran else self._tran[1](dist.vals)
    pscale = dist.ret_pscale()
    return Dist(name, vals, dims, prob, pscale)

#-------------------------------------------------------------------------------
  def subfield(self, vertices):
    """ Returns a view of vertices, which must all be members, as an RF.

    :param vertices: str/RV/RF or list/tuple of str/RVs/RF to include in RF

    :return: an RF including only those vertices
    """

    # Convert to list
    if isinstance(vertices, tuple):
      vertices = list(vertices)
    if not isinstance(vertices, list):
      vertices = [vertices]

    # Collate RVs and return as RF
    rvs = []
    rv_dict = self.ret_rvs(aslist=False)
    for vertex in vertices:
      if isinstance(vertex, RF):
        rvs += [vertex.ret_rvs(aslist=True)]
      elif isinstance(vertex, RV):
        rvs += [vertex]
      elif isinstance(vertex, str):
        rvs += [rv_dict[vertex]]
      else:
        raise TypeError("Unrecognised vertex specification type: {}".format(
            type(vertex)))
    return RF(*tuple(rvs))

#-------------------------------------------------------------------------------
  def _eval_iid(self, dist_name, vals, dims, prob, iid):
    if not iid: 
      return Dist(dist_name, vals, dims, prob, self._pscale)

    # Deal with IID cases
    max_dim = None
    for dim in dims.values():
      if dim is not None:
        max_dim = dim if max_dim is None else max(dim, max_dim)

    # If scalar or prob is expected shape then perform product here
    if max_dim is None or max_dim == prob.ndim - 1:
      dist = Dist(dist_name, vals, dims, prob, self._pscale)
      return dist.prod(iid)

    # Otherwise it is left to the user function to perform the iid product
    for key in iid:
      vals[key] = {len(vals[key])}
      dims[key] = None

    # Tidy up probability
    return Dist(dist_name, vals, dims, prob, self._pscale)

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Returns a joint distribution p(args) """
    if not self._nrvs:
      return None
    iid = False if 'iid' not in kwds else kwds.pop('iid')
    if type(iid) is bool and iid:
      iid = self._defiid
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values)
    vals, dims = self.eval_vals(values, _skip_parsing=True)
    prob = self.eval_prob(vals, dims)
    return self._eval_iid(dist_name, vals, dims, prob, iid)

#-------------------------------------------------------------------------------
  def propose(self, *args, **kwds):
    """ Returns a proposal distribution p(args[0]) for values """
    suffix = "'" if 'suffix' not in kwds else kwds.pop('suffix')
    values = self.parse_args(*args, **kwds)
    dist_name = self.eval_dist_name(values, suffix)
    vals, dims = self.eval_vals(values, _skip_parsing=True)
    prop = self.eval_prop(vals) if self._prop is not None else \
           self.eval_prob(vals, dims)
    if suffix:
      keys = list(vals.keys())
      for key in keys:
        mod_key = key + suffix
        vals.update({mod_key: vals.pop(key)})
        if key in dims:
          dims.update({mod_key: dims.pop(key)})
    return Dist(dist_name, vals, dims, prop, self._pscale)

#-------------------------------------------------------------------------------
  def step(self, *args, **kwds):
    """ Returns a proposal distribution p(args[1]) given args[0], depending on
    whether using self._prop, that denotes a simple proposal distribution,
    or self._tran, that denotes a transitional distirbution. """

    reverse = False if 'reverse' not in kwds else kwds.pop('reverse')
    pred_vals, succ_vals = None, None 
    if len(args) == 1:
      if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        pred_vals, succ_vals = args[0][0], args[0][1]
      else:
        pred_vals = args[0]
    elif len(args) == 2:
      pred_vals, succ_vals = args[0], args[1]

    # Evaluate predecessor values
    pred_vals = self.parse_args(pred_vals, pass_all=True)
    dist_pred_name = self.eval_dist_name(pred_vals)
    pred_vals, pred_dims = self.eval_vals(pred_vals)

    # Default successor values if None and delta is None
    if succ_vals is None and self._delta is None:
      pred_values = list(pred_vals.values())
      if all([isscalar(pred_value) for pred_value in pred_values]):
        succ_vals = {0}
      else:
        succ_vals = pred_vals

    # Evaluate successor evaluates
    vals, dims, kwargs = self.eval_step(pred_vals, succ_vals, reverse=reverse)
    succ_vals = {key[:-1]: val for key, val in vals.items() if key[-1] == "'"}
    cond = self.eval_tran(vals, **kwargs)
    dist_succ_name = self.eval_dist_name(succ_vals, "'")
    dist_name = '|'.join([dist_succ_name, dist_pred_name])

    return Dist(dist_name, vals, dims, cond, self._pscale)

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._nrvs

#-------------------------------------------------------------------------------
  def __eq__(self, other):
    """ Equality for RFs is defined as comprising the same RVs """
    if type(self) is not RF or type(other) is not RF:
      return super().__eq__(other)
    return self.ret_keys(aslist=False) == other.ret_keys(aslist==False)

#-------------------------------------------------------------------------------
  def __ne__(self, other):
    """ Equality for RFs is defined as comprising the same RVs """
    return not self.__eq__(other)

#-------------------------------------------------------------------------------
  def ret_prob(self):
    """ Returns object set by set_prob() """
    return self._prob

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = self._keys[key]
    if isinstance(key, str):
      if key not in self._keys:
        return None
    return self.ret_rvs(False)[key]

#-------------------------------------------------------------------------------
  def __repr__(self):
    if not self._name:
      return super().__repr__()
    return super().__repr__() + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
  def __mul__(self, other):
    from probayes.rv import RV
    from probayes.sd import SD
    if isinstance(other, SD):
      leafs = self.ret_rvs() + other.ret_leafs().ret_rvs()
      stems = other.ret_stems()
      roots = other.ret_roots()
      args = RF(*tuple(leafs))
      if stems:
        args += list(stems.values())
      if roots:
        args += roots.ret_rvs()
      return SD(*tuple(args))

    if isinstance(other, RF):
      rvs = self.ret_rvs() + other.ret_rvs()
      return RF(*tuple(rvs))

    if isinstance(other, RV):
      rvs = self.ret_rvs() + [other]
      return RF(*tuple(rvs))

    raise TypeError("Unrecognised post-operand type {}".format(type(other)))

#-------------------------------------------------------------------------------
  def __truediv__(self, other):
    """ Conditional operator between RF and another RV, RF, or SD. """
    from probayes.sd import SD
    return SD(self, other)

#-------------------------------------------------------------------------------
