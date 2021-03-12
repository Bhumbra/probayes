"""
A field is a collection of a functionally-related variables without explictly
directional dependences.
"""
#-------------------------------------------------------------------------------
import collections
import numpy as np

from probayes.variable import Variable
from probayes.variable_utils import parse_as_str_dict
from probayes.vtypes import isscalar, isunitsetint, issingleton, isdimensionless
from probayes.pscales import real_sqrt
from probayes.ops import and_op, or_op
from probayes.expression import Expression
from probayes.functional import NX_UNDIRECTED_GRAPH
from probayes.distribution import Distribution

#-------------------------------------------------------------------------------
class Field (NX_UNDIRECTED_GRAPH):
  """
  A field is a collection of a variables that are functionally inter-related
  without explicit directional dependencies. This lack of directionality means
  Field does not support Functionals. 
  """

  # Protected
  _name = None       # Random field name cannot be set externally
  _Delta = None      # A namedtuple generator for delta operations
  _vars = None       # OrderedDict of variables
  _nvars = None      # Number of variables
  _unitvar = None    # Boolean flag for one variable
  _anyfloat = None   # Boolean flag to denote if any variables are float type
  _anystoch = None   # Boolean flag of whether any variable stochastic
  _varlist = None    # List of variable objects
  _keylist = None    # List of keys of variable names
  _keyset = None     # Unordered set of keys of random variable names
  _delta = None      # Delta function (to replace step)
  _delta_args = None # Optional delta args (must be dictionaries)
  _delta_kwds = None # Optional delta kwds
  _delta_type = None # Proxy for delta used for casting
  _length = None     # Length of field
  _lengths = None    # Lengths of Variables
  _leafs = None      # Field of Variables that do not condition others (for SD)
  _roots = None      # Field of Vairables not dependent on others (for SD)
  _stems = None      # OrderedDict of latent Variables (for Dependencies)
  _is_stochastic = False      # Flag of whether stochastic
  _func = False      # Functional

#-------------------------------------------------------------------------------
  @property
  def is_stochastic(self):
    return self._is_stochastic

  def __init__(self, *args): # over-rides NX_GRAPH.__init__()
    """ Initialises a random field with Variables for in args. See set_vars(). """
    super().__init__()
    self.set_vars(*args)
    self.set_delta()

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
  @property
  def leafs(self):
    return self._leafs

  @property
  def stems(self):
    return self._stems

  @property
  def roots(self):
    return self._roots

  def set_vars(self, *args):
    """ Initialises a field with variables for each arg in args.

    :param *args: each arg may be an Variable instance or the first arg may be a Field.
    
    """
    self._leafs, self._stems, self._roots = self, None, None
    if len(args) == 1 and isinstance(args[0], (Variable, set, tuple, list)):
      args = args[0]
    else:
      args = tuple(args)
    self.add_var(args)

#-------------------------------------------------------------------------------
  def add_var(self, var):
    """ Adds one or more variables to the field.

    :param var: a Variable or Field instance, or a list/tuple of Variable instances.
    """
    variables = []
    if isinstance(var, (Field, dict, set, tuple, list)):
      variables = var
      if isinstance(variables, Field):
        variables = variables.varlist
      elif isinstance(variables, dict):
        variables = list(variables.values())
    for var in variables:
      assert isinstance(var, Variable), \
          "Input not a variable instance but of type: {}".format(type(var))
      if self._nvars:
        assert var not in list(self.nodes), \
            "Existing variable {} already present in collection".format(var)
      super().add_node(var)
    else:
      assert isinstance(var, Variable), \
          "Input not a variable instance but of type: {}".format(type(var))
      if self._nvars:
        assert var not in list(self.nodes), \
            "Existing variable {} already present in collection".format(var)
      super().add_node(var)
    self._refresh()

#-------------------------------------------------------------------------------
  @property
  def vars(self):
    return self._vars

  @property
  def nvars(self):
    return self._nvars

  @property
  def unitvar(self):
    return self._unitvar

  @property
  def anyfloat(self):
    return self._anyfloat

  @property
  def anystoch(self):
    return self._anystoch

  def __len__(self):
    return self._nvars

  def __bool__(self):
    if self._nvars:
      return True
    return False

  @property
  def varlist(self):
    return self._varlist

  @property
  def keylist(self):
    return self._keylist

  @property
  def keyset(self):
    return self._keyset

  @property
  def name(self):
    return self._name

  @property
  def id(self):
    return self._id

  @property
  def Delta(self):
    return self._Delta

  def _refresh(self):
    """ Updates Variable summary objects, Field name and id, and delta factory. """
    self._vars = collections.OrderedDict({var.name: var for var in list(self.nodes)})
    self._nvars = self.number_of_nodes()
    self._unitvar = self._nvars == 1
    self._varlist = list(self._vars.values())
    self._anyfloat = any([var.vtype is float for var in self._varlist])
    self._anystoch = any([var.is_stochastic for var in self._varlist])
    if not self._is_stochastic:
      assert not self._anystoch,\
          "Stochastic variables cannot contribute to a non-stochastic field"
    self._keylist = list(self._vars.keys())
    self._keyset = frozenset(self._keylist)
    self._defiid = self._leafs.keylist
    self._name = ','.join(self._keylist)
    self._id = '_and_'.join(self._keylist)
    if self._id:
      self._Delta = collections.namedtuple('ð', self._keylist)
      self._delta_type = self._Delta
    self.eval_length()

#-------------------------------------------------------------------------------
  @property
  def delta(self):
    """ Returns the default delta object if specified """
    return self._delta

  def set_delta(self, delta=None, *args, **kwds):
    """ Sets the default delta function or operation.

    :param delta: the delta function or operation (see below)
    :param *args: optional arguments to pass if delta is callable.
    :param **kwds: optional keywords to pass if delta is callable.

    The input delta may be:

    1. A callable function (for which args and kwds are passed on as usual).
    2. An Variable.Delta instance (this defaults all Variable Deltas).
    3. A dictionary for Variables, this is converted to an Field.Delta.
    4. A scalar that may contained in a list or tuple:
      a) No container - the scalar is treated as a fixed delta.
      b) List - delta is uniformly and independently sampled across Variabless.
      c) Tuple - delta is spherically sampled across Variables.

      For non-tuples, an optional argument (args[0]) can be included as a 
      dictionary to specify by Variable-name deltas following the above 
      conventions except their values are not subject to scaling even if 
      'scale' is given, but they are subject to bounding if 'bound' is 
      specified.

    For setting types 2-4, optional keywords are (default False):
      'scale': Flag to denote scaling deltas to Variable lengths
      'bound': Flag to constrain delta effects to Variable bounds (None bounces)
      
    """
    self._delta = delta
    self._delta_args = args
    self._delta_kwds = dict(kwds)
    self._spherise = {}
    if self._delta is None:
      return
    elif callable(self._delta):
      self._delta = Expression(self._delta, *args, **kwds)
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
      for i, var in enumerate(self._varlist):
        var.set_delta(self._delta[i], scale=scale, bound=bound)
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
      delta = self._delta 
      urand = isinstance(delta, list)
      if urand:
        assert len(delta) == 1, "List delta requires a single element"
        delta = delta[0]
      deltas = {key: delta for key in self._keylist}
      unscale = {} if not self._delta_args else self._delta_args[0]
      deltas.update(unscale)
      delta_dict = collections.OrderedDict(deltas)
      for i, (key, val) in enumerate(deltas.items()):
        delta = val
        if scale and key not in unscale:
          assert np.isfinite(self._lengths[i]), \
              "Cannot scale by infinite length for Variable {}".format(key)
          delta = val * self._lengths[i]
        if urand:
          delta = [delta]
        delta_dict.update({key: delta})
      self._delta = self._delta_type(**delta_dict)
      for i, var in enumerate(self._varlist):
        var.set_delta(self._delta[i], scale=False, bound=bound)

    # Tuple deltas must be evaluated on-the-fly and cannot be pre-scaled
    else:
      unscale = {} if not self._delta_args else self._delta_args[0]
      self._spherise = {}
      for i, key in enumerate(self._keylist):
        if key not in unscale.keys():
          length = self._lengths[i]
          assert np.isfinite(length), \
              "Cannot spherise Variable {} with infinite length".format(key)
          self._spherise.update({key: length})

#-------------------------------------------------------------------------------
  def eval_length(self):
    """ Evaluates and returns the joint length of the field. """
    self._lengths = np.array([var.length for var in self._varlist], 
                             dtype=float)
    self._length = np.sqrt(np.sum(self._lengths**2))
    return self._length

#-------------------------------------------------------------------------------
  def parse_args(self, *args, **kwds):
    """ Returns (values, iid) from *args and **kwds """
    pass_all = False if 'pass_all' not in kwds else kwds.pop('pass_all')
    values = parse_as_str_dict(*args, **kwds)
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
        assert seen_keys[-1] in self._keyset, \
            "Unrecognised key {} among available Variables {}".format(
                seen_keys[-1], self._keyset)

    # Default values
    def_val = None
    if self._anyfloat:
      if not len(values):
        def_val = {0}
      else:
        def_val = {0}
        for val in values.values():
          if isunitsetint(val):
            if val == {0}:
              def_val = None
    for key in self._keylist:
      if key not in seen_keys:
        values.update({key: def_val})
    if pass_all:
      list_keys = list(values.keys())
      for key in list_keys:
        if key not in self._keylist:
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
      for key in self._keylist:
        if key not in vals.keys():
          vals.update({key: None})
    else:
      vals.update({key: values for key in self._keylist})
    var_dist_names = [var.eval_dist_name(vals[var.name], suffix) \
                     for var in self._varlist]
    dist_name = ','.join(var_dist_names)
    return dist_name

#-------------------------------------------------------------------------------
  def evaluate(self, *args, _skip_parsing=False, min_dim=0, **kwds):
    """ 
    Keep args and kwds since could be called externally. This ignores self._prob.
    """
    values = self.parse_args(*args, **kwds) if not _skip_parsing else args[0]
    dims = collections.OrderedDict()
    
    # Don't reshape if all scalars (and therefore by definition no shared keys)
    if all([np.isscalar(value) for value in values.values()]): # use np.scalar
      return values, dims

    # Create reference mapping for shared keys across vars
    values_ref = collections.OrderedDict({key: [key, None] for key in self._keylist})
    for key in values.keys():
      if ',' in key:
        subkeys = key.split(',')
        for i, subkey in enumerate(subkeys):
          values_ref[subkey] = [key, i]

    # Share dimensions for joint variables and do not dimensionalise scalars
    ndim = min_dim
    dims = collections.OrderedDict({key: None for key in self._keylist})
    seen_keys = set()
    for i, key in enumerate(self._keylist):
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
    for i, var in enumerate(self._varlist):
      key = var.name
      reshape = True
      if key in values.keys():
        vals.update({key: values[key]})
        reshape = not np.isscalar(vals[key])
        if vals[key] is None or isinstance(vals[key], set):
          vals.update(var.evaluate(vals[key]))
      else:
        val_ref = values_ref[key]
        vals_val = values[val_ref[0]][val_ref[1]]
        if vals_val is None or isinstance(vals_val, set):
          vals.update(var.evaluate(vals_val))
        else:
          vals.update({key: vals_val})
      if reshape and not isscalar(vals[key]):
        re_shape = np.copy(ones_ndims)
        re_dim = dims[key]
        re_shape[re_dim] = vals[key].size
        vals[key] = vals[key].reshape(re_shape)
    
    # Remove dimensionality for singletons
    for key in self._keylist:
      if issingleton(vals[key]):
        dims[key] = None
    return vals, dims

#-------------------------------------------------------------------------------
  def eval_delta(self, delta=None):

    # Handle native delta types within Variable deltas
    if delta is None: 
      if self._delta is None:
        variables = list(self._varlist)
        if len(variables) == 1 and variables[0]._delta is not None:
          return variables[0]._delta
        return None
      elif isinstance(self._delta, Expression):
        delta = self._delta()
      elif isinstance(self._delta, self._delta_type):
        delta_dict = collections.OrderedDict()
        variables = self._varlist
        for i, key in enumerate(self._keylist):
          delta_dict.update({key: variables[i].eval_delta()})
        delta = self._delta_type(**delta_dict)
      else:
        delta = self._delta
    elif isinstance(delta, Expression):
      delta = delta()
    elif isinstance(delta, self._delta_type):
      delta_dict = collections.OrderedDict()
      variables = self.ret_vars(aslist=True)
      for i, key in enumerate(self._keylist):
        delta_dict.update({key: variables[i].eval_delta(delta[i])})
      delta = self._delta_type(**delta_dict)

    # Non spherical case
    if not isinstance(self._delta_type, Expression) and \
         isinstance(delta, self._delta_type): # i.e. non-spherical
      return delta

    # Rule out possibility of all variables contained in unscaling argument
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
    return delta

#-------------------------------------------------------------------------------
  def apply_delta(self, values, delta=None):
    delta = delta or self._delta
    if delta is None:
      return values
    if not isinstance(delta, self._delta_type):
      variables = self._varlist
      if len(variables) == 1 and isinstance(delta, variables[0].delta):
        return variables[0].apply_delta(values, delta)
      raise TypeError("Cannot apply delta without providing delta type {}".\
        format(self._delta_type))
    bound = False if 'bound' not in self._delta_kwds \
           else self._delta_kwds['bound']
    vals = collections.OrderedDict(values)
    keys = delta._fields
    for i, key in enumerate(keys):
      vals.update({key: self._vars[key].apply_delta(values[key], 
                                                    delta[i], 
                                                    bound=bound)})
    return vals

#-------------------------------------------------------------------------------
  def eval_step(self, pred_vals, succ_vals, reverse=False):
    """ Returns adjusted succ_vals """

    # Evaluate deltas if required
    if succ_vals is None:
      if self._delta is None:
        pred_values = list(pred_vals.values())
        if all([isscalar(pred_value) for pred_value in pred_values]):
          raise ValueError("Stochastic step sampling not supported for Field; use RF")
        else:
          succ_vals = pred_vals
      else:
        succ_vals = self.eval_delta()
    elif isinstance(succ_vals, Expression) or \
        isinstance(succ_vals, (tuple, self._delta_type)):
      succ_vals = self.eval_delta(succ_vals)

    # Apply deltas
    cond = None
    if isinstance(succ_vals, self._delta_type):
      succ_vals = self.apply_delta(pred_vals, succ_vals)
    elif isunitsetint(succ_vals):
      raise ValueError("Stochastic step sampling not supported for Field; use RF")

    # Initialise outputs with predecessor values
    dims = {}
    kwargs = {'reverse': reverse}
    if cond is not None:
      kwargs = {'cond': cond}
    vals = collections.OrderedDict()
    for key in self._keylist:
      vals.update({key: pred_vals[key]})
    if succ_vals is None:
      return vals, dims, kwargs

    # If stepping, add successor values
    for key in self._keylist:
      mod_key = key+"'"
      succ_key = key if mod_key not in succ_vals else mod_key
      vals.update({key+"'": succ_vals[succ_key]})

    return vals, dims, kwargs

#-------------------------------------------------------------------------------
  def subfield(self, vertices):
    """ Returns a view of vertices, which must all be members, as a Field.

    :param vertices: str/variable/field or list/tuple of str/variable/field use.

    :return: an Field including only those vertices
    """

    # Convert to list
    if isinstance(vertices, tuple):
      vertices = list(vertices)
    if not isinstance(vertices, list):
      vertices = [vertices]

    # Collate variables and return as Field
    variables = []
    for vertex in vertices:
      if isinstance(vertex, Field):
        variables += [vertex.ret_vars(aslist=True)]
      elif isinstance(vertex, Variable):
        variables += [vertex]
      elif isinstance(vertex, str):
        variables += [self._vars[vertex]]
      else:
        raise TypeError("Unrecognised vertex specification type: {}".format(
            type(vertex)))
    return Field(*tuple(variables))

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    """ Returns a distribution p(args) """
    if not self._nvars:
      return None
    iid = False if 'iid' not in kwds else kwds.pop('iid')
    if type(iid) is bool and iid:
      iid = self._defiid
    values = self.parse_args(*args, **kwds)
    vals, dims = self.evaluate(values, _skip_parsing=True)
    name = ','.join(list(vals.keys()))
    return Distribution(name, vals, dims=dims)

#-------------------------------------------------------------------------------
  def __and__(self, other):
    return and_op(self, other)

#-------------------------------------------------------------------------------
  def __or__(self, other):
    return or_op(self, other)

#-------------------------------------------------------------------------------
  def __eq__(self, other):
    """ Equality for Fields is defined as comprising the same Variables """
    if type(self) is not Field or type(other) is not Field:
      return super().__eq__(other)
    return self._keyset == other.keyset

#-------------------------------------------------------------------------------
  def __ne__(self, other):
    """ Equality for Field is defined as comprising the same Variables """
    return not self.__eq__(other)

#-------------------------------------------------------------------------------
  def __getitem__(self, key):
    if type(key) is int:
      key = self._keylist[key]
    if isinstance(key, str):
      if key not in self._keylist:
        return None
    return self._vars[key]

#-------------------------------------------------------------------------------
  def __repr__(self):
    if not self._name:
      return NX_UNDIRECTED_GRAPH.__repr__(self)
    return NX_UNDIRECTED_GRAPH.__repr__(self) + ": '" + self._name + "'"

#-------------------------------------------------------------------------------
