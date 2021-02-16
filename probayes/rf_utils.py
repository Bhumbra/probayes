# Utility module for RF objects

#-------------------------------------------------------------------------------
import collections
import numpy as np
from probayes.vtypes import isscalar
from probayes.pscales import iscomplex, rescale, prod_rule, prod_pscale

#-------------------------------------------------------------------------------
def rv_prod_rule(*args, rvs, pscale=None):
  """ Returns the probability product treating all rvs as independent.
  Values (=args[0]) are keyed by RV name and rvs are a list of RVs.
  """
  values = args[0]
  pscales = [rv.pscale for rv in rvs]
  pscale = pscale or prod_pscale(pscales)
  use_logs = iscomplex(pscale)
  probs = [rv.eval_prob(values[rv.name]) for rv in rvs]
  prob, pscale = prod_rule(*tuple(probs),
                           pscales=pscales,
                           pscale=pscale)

  # This section below is there just to play nicely with conditionals
  if len(args) > 1:
    if use_logs:
      prob = rescale(prob, pscale, 0.j)
    else:
      prob = rescale(prob, pscale, 1.)
    for arg in args[1:]:
      if use_logs:
        offs, _ = rv_prod_rule(arg, rvs=rvs, pscale=0.j)
        prob = prob + offs
      else:
        coef, _ = rv_prod_rule(arg, rvs=rvs, pscale=1.)
        prob = prob * coef
    if use_logs:
      prob = prob / float(len(args))
      prob = rescale(prob, 0.j, pscale)
    else:
      prob = prob ** (1. / float(len(args)))
      prob = rescale(prob, 1., pscale)
  return prob, pscale

#-------------------------------------------------------------------------------
def call_scipy_prob(func, pscale, *args, **kwds):
  index = 1 if iscomplex(pscale) else 0
  return func[index](*args, **kwds)

#-------------------------------------------------------------------------------
def sample_cond_cov(*args, cond_cov=None, unknown=None, **kwds):
    kwds = dict(kwds)
    cond_pdf = False if 'cond_pdf' not in kwds else kwds.pop('cond_pdf')
    assert cond_cov, "coveig object mandatory"
    if len(args) == 1 and isinstance(args[0], dict):
      vals = dict(args[0])
      if unknown is not None:
        vals[unknown] = None
      args = [np.array(val) if val is not None else {0} \
              for val in vals.values()]
    elif not len(args) and len(kwds):
      vals = dict(kwds)
      if unknown is not None:
        kwds[unknown] = {0}
      args = list(kwds.values())
    return cond_cov.interp(*tuple(args), cond_pdf=cond_pdf)

#-------------------------------------------------------------------------------
def slice_by_keyvals(spec, vals, prob, vals_dims=None, spec_dims=None):
  """ Slices prob by values of spec in vals.

  :param spec: dictionary of {key:val} to match with vals.
  :param vals: dictionary of {key:val} describing prob.
  :param prob: a multidimensional NumPy array to slice.
  :param vals_dims: dictionary of dimensional decription for vals.
  :param spec_dims: dictionary of dimensional decription for spec.

  If vals_dims and/or spec_dims are not entered, they are 'guessed' from vals
  and spec respectively, but correct guessing is not assured. Non-none dimensions
  for vals_dims and spec_dims must be mutually ordered monotically by key.
  """

  # Check for consistent keys
  keys = list(spec.keys())
  assert set(keys) == set(vals.keys()), "Keys for spec and vals unmatched"

  # Function to default dimensions if not given
  def dims_from_vals(vals_dict):
    if not isinstance(vals_dict, dict):
      raise TypeError("Dictionary type expected")
    dims = collections.OrderedDict()
    run_dim = 0
    for key, val in vals_dict.items():
      if isscalar(val):
        dims.update({key: None})
      else:
        assert val.size == np.product(vals.shape), \
            "Multiple non-singleton dimensions: {}".format(val.size)
        if val.size > 1:
          run_dim = np.argmax(val.shape)
        dims.update({key: run_dim})
        run_dim += 1

  # Default spec_dims and vals_dims if not given 
  if spec_dims is None:
    spec_dims = dims_from_vals(spec)
  else:
    assert set(spec.keys()) == set(spec_dims.keys()), \
        "Keys for spec and spec_dims unmatched"
  if vals_dims is None:
    vals_dims = dims_from_vals(vals)
  else:
    assert set(vals.keys()) == set(vals_dims.keys()), \
        "Keys for spec and spec_dims unmatched"
  
  # Determine maximum dimensionality of input
  vals_ndim = 0
  for dim in vals_dims.values():
    if dim:
      vals_ndim = max(vals_ndim, dim)

  # Determine maximum dimensionality of output from spec and spec_dims
  spec_ndim = 0
  for key, dim in spec_dims.items():
    if dim:
      spec_ndim = max(spec_ndim, dim)
    if not isscalar(spec[key]):
      spec_ndim = max(spec[key].ndim, dim)

  # Check for monotonic ordering of dimensions
  dims = [dim for dim in vals_dims.values() if dim is not None]
  if len(dims) > 1:
    assert np.min(np.diff(dims)) > 0, "Dimensionality not monotically ordered"
  dims = [dim for dim in spec_dims.values() if dim is not None]
  if len(dims) > 1:
    assert np.min(np.diff(dims)) > 0, "Dimensionality not monotically ordered"

  # Evaluate reshape and slices from matches between spec and vals
  reshape = [1] * spec_ndim
  slices = [slice(None) for _ in range(vals_ndim+1)]
  for i, key in enumerate(keys):
    if vals_dims[key] is None: # If source is scalar
      if not isscalar(spec[key]):
        assert np.all(spec[key] == vals[key]), \
          "Cannot slice by multiple values"
      elif spec[key] != vals[key]:
        return np.empty(np.zeros(len(dims), dtype=int), dtype=float)
      else:
        pass
    if spec_dims[key] is None: # If target is scalar
      dim = vals_dims[key]
      match = np.ravel(vals[key]) == spec[key]
      n_matches = match.sum()
      if n_matches == 0:
        slices[dim] = slice(0, 0)
      elif n_matches == 1:
        slices[dim] = np.nonzero(match)[0]
      else:
        raise ValueError("Non-unique matches found")
    else:
      assert np.all(np.ravel(vals[key]) == np.ravel(spec[key])), \
          "Ambiguous specification with values mismatch"
      dim = spec_dims[key]
      reshape[dim] = vals[key].size
  return prob[tuple(slices)].reshape(reshape)

#-------------------------------------------------------------------------------
