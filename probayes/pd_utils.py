
# A module to perform utility functions for PD() instances.

#-------------------------------------------------------------------------------
import collections
import functools
import numpy as np
import h5py
from probayes.vtypes import isscalar, issingleton, isunitsetint
from probayes.pscales import prod_pscale, prod_rule, iscomplex

#-------------------------------------------------------------------------------
def str2key(string):
  if isinstance(string, str):
    k = string.find('=')
    if k > 0:
      return string[:k]
    return string
  return [str2key(element) for element in string]

#-------------------------------------------------------------------------------
def str_margcond(name=None):
  """ Returns (marg,cond) tuple of OrderedDicts from a name """
  marg_str = name
  marg = collections.OrderedDict()
  cond = collections.OrderedDict()
  if not marg_str:
    return marg, cond
  lt_paren = marg_str.find('(')
  rt_paren = marg_str.find(')')
  if lt_paren >= 0 or rt_paren >= 0:
    assert lt_paren >= 0 and rt_paren > lt_paren, \
      "Unmatched parenthesis in name"
    marg_str = marg_str[lt_paren:1:rt_paren]
  cond_str = ''
  if '|' in marg_str:
    split_str = name.split('|')
    assert len(split_str) == 2, "Ambiguous name: {}".format(name)
    marg_str, cond_str = split_str
  marg_strs = []
  cond_strs = []
  if len(marg_str):
    marg_strs = marg_str.split(',') if ',' in marg_str else [marg_str] 
  if len(cond_str):
    cond_strs = cond_str.split(',') if ',' in cond_str else [cond_str]
  for string in marg_strs:
    marg.update({str2key(string): string})
  for string in cond_strs:
    cond.update({str2key(string): string})
  return marg, cond

#-------------------------------------------------------------------------------
def margcond_str(marg, cond):
  """ Returns a name from OrderedDict values in marg and cond """
  marg = list(marg.values()) if isinstance(marg, dict) else list(marg)
  cond = list(cond.values()) if isinstance(cond, dict) else list(cond)
  marg_str = ','.join(marg)
  cond_str = ','.join(cond)
  return '|'.join([marg_str, cond_str]) if cond_str else marg_str

#-------------------------------------------------------------------------------
def rekey_dict(ordereddict, keymap):
  assert isinstance(ordereddict, collections.OrderedDict), \
      "Marg or Cond dict objects must be Ordered Collections"
  assert isinstance(keymap, dict), \
      "Key map must be a dictionary"
  rekeyed = collections.OrderedDict()
  for key, val in ordereddict.items():
    if key not in keymap.keys():
      rekeyed.update({key: val})
    else:
      map_key = keymap[key]
      assert map_key not in rekeyed.keys(), \
          "Key remapping results in duplicate key: {}".format(map_key)
      string = val
      k = string.find('=')
      if k > 0:
        string = map_key + string[k:]
      else:
        string = map_key
      rekeyed.update({map_key: string})
  return rekeyed

#-------------------------------------------------------------------------------
def product(*args, **kwds):
  """ Multiplies two or more PDs subject to the following:
  1. They must not share the same marginal variables. 
  2. Conditional variables must be identical unless contained as marginal from
     another distribution.
  """
  from probayes.pd import PD

  # Check pscales, scalars, possible fasttrack
  if not len(args):
    return None
  kwds = dict(kwds)
  pscales = [arg.pscale for arg in args]
  pscale = kwds.get('pscale', None) or prod_pscale(pscales)
  aresingleton = [arg.issingleton for arg in args]
  maybe_fasttrack = all(aresingleton) and \
                    np.all(pscale == np.array(pscales)) and \
                    pscale in [0, 1.]


  # Collate vals, probs, marg_names, and cond_names as lists
  vals = [collections.OrderedDict(arg) for arg in args]
  probs = [arg.prob for arg in args]
  marg_names = [list(arg.marg.values()) for arg in args]
  cond_names = [list(arg.cond.values()) for arg in args]

  # Detect uniqueness in marginal keys and identical conditionals
  all_marg_keys = []
  for arg in args:
    all_marg_keys.extend(list(arg.marg.keys()))
  marg_sets = None
  if len(all_marg_keys) != len(set(all_marg_keys)):
    marg_keys, cond_keys, marg_sets, = None, None, None
    for arg in args:
      if marg_keys is None:
        marg_keys = list(arg.marg.keys())
      elif marg_keys != list(arg.marg.keys()):
        marg_keys = None
        break
      if cond_keys is None:
        cond_keys = list(arg.cond.keys())
      elif cond_keys != list(arg.cond.keys()):
        marg_keys = None
        break
      if marg_keys:  
        are_marg_sets = np.array([isunitsetint(arg[marg_key]) for
                                  marg_key in marg_keys])
        if marg_sets is None:
          if np.any(are_marg_sets):
            marg_sets = are_marg_sets
          else:
            marg_keys = None
            break
        elif not np.all(marg_sets == are_marg_sets):
          marg_keys = None
          break
    assert marg_keys is not None and marg_sets is not None, \
      "Non-unique marginal variables for currently not supported: {}".\
      format(all_marg_keys)
    maybe_fasttrack = True

  # Maybe fast-track identical conditionals
  if maybe_fasttrack:
    marg_same = True
    cond_same = True
    if marg_sets is None: # no need to recheck if not None (I think)
      marg_same = True
      for name in marg_names[1:]:
        if marg_names[0] != name:
          marg_same = False
          break
      cond_same = not any(cond_names)
      if not cond_same:
        cond_same = True
        for name in cond_names[1:]:
          if cond_names[0] != name:
            cond_same = False
            break
    if marg_same and cond_same:
      marg_names = marg_names[0]
      cond_names = cond_names[0]
      prod_marg_name = ','.join(marg_names)
      prod_cond_name = ','.join(cond_names)
      prod_name = '|'.join([prod_marg_name, prod_cond_name])
      prod_vals = collections.OrderedDict()
      for i, val in enumerate(vals):
        areunitsetints = np.array([isunitsetint(_val) 
                                   for _val in val.values()])
        if not np.any(areunitsetints):
          prod_vals.update(val)
        else:
          assert marg_sets is not None, "Variable mismatch"
          assert np.all(marg_sets == areunitsetints[:len(marg_sets)]), \
              "Variable mismatch"
          if not len(prod_vals):
            prod_vals.update(collections.OrderedDict(val))
          else:
            for j, key in enumerate(prod_vals.keys()):
              if areunitsetints[j]:
                prod_vals.update({key: {list(prod_vals[key])[0] + \
                                        list(val[key])[0]}})
      if marg_sets is not None:
        prob, pscale = prod_rule(*tuple(probs), pscales=pscales, pscale=pscale)
        return PD(prod_name, prod_vals, dims=args[0].dims, prob=prob, pscale=pscale)
      else:
        prod_prob = float(sum(probs)) if iscomplex(pscale) else float(np.prod(probs))
        return PD(prod_name, prod_vals, prob=prod_prob, pscale=pscale)

  # Check cond->marg accounts for all differences between conditionals
  prod_marg = [name for dist_marg_names in marg_names \
                          for name in dist_marg_names]
  prod_marg_name = ','.join(prod_marg)
  flat_cond_names = [name for dist_cond_names in cond_names \
                          for name in dist_cond_names]
  cond2marg = [cond_name for cond_name in flat_cond_names \
                         if cond_name in prod_marg]
  prod_cond = [cond_name for cond_name in flat_cond_names \
                         if cond_name not in cond2marg]
  cond2marg_set = set(cond2marg)

  # Check conditionals compatible
  prod_cond_set = set(prod_cond)
  cond2marg_dict = {name: None for name in cond2marg}
  for i, arg in enumerate(args):
    cond_set = set(cond_names[i]) - cond2marg_set
    if cond_set:
      assert prod_cond_set == cond_set, \
          "Incompatible product conditional {} for conditional set {}: ".format(
              prod_cond_set, cond_set)
    for name in cond2marg:
      if name in arg.keys():
        values = arg[name]
        if not isscalar(values):
          values = np.ravel(values)
        if cond2marg_dict[name] is None:
          cond2marg_dict[name] = values
        elif not np.allclose(cond2marg_dict[name], values):
          raise ValueError("Mismatch in values for condition {}".format(name))

  # Establish product name, values, and dimensions
  prod_keys = str2key(prod_marg + prod_cond)
  prod_nkeys = len(prod_keys)
  prod_aresingleton = np.zeros(prod_nkeys, dtype=bool)
  prod_areunitsetints = np.zeros(prod_nkeys, dtype=bool)
  prod_cond_name = ','.join(prod_cond)
  prod_name = prod_marg_name if not len(prod_cond_name) \
              else '|'.join([prod_marg_name, prod_cond_name])
  prod_vals = collections.OrderedDict()
  for i, key in enumerate(prod_keys):
    values = None
    for val in vals:
      if key in val.keys():
        values = val[key]
        prod_areunitsetints[i] = isunitsetint(val[key])
        if prod_areunitsetints[i]:
          values = {0}
        break
    assert values is not None, "Values for key {} not found".format(key)
    prod_aresingleton[i] = issingleton(values)
    prod_vals.update({key: values})
  if np.any(prod_areunitsetints):
    for i, key in enumerate(prod_keys):
      if prod_areunitsetints[i]:
        for val in vals:
          if key in val:
            assert isunitsetint(val[key]), "Mismatch in variables {} vs {}".\
                format(prod_vals, val)
            prod_vals.update({key: {list(prod_vals[key])[0] + list(val[key])[0]}})
  prod_newdims = np.array(np.logical_not(prod_aresingleton))
  dims_shared = False
  for arg in args:
    argdims = [dim for dim in arg.dims.values() if dim is not None]
    if len(argdims) != len(set(argdims)):
      dims_shared = True

  # Shared dimensions limit product dimensionality
  if dims_shared:
    seen_keys = set()
    for i, key in enumerate(prod_keys):
      if prod_newdims[i] and key not in seen_keys:
        for arg in args:
          if key in arg.dims:
            dim = arg.dims[key]
            seen_keys.add(key)
            for argkey, argdim in arg.dims.items():
              seen_keys.add(argkey)
              if argkey != key and argdim is not None:
                if dim == argdim:
                  index = prod_keys.index(argkey)
                  prod_newdims[index] = False

  prod_cdims = np.cumsum(prod_newdims)
  prod_ndims = prod_cdims[-1]

  # Fast-track scalar products
  if maybe_fasttrack and prod_ndims == 0:
     prob = float(sum(probs)) if iscomplex(pscale) else float(np.prod(probs))
     return PD(prod_name, prod_vals, prob=prob, pscale=pscale)

  # Reshape values - they require no axes swapping
  ones_ndims = np.ones(prod_ndims, dtype=int)
  prod_shape = np.ones(prod_ndims, dtype=int)
  scalarset = set()
  prod_dims = collections.OrderedDict()
  for i, key in enumerate(prod_keys):
    if prod_aresingleton[i]:
      scalarset.add(key)
    else:
      values = prod_vals[key]
      re_shape = np.copy(ones_ndims)
      dim = prod_cdims[i]-1
      prod_dims.update({key: dim})
      re_shape[dim] = values.size
      prod_shape[dim] = values.size
      prod_vals.update({key: values.reshape(re_shape)})
  
  # Match probability axes and shapes with axes swapping then reshaping
  for i in range(len(args)):
    prob = probs[i]
    if not isscalar(prob):
      dims = collections.OrderedDict()
      for key, val in args[i].dims.items():
        if val is not None:
          dims.update({val: prod_dims[key]})
      old_dims = []
      new_dims = []
      for key, val in dims.items():
        if key not in old_dims:
          old_dims.append(key)
          new_dims.append(val)
      if len(old_dims) > 1 and not old_dims == new_dims:
        max_dims_inc = max(new_dims) + 1
        while prob.ndim < max_dims_inc:
          prob = np.expand_dims(prob, -1)
        prob = np.moveaxis(prob, old_dims, new_dims)
      re_shape = np.copy(ones_ndims)
      for dim in new_dims:
        re_shape[dim] = prod_shape[dim]
      probs[i] = prob.reshape(re_shape)

  # Multiply the probabilities and output the result as a distribution instance
  prob, pscale = prod_rule(*tuple(probs), pscales=pscales, pscale=pscale)

  return PD(prod_name, prod_vals, dims=prod_dims, prob=prob, pscale=pscale)


#-------------------------------------------------------------------------------
def summate(*args):
  """ Quick and dirty concatenation """
  from probayes.pd import PD
  if not len(args):
    return None
  pscales = [arg.pscale for arg in args]
  vals = [dict(arg) for arg in args]
  probs = [arg.prob for arg in args]

  # Check pscales are the same
  pscale = pscales[0]
  for _pscale in pscales[1:]:
    assert pscale == _pscale, \
        "Cannot summate distributions with different pscales"

  # Check marginal and conditional keys
  marg_keys = list(args[0].marg.keys())
  cond_keys = list(args[0].cond.keys())
  for arg in args[1:]:
    assert marg_keys == list(arg.marg.keys()), \
      "Marginal variable names not identical across distributions: {}"
    assert cond_keys == list(arg.cond.keys()), \
      "Conditional variable names not identical across distributions: {}"
  sum_keys = marg_keys + cond_keys
  sum_name = ','.join(marg_keys)
  if cond_keys:
    sum_name += '|' + ','.join(cond_keys)

  # If all singleton, concatenate in dimension 0
  if all([arg.issingleton for arg in args]):
    unitsets = {key: isunitsetint(args[0][key]) for key in sum_keys}
    sum_dims = {key: None if unitsets[key] else 0 for key in sum_keys}
    sum_vals = {key: 0 if unitsets[key] else [] for key in sum_keys}
    sum_prob = []
    for arg in args:
      for key, val in arg.items():
        if unitsets[key]:
          assert isunitsetint(val), \
              "Cannot mix unspecified set and specified values"
          sum_vals[key] += list(val)[0]
        else:
          assert not isunitsetint(val), \
              "Cannot mix unspecified set and specified values"
          sum_vals[key].append(val)
      sum_prob.append(arg.prob)
    for key in sum_keys:
      if unitsets[key]:
        sum_vals[key] = {sum_vals[key]}
      else:
        sum_vals[key] = np.ravel(sum_vals[key])
    sum_prob = np.ravel(sum_prob)
    return PD(sum_name, sum_vals, dims=sum_dims, prob=sum_prob, pscale=pscale)

  # 2. all identical but in one dimension: concatenate in that dimension
  # TODO: fix the remaining code of this function below
  sum_vals = collections.OrderedDict(args[0])
  sum_dims = [None] * (len(args) - 1)
  for i, arg in enumerate(args):
    if i == 0:
      continue
    for key in marg_keys:
      if sum_dims[i-1] is not None:
        continue
      elif not arg.singleton(key):
        key_vals = arg[key]
        if key_vals.size == sum_vals[key].size:
          if np.allclose(key_vals, sum_vals[key]):
            continue
        sum_dims[i-1] = arg.dims[key]
  assert len(set(sum_dims)) > 1, "Cannot find unique concatenation axis"
  sum_dim = sum_dims[0]
  sum_dims = args[0].dims
  key = marg_keys[sum_dim]
  sum_prob = np.copy(probs[0])
  for i, val in enumerate(vals):
    if i == 0:
      continue
    sum_vals[key] = np.concatenate([sum_vals[key], val[key]], axis=sum_dim)
    sum_prob = np.concatenate([sum_prob, probs[i]], axis=sum_dim)
  return PD(sum_name, sum_vals, dims=sum_dims, prob=sum_prob, pscale=pscale)

#-------------------------------------------------------------------------------
def ismonotonic(vals):
  if issingleton(vals):
    return True
  vals = np.ravel(vals)
  is_ge = vals[1:] >= vals[:-1]
  if len(np.unique(is_ge)) == 1:
    return True
  return False

#-------------------------------------------------------------------------------
def iterdict(dicts):
  """ Converts a tuple/list of dicts to a dict of lists """
  return collections.OrderedDict(
      {key: [_dict[key] for _dict in dicts if key in _dict]
             for key in functools.reduce(set.union, [set(_dict.keys()) 
                                                     for _dict in dicts])}
                                )

#-------------------------------------------------------------------------------
def serialise(*args):
  from probayes import Distribution
  from probayes import PD
  serialised = {}
  for arg in args:
    if not isinstance(arg, (PD, Distribution)):
      raise TypeError(f"Unrecogised type to serialise: {type(arg)}")
    serialised.update(arg.serialise())
  return serialised

#-------------------------------------------------------------------------------
def deserialise(serialised):
  from probayes import Distribution
  from probayes import PD
  assert isinstance(serialised, dict), \
    f"Dict-type serialised input expected, not {type(serialised)}"
  dists = []
  for dist_name, _dist_dict in serialised.items():
    dist_dict = dict(_dist_dict)
    dims = {} if 'attrs' not in dist_dict else dist_dict.pop('attrs')
    prob = None if 'prob' not in dist_dict else dist_dict.pop('prob')
    pscale = None if 'pscale' not in dist_dict else dist_dict.pop('pscale')
    if prob is None:
      dists.append(Distribution(dist_name, dist_dict, dims=dims))
    else:
      pscale = pscale if pscale is None else np.atleast_1d(pscale).tolist()[0]
      dists.append(PD(dist_name, dist_dict, prob=prob, pscale=pscale, dims=dims))
  return tuple(dists)

#-------------------------------------------------------------------------------
def write_serialised(path, serialised, aux_dict={}):
  assert isinstance(serialised, dict), \
    f"Dict-type serialised input expected, not {type(serialised)}"
  for key, val in aux_dict.items():
    assert isinstance(key, str) and ' ' in key, \
      f"Aux dict must keyed by space-containing string, found {key}"
    assert isinstance(val, dict), \
      f"Aux dict must be a nested dictionary of dicts - found {type(val)}"
    for subkey, subval in val.items():
      assert isinstance(subkey, str), \
        f"Aux dict subkeys must be str, found {subkey}"
      assert isinstance(subval, np.ndarray), \
        f"Aux dict subvals must be NumPy arrays, found {type(subval)}"
  with h5py.File(path, 'w', libver='latest') as hdf_write:
    for dist_name, dist_dict in serialised.items():
      group_write = hdf_write.create_group(dist_name)
      for key, val in dist_dict.items():
        if key == 'attrs':
          attrs = dist_dict['attrs']
          for k, v in attrs.items():
            if v is None:
              attrs[k] = 'None'
          group_write[key] = np.array(len(attrs))
          group_write[key].attrs.update(attrs)
          if isinstance(attrs, collections.OrderedDict):
            group_write[key].attrs.update({'order': list(attrs.keys())})
        else:
          if isinstance(val, np.ndarray):
            group_write[key] = val
          elif isinstance(val, set):
            val_set = np.zeros(sorted(val) + [0], dtype='S')
            group_write[key] = val_set
          else:
            group_write[key] = np.array(val)
    for aux_name, aux_data in aux_dict.items():
      group_write = hdf_write.create_group(aux_name)
      for key, val in aux_data.items():
        group_write[key] = val

#-------------------------------------------------------------------------------
def read_serialised(path):
  serialised = {}
  aux_data = {}
  with h5py.File(path, 'r', libver='latest') as hdf_read:
    dist_names = hdf_read.keys()
    for dist_name in dist_names:
      group_read = hdf_read[dist_name]
      order = None
      attrs = {}
      if ' ' not in dist_name:
        serialised[dist_name] = {}
        for key, val in group_read.items():
          if key == 'attrs':
            attrs = dict(val.attrs)
            if 'order' in attrs:
              order = attrs.pop('order')
              ordered_dict = collections.OrderedDict()
              for ordered_key in order:
                ordered_dict.update({ordered_key: attrs[ordered_key]})
              attrs = ordered_dict
          else:
            for key, val in group_read.items():
              serialised[dist_name][key] = np.array(val)
              if serialised[dist_name][key].dtype in (np.dtype('S'), np.dtype('S21'), np.dtype('|S1')):
                serialised[dist_name][key] = set(val.shape[:-1])
        if attrs: 
          for k, v in attrs.items():
            if v == 'None':
              attrs[k] = None
          if order is not None:
            ordered_dict = collections.OrderedDict()
            for ordered_key in order:
              ordered_dict.update({ordered_key: serialised[dist_name][ordered_key]})
            for key, val in serialised[dist_name].items():
              if key not in ordered_dict:
                ordered_dict.update({key: val})
            serialised[dist_name] = ordered_dict
          serialised[dist_name].update({'attrs': attrs})
      else:
        aux_data[dist_name] = {}
        for key, val in group_read.items():
          aux_data[dist_name][key] = np.array(val)
  return serialised, aux_data

#-------------------------------------------------------------------------------
def write_dist(path, *args):
  return write_serialised(path, serialise(*args))

#-------------------------------------------------------------------------------
def read_dist(path):
  return deserialise(read_serialised(path)[0])

#-------------------------------------------------------------------------------
