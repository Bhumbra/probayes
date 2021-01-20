"""
Likelihood function utility module.
"""

#-------------------------------------------------------------------------------
import numpy as np
import collections
from probayes.rf_utils import slice_by_keyvals

#-------------------------------------------------------------------------------
def int_to_bin(num, min_dim=0):
  """Convert positive integer(s) to binary array(s).

  :param num: integer or 1D array of integers
  :param pad: optional minimum dimension (default 0):
  """

  if isinstance(num, (list, tuple)):
    num = np.array(num, dtype=int)
  if isinstance(num, np.ndarray):
    assert num.ndim == 1, "Input num must be integer or one dimensional array"
    min_dim = np.maximum(min_dim, len(np.binary_repr(np.max(num))))
    return np.vstack([int_to_bin(element, min_dim) for element in num])
  bin_str = np.binary_repr(num)
  if min_dim:
    bin_str = bin_str.zfill(min_dim)
  return np.array(np.array(list(bin_str)).astype(np.int8), bool)

#-------------------------------------------------------------------------------
def bin_to_int(arr, _multiple=None):
  """ Convert binary array(s) to positive integer(s)."""
  if isinstance(arr, (list, tuple)):
    arr = np.array(arr, dtype=int)
  assert isinstance(arr, np.ndarray) and arr.ndim and arr.ndim < 3, \
    "Input must be array type of not more than two dimensions"
  if arr.ndim == 2:
    if _multiple is None:
      _multiple = 1 << np.arange(arr.shape[1])[::-1]
    return np.hstack([bin_to_int(row, _multiple) for row in arr])
  if _multiple is None:
    _multiple = 1 << np.arange(arr.size)[::-1]
  return arr.dot(_multiple)

#-------------------------------------------------------------------------------
def bool_perm_freq(bool_2d, col_labels=None, base_freq=0):
  """ Returns a multidimensional array of counts of boolean permutations 
  contained in rows of bool_2d.

  :param bool_2d: a rXc 2D NumPy bool array with r examples of boolean
                  permutations of length c.
  :param col_labels: an optional list/tuple of keys for the labels.
  :param base_freq: baseline frequency to add to counts
                    (set to one for unit Laplacian smoothing).

  :return: counts or a tuple of (function, relative counts) if labels are given:

  counts: c-dimensional integer array of counts with each dimension ordered as
          [False, True] for each of the columns in bool_2d.
  function: returns relative frequency given keywords corresponding to labels

  """

  assert isinstance(bool_2d, np.ndarray) and bool_2d.ndim == 2 and \
      bool_2d.dtype == bool, "First input must be a 2D NumPy boolean array"
  
  rows, cols = bool_2d.shape
  counts = np.zeros([2] * cols, dtype=int)
  sequences = bool_2d.astype(int).tolist()
  for sequence in sequences:
    counts[tuple(sequence)] += 1
  if col_labels is None:
    return counts
  assert len(col_labels) == cols, \
      "Labels size {} incommensurate with input column number {}".format(
          len(col_labels), cols)
  dims = collections.OrderedDict()
  vals = collections.OrderedDict()
  ones = np.ones(cols, dtype=int)
  for dim, lbl in enumerate(col_labels):
    reshape = np.copy(ones)
    reshape[dim] = 2
    dims.update({lbl: dim})
    vals.update({lbl: np.array([False, True]).reshape(reshape)})
  if base_freq:
    base_freq += counts
  rel_freq = counts / rows

  def _func_bool_perm_freq(spec=None, **kwds):
    assert 'dims' in kwds, \
        "Output dimensionality not given - " + \
        "use SD.set_prob(function, passdims=True)"
    kwds = dict(kwds)
    spec_dims = kwds.pop('dims')
    if spec is None:
      spec = kwds
    else:
      assert not kwds, \
          "Unknown keywords: {}".format(kwds)
    return slice_by_keyvals(spec, vals, rel_freq, dims, spec_dims)
  
  return _func_bool_perm_freq, rel_freq

#-------------------------------------------------------------------------------
