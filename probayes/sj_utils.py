# Module to support conditional sampling of multivariate distributions in SJ

import collections
import numpy as np
import scipy.stats
from probayes.pscales import iscomplex, rescale

#-------------------------------------------------------------------------------
def call_scipy_prob(func, pscale, *arg, **kwds):
  index = 1 if iscomplex(pscale) else 0
  return func[index](*args, **kwds)

#-------------------------------------------------------------------------------
def sample_cond_cov(*args, cond_cov=None, **kwds):
    assert cond_cov, "coveig object mandatory"
    if len(args) == 1 and isinstance(args[0], dict):
      vals = args[0]
      idx = {key: i for i, key in enumerate(vals.keys())}
      args = [np.array(val) for val in vals.values()]
    elif not len(args) and len(kwds):
      vals = dict(kwds)
      idx = {key: i for i, key in enumerate(vals.keys())}
      args = list(kwds.values())
    return cond_cov.interp(*args)

#-------------------------------------------------------------------------------
