# RJ utility module

#-------------------------------------------------------------------------------
import collections
import numpy as np
import scipy.stats
from probayes.pscales import iscomplex, rescale, prod_rule, prod_pscale

#-------------------------------------------------------------------------------
def rv_prod_rule(*args, rvs, pscale=None):
  """ Returns the probability product treating all rvs as independent.
  Values (=args[0]) are keyed by RV name and rvs are a list of RVs.
  """
  values = args[0]
  pscales = [rv.ret_pscale() for rv in rvs]
  pscale = pscale or prod_pscale(pscales)
  use_logs = iscomplex(pscale)
  probs = [rv.eval_prob(values[rv.ret_name()]) for rv in rvs]
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
def sample_cond_cov(*args, cond_cov=None, **kwds):
    kwds = dict(kwds)
    cond_pdf = False if 'cond_pdf' not in kwds else kwds.pop('cond_pdf')
    assert cond_cov, "coveig object mandatory"
    if len(args) == 1 and isinstance(args[0], dict):
      vals = args[0]
      idx = {key: i for i, key in enumerate(vals.keys())}
      args = [np.array(val) for val in vals.values()]
    elif not len(args) and len(kwds):
      vals = dict(kwds)
      idx = {key: i for i, key in enumerate(vals.keys())}
      args = list(kwds.values())
    return cond_cov.interp(*args, cond_pdf=cond_pdf)

#-------------------------------------------------------------------------------
