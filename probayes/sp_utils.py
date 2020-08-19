""" A utility module for stochasptic process classes """

import numpy as np
from probayes.vtypes import isscalar
from probayes.pscales import iscomplex, rescale, log_prob, div_prob

#-------------------------------------------------------------------------------
def sample_generator(sp, sampler_id, *args, stop=None, **kwds):
  if stop is None:
    while True:
      yield sp.next(sampler_id, *args, **kwds)
  else:
    while sp.ret_counter(sampler_id) < stop:
      yield sp.next(sampler_id, *args, **kwds)
    else:
      sp.reset(sampler_id)

#-------------------------------------------------------------------------------
def metropolis_scores(opqr, pscale=None):
  pred, succ = opqr.o, opqr.p
  message = "No valid scalar probability distribution found"
  assert succ is not None, message
  assert isscalar(succ.prob), message 
  if pred is None:
    return None
  assert isscalar(pred.prob), "Preceding probability distribution non-scalar"
  return min(1., div_prob(succ.prob, pred.prob, pscale, pscale, pscale=1.))

#-------------------------------------------------------------------------------
def metropolis_thresh(*args, **kwds):
  return np.random.uniform(*args, **kwds)

#-------------------------------------------------------------------------------
def metropolis_update(stu):
  if stu.s is None or stu.s >= stu.t:
    return True
  return None

#-------------------------------------------------------------------------------
def hastings_scores(opqr, pscale=None):
  pred, succ, prop, revp = opqr.o, opqr.p, opqr.q, opqr.r
  message = "No valid scalar probability distribution found"
  assert succ is not None, message
  assert isscalar(succ.prob), message 
  if pred is None:
    return None
  assert isscalar(pred.prob), "Preceding probability non-scalar"
  if prop is None:
    return None
  else:
    assert isscalar(prop.prob), "Proposal probability non-scalar"
    prop = rescale(prop.prob, pscale, 1.)
    if prop <= 0.:
      return None
    if revp is None:
      return min(1., div_prob(succ.prob, pred.prob, pscale, pscale, pscale=1.))
    else:
      assert isscalar(revp.prob), "Reverse proposal probability non-scalar"
      revp = rescale(revp.prob, pscale, 1.)
      if revp <= 0.:
        return 1.
      return min(1., div_prob(succ.prob * prop, 
                              pred.prob * revp, 
                              pscale, pscale, pscale=1.))

#-------------------------------------------------------------------------------
def hastings_thresh(*args, **kwds):
  return metropolis_thresh(*args, **kwds)

#-------------------------------------------------------------------------------
def hastings_update(stu):
  return metropolis_update(stu)

#-------------------------------------------------------------------------------
def gibbs_scores(*args, **kwds):
  return np.nan

#-------------------------------------------------------------------------------
def gibbs_thresh(*args, **kwds):
  return np.nan

#-------------------------------------------------------------------------------
def gibbs_update(*args, **kwds):
  return True

#-------------------------------------------------------------------------------
MCMC_SAMPLERS = {
    'metropolis': (metropolis_scores, metropolis_thresh, metropolis_update),
    'hastings': (hastings_scores, hastings_thresh, hastings_update),
    'gibbs': (gibbs_scores, gibbs_thresh, gibbs_update),
                }

#-------------------------------------------------------------------------------
