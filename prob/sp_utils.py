""" A utility module for stochasptic process classes """

import numpy as np
from prob.vtypes import isscalar
from prob.pscales import iscomplex, log_prob, div_prob

#-------------------------------------------------------------------------------
def sample_generator(sp, *args, stop=None, **kwds):
  if stop is None:
    while True:
      yield sp.next(*args, **kwds)
  else:
    while sp.ret_counter() is None or sp.ret_counter() < stop:
      yield sp.next(*args, **kwds)
    else:
      sp.reset(sp.ret_last())

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
  if revp is not None:
    raise NotImplementedError("Non-symmetrical transitions not yet supported")
  message = "No valid scalar probability distribution found"
  assert succ is not None, message
  assert isscalar(succ.prob), message 
  if pred is None:
    return None
  assert isscalar(pred.prob), "Preceding probability distribution non-scalar"
  if prop is not None:
    assert isscalar(prop.prob), "Proposal probability distirbution non-scalar"
    prop = rescale(prop.prob, pscale, 1.)
    if prop <= 0.:
      return None
    return min(1., div_prob(succ.prob, pred.prob, pscale, pscale, pscale=1.))

#-------------------------------------------------------------------------------
def hastings_thresh(*args, **kwds):
  return metropolis_thresh(*args, **kwds)

#-------------------------------------------------------------------------------
def hastings_update(stu):
  return metropolis_update(stu)

#-------------------------------------------------------------------------------
