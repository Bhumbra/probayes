"""
A stocastic process is an indexable sequence of realisations of a stochastic 
condition. It is therefore implemented here using a sample generator that 
iteratively samples a stochastic condition.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
from prob.sc import SC
from prob.func import Func
from prob.dist import Dist
from prob.dist_ops import summate
from prob.sp_utils import sample_generator, \
                          metropolis_scores, metropolis_thresh, metropolis_update, \
                          hastings_scores, hastings_thresh, hastings_update \

#-------------------------------------------------------------------------------
class SP (SC):

  # Public
  stuv = None     # scores + thresholds + update + veridct
  opqrstuv = None # opqr + stuv

  # Protected
  _scores = None # Scores function used for the basis of acceptance
  _thresh = None # Threshold function to compare with scores
  _update = None # Update function (output True, None, or False)

  # Private
  __last = None    # Last accepted Dist or OPQR object
  __counter = None # Step counter

#-------------------------------------------------------------------------------
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    self.reset()

#-------------------------------------------------------------------------------
  def _refresh(self):
    super()._refresh()
    if self._marg is None and self._cond is None:
      return
    self.stuv = collections.namedtuple(self._id, ['s', 't', 'u', 'v'])
    self.opqrstuv = collections.namedtuple(self._id, 
                        ['o', 'p', 'q', 'r', 's', 't', 'u', 'v'])

#-------------------------------------------------------------------------------
  def set_scores(self, scores=None, *args, **kwds):
    self._scores = scores
    if self._scores is None:
      return
    if self._scores == 'metropolis':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._scores)
      self.set_scores(metropolis_scores, pscale=self._pscale)
      self.set_thresh('metropolis')
      return
    elif self._scores == 'hastings':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._scores)
      self.set_scores(hastings_scores, pscale=self._pscale)
      self.set_thresh('hastings')
      return
    self._scores = Func(self._scores, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_thresh(self, thresh=None, *args, **kwds):
    self._thresh = thresh
    if self._thresh is None:
      return
    if self._thresh == 'metropolis':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._thresh)
      self.set_thresh(metropolis_thresh)
      self.set_update('metropolis')
      return
    elif self._thresh == 'hastings':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._thresh)
      self.set_thresh(hastings_thresh)
      self.set_update('hastings')
      return
    self._thresh = Func(self._thresh, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_update(self, update=None, *args, **kwds):
    self._update = update
    if self._update is None:
      return
    if self._update == 'metropolis':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._update)
      self.set_update(metropolis_update)
      return
    elif self._update == 'hastings':
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._update)
      self.set_update(hastings_update)
      return
    self._update = Func(self._update, *args, **kwds)

#-------------------------------------------------------------------------------
  def eval_func(self, func=None, *args, **kwds):
    if func is None:
      return func
    assert isinstance(func, Func), \
        "Evaluation of func no possible for type {}".format(type(func))
    if not func.ret_callable():
      return func()
    return func(*args, **kwds)

#-------------------------------------------------------------------------------
  def reset(self, last=None, counter=None): # this allows optional preservation
    self.__last = last
    self.__counter = counter

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    if not len(args) or len(kwds) or not isinstance(args[0], (list, tuple)):
      return super().__call__(*args, **kwds)
    samples = args[0]
    if not len(samples):
      return super().__call__(*args, **kwds)

    # Summating distributions is straightforward
    if isinstance(samples[0], Dist):
      for sample in samples[1:]:
        assert isinstance(samples, Dist),\
            "If using distributions, all samples must be distributions"
      if isinstance(samples, list):
        samples = tuple(samples)
      return summate(*samples)

    opqrstuv = collections.OrderedDict({key: None for key in \
        ['o', 'p', 'q', 'r', 's', 't', 'u', 'v']})

    # We exclude update=False from the summation
    opqrstuv['u'] = []

    def _maybe_append(element, key):
      if element is not None:
        if opqrstuv[key] is None:
          opqrstuv[key] = []
        opqrstuv[key].append(element)

    for sample in samples:
      assert isinstance(sample, self.opqrstuv), \
          "Sample must be outputted from sampler: {}".format(self._id)
      if sample.u == False:
        continue
      opqrstuv['u'].append(sample.u)
      _maybe_append(sample.o, 'o')
      _maybe_append(sample.p, 'p')
      _maybe_append(sample.q, 'q')
      _maybe_append(sample.r, 'r')
      _maybe_append(sample.s, 's')
      _maybe_append(sample.t, 't')
      _maybe_append(sample.v, 'v')
          
    for key in ['o', 'p', 'q', 'r', 'v']:
      if opqrstuv[key] is not None:
        opqrstuv[key] = summate(*tuple(opqrstuv[key]))
    return self.opqrstuv(**opqrstuv)

#-------------------------------------------------------------------------------
  def ret_last(self):
    return self.__last

#-------------------------------------------------------------------------------
  def ret_counter(self):
    return self.__counter

#-------------------------------------------------------------------------------
  def next(self, *args, **kwds):

    # Reset counters
    if self.__counter is None:
      self.reset(self.ret_last(), 0)
    self.__counter += 1

    # Treat sampling without proposals as a distribution call
    if self.__last is None or (self._tran is None and self._prop is None):
      opqr = self.sample(*args, **kwds)
      if self._tran is None and self._prop is None:
        return opqr

    # Otherwise refeed last proposals into sample function
    else:
      last = self.__last if self._tran is not None or self._delta is not None \
             else {0}
      if len(args) < 2:
        opqr = self.sample(last, **kwds)
      else:
        args = tuple(last + list(args[1:]))
        opqr = self.sample(*args, **kwds)

    # Set to last if accept is not False
    stuv = self.stuv(self.eval_func(self._scores, opqr),
                     self.eval_func(self._thresh),
                     None,
                     None)
    update = self.eval_func(self._update, stuv)
    verdit = opqr.o
    if self.__last is None or update:
      self.__last = opqr
      verdit = opqr.p
    return self.opqrstuv(opqr.o, opqr.p, opqr.q, opqr.r, 
                         stuv.s, stuv.t, update, verdit)

#-------------------------------------------------------------------------------
  def sampler(self, *args, **kwds):
    self.reset()
    if not args:
      args = {0},
    elif len(args) == 1 and type(args[0]) is int and 'stop' not in kwds:
      kwds.update({'stop': args[0]})
      args = {0},
    stop = None if 'stop' not in kwds else kwds.pop('stop')
    return sample_generator(self, *args, stop=stop, **kwds)

#-------------------------------------------------------------------------------
