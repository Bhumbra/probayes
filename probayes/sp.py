"""
A stocastic process is an indexable sequence of realisations of a stochastic 
condition. It is therefore implemented here using a sample generator that 
iteratively samples a stochastic condition.
"""
#-------------------------------------------------------------------------------
import numpy as np
import collections
import inspect
import warnings
from probayes.rf import RF
from probayes.func import Func
from probayes.dist import Dist
from probayes.dist_utils import summate
from probayes.sp_utils import sample_generator, MCMC_SAMPLERS

#-------------------------------------------------------------------------------
class SP (RF):

  # Public
  stuv = None     # scores + thresholds + update + veridct
  opqrstuv = None # opqr + stuv

  # Protected
  _scores = None # Scores function used for the basis of acceptance
  _thresh = None # Threshold function to compare with scores
  _update = None # Update function (output True, None, or False)

  # Private
  __samplers = None  # List of samplers
  __counter = None   # Step counter
  __last = None      # Last argument ordereddict

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
    if self._scores in MCMC_SAMPLERS:
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._scores)
      self.set_scores(MCMC_SAMPLERS[self._scores][0], pscale=self._pscale)
      self.set_thresh(scores)
      return
    self._scores = Func(self._scores, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_thresh(self, thresh=None, *args, **kwds):
    self._thresh = thresh
    if self._thresh is None:
      return
    if self._thresh in MCMC_SAMPLERS:
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._thresh)
      self.set_thresh(MCMC_SAMPLERS[self._thresh][1])
      self.set_update(thresh)
      return
    self._thresh = Func(self._thresh, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_update(self, update=None, *args, **kwds):
    self._update = update
    if self._update is None:
      return
    if self._update in MCMC_SAMPLERS:
      assert not args and not kwds, \
          "Neither args nor kwds permitted with spec '{}'".format(self._update)
      self.set_update(MCMC_SAMPLERS[self._update][2])
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
  def reset(self, sampler_id=None, reset_last=True): # Leave option to preseve
    if self.__samplers is None or sampler_id is None:
      self.__samplers = []
    if self.__counter is None:
      self.__counter = collections.Counter()
    if self.__last is None:
      self.__last = collections.OrderedDict()
    if sampler_id is None:
      return
    sampler = self.ret_sampler(sampler_id)
    self.__counter[sampler] = 0
    if sampler not in self.__last:
      self.__last[sampler] = None
    elif reset_last:
      self.__last[sampler] = None
    return self.__samplers, self.__counter, self.__last

#-------------------------------------------------------------------------------
  def __call__(self, *args, **kwds):
    if not len(args) or len(kwds) or \
        not isinstance(args[0], (list, tuple, collections.deque)):
      return super().__call__(*args, **kwds)
    samples = args[0]
    if not len(samples):
      return super().__call__(*args, **kwds)

    # Summating distributions is straightforward
    if isinstance(samples[0], Dist):
      samples = list(samples)
      for sample in samples[1:]:
        assert isinstance(sample, Dist),\
            "If using distributions, all samples must be distributions, not {}".\
            format(type(sample))
      if isinstance(samples, (list, collections.deque)):
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
      if isinstance(sample, self.opqr):
        _maybe_append(sample.o, 'o')
        _maybe_append(sample.p, 'p')
        _maybe_append(sample.q, 'q')
        _maybe_append(sample.r, 'r')
      
      else:
        assert isinstance(sample, self.opqrstuv), \
            "Sample must be outputted from sampler: {}".format(self._id)
        if sample.u == False:
          continue
        if self._update is not None:
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
  def ret_sampler(self, sampler_id=None):
    if sampler_id is None:
      return self.__samplers
    if type(sampler_id) is int:
      return self.__samplers[sampler_id]
    return sampler_id

#-------------------------------------------------------------------------------
  def ret_counter(self, sampler_id=None):
    if sampler_id is None:
      return self.__counter
    return self.__counter[self.ret_sampler(sampler_id)]

#-------------------------------------------------------------------------------
  def ret_last(self, sampler_id=None):
    if sampler_id is None:
      return self.__last
    return self.__last[self.ret_sampler(sampler_id)]

#-------------------------------------------------------------------------------
  def next(self, sampler_id, *args, **kwds):

    # Reset counters
    sampler = self.ret_sampler(sampler_id)
    self.__counter[sampler] += 1
    last = self.__last[sampler]

    # Treat sampling without proposals as a distribution call
    if last is None or \
        (self._tran is None and not self._unit_tran and self._prop is None):
      opqr = self.sample(*args, **kwds)
      if self._tran is None and not self._unit_tran and self._prop is None:
        return opqr

    # Otherwise refeed last proposals into sample function
    else:
      if self._tran is None and not self._unit_tran and self._delta is None:
        last = {0}
      if len(args) < 2:
        opqr = self.sample(last, **kwds)
      else:
        args = tuple([last] + list(args[1:]))
        opqr = self.sample(*args, **kwds)

    # Set to last if accept is not False
    stuv = self.stuv(self.eval_func(self._scores, opqr),
                     self.eval_func(self._thresh),
                     None,
                     None)
    update = self.eval_func(self._update, stuv)
    verdit = opqr.o
    if self._update is None or self.__last[sampler] is None or update:
      self.__last[sampler] = opqr
      verdit = opqr.p
    return self.opqrstuv(opqr.o, opqr.p, opqr.q, opqr.r, 
                         stuv.s, stuv.t, update, verdit)

#-------------------------------------------------------------------------------
  def sampler(self, *args, **kwds):
    if self.__samplers is None:
      self.reset()
    if not args:
      args = {0},
    elif len(args) == 1 and type(args[0]) is int and 'stop' not in kwds:
      kwds.update({'stop': args[0]})
      args = {0},
    stop = None if 'stop' not in kwds else kwds.pop('stop')
    sampler = sample_generator(self, 
                               len(self.__samplers), 
                               *args, 
                               stop=stop, 
                               **kwds)
    self.__samplers.append(sampler)
    self.__counter[sampler] = 0
    self.__last[sampler] = None
    return sampler

#-------------------------------------------------------------------------------
  def walk(self, sampler, stop=None):
    gen_vars = inspect.getgeneratorlocals(sampler)
    assert 'stop' in gen_vars, \
        'Sampler must be a sampler_generator instance returned by SP.sampler()'
    if stop is None and gen_vars['stop'] is None:
      warnings.warn(
        "No stop specification set - this walk may proceed indefinitely")
    if stop is None: 
      return collections.deque([sample for sample in sampler])
    steps = collections.deque()
    for sample in sampler:
      if len(steps) >= stop:
        break
      steps.append(sample)
    return steps

#-------------------------------------------------------------------------------

