"""
A stocastic process is indexable sequence of realisations of a stochastic condition
"""
#-------------------------------------------------------------------------------
import numpy as np
from prob.sc import SC
import collections

#-------------------------------------------------------------------------------
def sample_generator(sp, *args, stop=None, **kwds):
  if stop is None:
    while True:
      yield sp.next(*args, **kwds)
  else:
    while sp.ret_counter() is None or sp.ret_counter() < stop:
      yield sp.next(*args, **kwds)
    else:
      sp.reset_last()

#-------------------------------------------------------------------------------
class SP (SC):

  # Public
  opqrstu = None # opqr + scores + thresh + update

  # Protected
  _scores = None # Scores function used for the basis of acceptance
  _thresh = None # Threshold function to compare with scores
  _update = None # Update function (output True, None, or False)
  _last = None   # Last accepted OPQR object

  # Private
  __counter = None # Step counter

#-------------------------------------------------------------------------------
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    self.reset_last()

#-------------------------------------------------------------------------------
  def _refresh(self):
    super()._refresh()
    if self._marg is None and self._cond is None:
      return
    self.opqrstu = collections.namedtuple(self._id, 
                       ['o', 'p', 'q', 'r', 's', 't', 'u'])

#-------------------------------------------------------------------------------
  def reset_last(self):
    self.__last = None
    self.__counter = None

#-------------------------------------------------------------------------------
  def ret_counter(self):
    return self.__counter

#-------------------------------------------------------------------------------
  def next(self, *args, **kwds):
    if self.__counter is None:
      self.__counter = 0

    # Treat sampling without proposals as a sk
    if self._last is None or (self._tran is None and self._prop is None):
      self._last = self.sample(*args, **kwds)
    elif len(args) < 2:
      self._last = self.sample(self._last, **kwds)
    else:
      args = tuple(self._list + list(args[1:]))
      self._last = self.sample(*args, **kwds)
    self.__counter += 1
    return self._last

#-------------------------------------------------------------------------------
  def sampler(self, *args, **kwds):
    if not args:
      args = {0},
    elif len(args) == 1 and type(args[0]) is int and 'stop' not in kwds:
      kwds.update({'stop': args[0]})
      args = {0},
    stop = None if 'stop' not in kwds else kwds.pop('stop')
    return sample_generator(self, *args, stop=stop, **kwds)

#-------------------------------------------------------------------------------
