"""
A stocastic process is an indexable sequence of realisations of a stochastic 
condition. It is therefore implemented here using a sample generator that 
iteratively samples a stochastic condition.
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
      sp.reset(sp.ret_last())

#-------------------------------------------------------------------------------
class SP (SC):

  # Public
  opqrstu = None # opqr + scores + thresh + update

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
    self.opqrstu = collections.namedtuple(self._id, 
                       ['o', 'p', 'q', 'r', 's', 't', 'u'])

#-------------------------------------------------------------------------------
  def reset(self, last=None, counter=None): # this allows optional preservation
    self.__last = last
    self.__counter = counter

#-------------------------------------------------------------------------------
  def ret_last(self):
    return self.__last

#-------------------------------------------------------------------------------
  def ret_counter(self):
    return self.__counter

#-------------------------------------------------------------------------------
  def next(self, *args, **kwds):
    if self.__counter is None:
      self.reset(self.ret_last(), 0)

    # Treat sampling without proposals as a distribution call
    if self.__last is None or (self._tran is None and self._prop is None):
      self.__last = self.sample(*args, **kwds)

    # Otherwise refeed last proposals into sample function
    elif len(args) < 2:
      self.__last = self.sample(self.__last, **kwds)
    else:
      args = tuple(self._list + list(args[1:]))
      self.__last = self.sample(*args, **kwds)
    
    # Increment counter
    self.__counter += 1
    return self.__last

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
