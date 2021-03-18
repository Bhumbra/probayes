""" Provides named dictionary """
#-------------------------------------------------------------------------------
import collections

#-------------------------------------------------------------------------------
class NamedDict(collections.OrderedDict):
  """ A named dictionary is an ordered dictionary with a name """
  _name = None
  
#-------------------------------------------------------------------------------
  def __init__(self, name, *args, **kwds):
    self.name = name
    super().__init__(*args, **kwds)

#-------------------------------------------------------------------------------
  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    assert isinstance(name, str), \
        "Name must be a string, not {}".format(type(name))
    self._name = name

#-------------------------------------------------------------------------------
  def __repr__(self):
    return "{}: {}".format(self.name, super().__repr__())

#-------------------------------------------------------------------------------
