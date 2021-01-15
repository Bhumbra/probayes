# Example of a 1D normal probability density function
import sympy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

class fingy:
  def __getitem__(*args, **kwds):
    _args, _kwds = args, kwds
  def __call__(*args, **kwds):
    _args, _kwds = args, kwds

set_lims = [-3., 3.]
set_size = {100}
rv = pb.RV("norm", float)
rv.set_prob(sympy.stats.Normal)
a = fingy()
