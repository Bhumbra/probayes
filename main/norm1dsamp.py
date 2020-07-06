# Example of a normal probability density function
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = {-2., 2.}
set_size = {-10000} # size negation denotes random sampling
rv = prob.RV("trun_norm", norm_range, prob=scipy.stats.norm,loc=0, scale=1)
samples = rv.eval_vals(set_size)
hist(samples, 100)
