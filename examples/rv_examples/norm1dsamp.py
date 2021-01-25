# Example of sampling from a normal probability density function
import scipy.stats
from pylab import *; ion()
import probayes as pb

norm_range = {-2., 2.}
set_size = {-10000} # size negation denotes random sampling
x = pb.RV("x", norm_range, prob=scipy.stats.norm, loc=0, scale=1)
rx = x.eval_vals(set_size)
hist(rx['x'], 100)
