# Example of a normal probability density function
import numpy as np
import scipy.stats as ss
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = (-3, 3)
num_samples = 200
norm_rv = prob.RV("norm", norm_range, prsc=0.)
norm_rv.set_prob(ss.norm.logpdf, loc=0, scale=1)
call_rv = norm_rv(num_samples)
plot(call_rv.samp, call_rv.prob)
