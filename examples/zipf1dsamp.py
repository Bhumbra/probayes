# Example of sampling from a Zipfian probability density function
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

zipf_range = [0.1, 10.]
set_size = {-10000} # size negation denotes random sampling
rv = pb.RV("zipf_rv", zipf_range)
rv.set_mfun((np.log, np.exp))
samples = rv.eval_vals(set_size)
figure()
hist(samples, 100)
