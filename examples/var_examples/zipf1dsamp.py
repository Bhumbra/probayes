# Example of sampling from a Zipfian probability density function
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

zipf_range = [0.1, 10.]
set_size = {-10000} # size negation denotes random sampling
var = pb.Variable("var", zipf_range)
var.set_ufun((np.log, np.exp))
samples = var.eval_vals(set_size)[var.name]
figure()
hist(samples, 100)
