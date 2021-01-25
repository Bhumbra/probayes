# Example of sampling from a Zipfian probability density function
import numpy as np
from pylab import *; ion()
import probayes as pb

zipf_range = [0.1, 10.]
set_size = {-10000} # size negation denotes random sampling
var = pb.Variable("var", zipf_range)
var.set_ufun((np.log, np.exp))
samples = var(set_size)
figure()
hist(samples[var.name], 100)
