# Example of a normal probability density function
import numpy as np
import scipy.stats as ss
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = (-3, 3)
num_samples = 200
rv = prob.RV("norm", norm_range, prob=ss.norm, ptype='log', loc=0, scale=1)
pdf = rv(num_samples)
plot(pdf.vals, pdf.prob)
