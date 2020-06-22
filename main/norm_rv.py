# Example of a normal probability density function
import numpy as np
import scipy.stats as ss
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
from prob.rv import RV

norm_range = (-3, 3)
num_samples = 100
norm_rv = RV("Norm RV", norm_range)
norm_rv.set_prob(ss.norm.pdf, loc=0, scale=1)
samples, sam_pdf = norm_rv(num_samples)
plot(samples, sam_pdf)
