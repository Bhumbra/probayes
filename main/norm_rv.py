# Example of a normal probability density function
import numpy as np
import scipy.stats as ss
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
from prob.rv import RV

norm_range = range(-3, 3)
num_samples = 100
norm_rv = RV("Norm RV", norm_range, log_fun=True)
norm_rv.set_fun(ss.norm.logpdf, loc=0, scale=1)
samples, sam_pdf = norm_rv(num_samples)
plot(samples, sam_pdf)
