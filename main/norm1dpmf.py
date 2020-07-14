# Example of a 1D normal probability density function
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = {-3., 3.}
set_size = {20}
rv = prob.RV("norm", norm_range, prob=scipy.stats.norm, 
             pscale='log', loc=0, scale=1)
logpdf = rv(set_size)
pdf = logpdf.rescaled()
plot(pdf.vals['norm'], pdf.prob)
