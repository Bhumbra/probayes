# Example of a 1D normal probability density function
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

set_lims = [-3., 3.]
set_size = {100}
rv = prob.RV("norm", set_lims, prob=scipy.stats.norm, 
             pscale='log', loc=0, scale=1)
logpdf = rv(set_size)
pdf = logpdf.rescaled()
figure()
plot(pdf.vals['norm'], pdf.prob)
