# Example of a 1D normal probability density function
import scipy.stats
from pylab import *; ion()
import probayes as pb

set_lims = [-3., 3.]
set_size = {100}
rv = pb.RV("norm", set_lims, prob=scipy.stats.norm, 
             pscale='log', loc=0, scale=1)
logpdf = rv(set_size)
pdf = logpdf.rescaled()
figure()
plot(pdf['norm'], pdf.prob)
