# Example of a 2D normal probability density function
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = {-3., 3.}
set_size_0 = {200}
set_size_1 = {300}
rv_0 = prob.RV("norm_0", norm_range, prob=scipy.stats.norm, 
               pscale='log', loc=0, scale=1)
rv_1 = prob.RV("norm_1", norm_range, prob=scipy.stats.norm, 
               pscale='log', loc=0, scale=1)
l_0 = rv_0(set_size_0)
l_1 = rv_1(set_size_1)
l_01 = l_0 * l_1
p_01 = l_01.rescaled()
pmf = p_01.prob[:-1, :-1]
pcolor(np.ravel(p_01.vals['norm_1']), 
       np.ravel(p_01.vals['norm_0']), 
       pmf, cmap=cm.jet)
colorbar()
