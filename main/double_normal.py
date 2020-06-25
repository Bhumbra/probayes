# Example of a normal probability density function
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

norm_range = {-3, 3}
set_size_0 = {20}
set_size_1 = {30}
rv_0 = prob.RV("norm_0", norm_range, prob=scipy.stats.norm, 
               ptype='log', loc=0, scale=1)
rv_1 = prob.RV("norm_1", norm_range, prob=scipy.stats.norm, 
               ptype='log', loc=0, scale=1)
p_0 = rv_0(set_size_0)
p_1 = rv_1(set_size_1)
p_01 = p_0 * p_1
p_01.rescale()
pcolor(np.ravel(p_01.vals['norm_1']), 
       np.ravel(p_01.vals['norm_0']), 
       p_01.prob[:-1, :-1])
colorbar()
