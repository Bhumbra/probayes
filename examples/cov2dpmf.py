# Example of a 2D normal probability density function with covariance
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

set_lims = (-3., 3.)
set_size_0 = {200}
set_size_1 = {300}
rv_0 = prob.RV("norm_0", set_lims, pscale='log')
rv_1 = prob.RV("norm_1", set_lims, pscale='log')
norm2d = rv_0 * rv_1
norm2d.set_prob(scipy.stats.multivariate_normal, \
             [0., 0.], [[2.0, -0.3], [-0.3, 0.5]])
p_norm2d = norm2d({'norm_0': set_size_0,
                   'norm_1': set_size_1})
p_01 = p_norm2d.rescaled()
pmf = p_01.prob[:-1, :-1]
figure()
pcolor(np.ravel(p_01.vals['norm_1']), 
       np.ravel(p_01.vals['norm_0']), 
       pmf, cmap=cm.jet)
colorbar()
