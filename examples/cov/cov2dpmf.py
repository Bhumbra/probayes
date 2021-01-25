# Example of a 2D normal probability density function with covariance
import scipy.stats
from pylab import *; ion()
import probayes as pb

set_lims = (-5., 5.)
set_size_0 = {200}
set_size_1 = {300}
means = [0.5, -0.5]
covar = [[1.5, -1.0], [-1.0, 2.]]
x = pb.RV("x", set_lims, pscale='log')
y = pb.RV("y", set_lims, pscale='log')
xy = x & y
xy.set_prob(scipy.stats.multivariate_normal, means, covar)
pxy = xy({'x': set_size_0, 'y': set_size_1})
p_xy = pxy.rescaled()
pmf = p_xy.prob[:-1, :-1]
figure()
pcolor(np.ravel(p_xy.vals['y']), 
       np.ravel(p_xy.vals['x']), 
       pmf, cmap=cm.jet)
colorbar()
