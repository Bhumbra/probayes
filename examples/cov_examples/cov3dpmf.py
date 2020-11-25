# Example of a 3D normal probability density function with covariance
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

set_lims = (-3., 3.)
set_size_0 = {200}
set_size_1 = {300}
set_size_2 = {400}
means = [0.5, 0., -0.5]
covar = [[2., 0.3, -0.3], [0.3, 1., -0.5], [-0.3, -0.5, 0.5]]
x = pb.RV("x", set_lims, pscale='log')
y = pb.RV("y", set_lims, pscale='log')
z = pb.RV("z", set_lims, pscale='log')
xyz = x & y & z
xyz.set_prob(scipy.stats.multivariate_normal, mean=means, cov=covar)
pxyz = xyz({'x': set_size_0, 'y': set_size_1, 'z': set_size_2})
p_xyz = pxyz.rescaled()
pmf = p_xyz.prob[:-1, :-1, :-1]
