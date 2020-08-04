# Example of a 3D normal probability density function with covariance
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

cm = [[2., 0.3, -0.3], [0.3, 1., -0.5], [-0.3, -0.5, 0.5]]
set_lims = (-3., 3.)
set_size_0 = {200}
set_size_1 = {300}
set_size_2 = {400}
rv_0 = prob.RV("norm_0", set_lims, pscale='log')
rv_1 = prob.RV("norm_1", set_lims, pscale='log')
rv_2 = prob.RV("norm_2", set_lims, pscale='log')
norm3d = rv_0 * rv_1  * rv_2
norm3d.set_prob(scipy.stats.multivariate_normal, \
             mean=None, cov=cm)
p_norm3d = norm3d({'norm_0': set_size_0,
                   'norm_1': set_size_1,
                   'norm_2': set_size_2})
p_012 = p_norm3d.rescaled()
pmf = p_012.prob[:-1, :-1, :-1]
