# Example of a 2D multivariate normal transition function covariance
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

set_lims = (-3., 3.)
nsteps
x = pb.RV(x, set_lims)
y = pb.RV(y, set_lims)
xy = x * y
norm2d.set_prob(scipy.stats.multivariate_normal, \
             [0., 0.], [[2.0, -0.3], [-0.3, 0.5]])
norm2d.set_tran(scipy.stats.multivariate_normal, \
             [0., 0.], [[2.0, -0.3], [-0.3, 0.5]])
x_vals = [], 
p_01 = p_norm2d.rescaled()
pmf = p_01.prob[:-1, :-1]
figure()
pcolor(np.ravel(p_01.vals['norm_1']), 
       np.ravel(p_01.vals['norm_0']), 
       pmf, cmap=cm.jet)
colorbar()
