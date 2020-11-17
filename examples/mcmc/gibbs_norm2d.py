"""
Gibbs sampler for a 2D Gaussian PDF.
"""

import probayes as pb
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

lims = (-10., 10.)
n_steps = 2000
means = [0.5, -0.5]
covar = [[1.5, -1.0], [-1.0, 2.]]

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  y, yprime = kwds['y'], kwds["y'"]
  return scipy.stats.norm.pdf(yprime, loc=y, scale=prop_stdv) * \
         scipy.stats.norm.pdf(xprime, loc=x, scale=prop_stdv)

x = pb.RV('x', lims, vtype=float)
y = pb.RV('y', lims, vtype=float)
process = pb.SP(x*y)
process.set_prob(scipy.stats.multivariate_normal, means, covar)
process.set_tran(scipy.stats.multivariate_normal, means, covar, tsteps=1)
process.set_scores('gibbs')
sampler = process.sampler({'x': 0., 'y': 1.}, stop=n_steps)
samples = [sample for sample in sampler]
summary = process(samples)
n_accept = summary.u.count(True)
inference = summary.v.rescaled()
xvals, yvals, post = inference.vals['x'], inference.vals['y'], inference.prob

# PLOT DATA
figure()
subplot(2, 2, 1)
c_norm = Normalize(vmin=np.min(post), vmax=np.max(post))
c_map = cm.jet(c_norm(post))
plot(xvals, yvals, '-', color=(0.7, 0.7, 0.7, 0.3))
scatter(xvals, yvals, color=c_map, marker='.', alpha=1.)
xlabel(r'$x$')
ylabel(r'$y$')

ax = subplot(2, 2, 2)
ybins = np.linspace(np.min(yvals), np.max(yvals), 128)
yhist, _ = np.histogram(yvals, ybins)
yprop = yhist / (n_steps * (ybins[1]-ybins[0]))
step(ybins[:-1], yprop, 'b')
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
xlabel(r'$y$')
ylabel('Relative frequency')

subplot(2, 2, 3)
xbins = np.linspace(np.min(xvals), np.max(xvals), 128)
xhist, _ = np.histogram(xvals, xbins)
xprop = xhist / (n_steps * (xbins[1]-xbins[0]))
step(xbins[:-1], xprop, 'b')
xlabel(r'$x$')
ylabel('Relative frequency')
