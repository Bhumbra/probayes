"""
Simple Metropolis Hastings sampler taken from Hogg and Foreman-MacKey(2018):

  From Problem 2:
  Sample in a single parameter x and give the sampler 
  as its density function p(x) a Gaussian density with 
  mean 2 and variance 2. Make the proposal distribution 
  q (x'∣x ) a Gaussian pdf for x′ with mean x and 
  variance 1. Initialize the sampler with x = 0 and run 
  the sampler for more than 10000 steps. Plot the results 
  as a histogram with the true density overplotted sensibly.
  
  From Problem 4a:
  Redo Problem 2, but now with an input density that is a 
  function of two variables (x, y). For the density function use
  two different functions. (a) The first density function is a
  covariant two-dimensional Gaussian density with variance
  tensor: V = [[2.0, 1.2], [1.2, 2.0]] ... For the proposal
  distribution q (x',y'∣x, y) use a two-dimensional Gaussian
  density with mean at [x, y] and variance tensor set to the
  two-dimensional identity matrix.
  
  Redo Problem 4(a) with a proposal q (x'∣x) with a stupidly 
  shifted mean of x + 2 and see what happens. Bonus points: 
  modify the acceptance–rejection criterion to deal with the 
  messed-up q (x'∣x) and show that everything works once again.

  (Solution used here: Use full 

"""

import probayes as pb
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 12288
prop_stdv = np.sqrt(1)

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  y, yprime = kwds['y'], kwds["y'"]
  return scipy.stats.norm.pdf(yprime, loc=y, scale=prop_stdv) * \
         scipy.stats.norm.pdf(xprime, loc=x+2., scale=prop_stdv)


def r(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  y, yprime = kwds['y'], kwds["y'"]
  return scipy.stats.norm.pdf(yprime, loc=y, scale=prop_stdv) * \
         scipy.stats.norm.pdf(xprime, loc=x-2., scale=prop_stdv)

x = pb.RV('x', (-np.inf, np.inf), vtype=float)
y = pb.RV('y', (-np.inf, np.inf), vtype=float)
process = pb.SP(x & y)
process.set_prob(scipy.stats.multivariate_normal, [0., 0.],
                 [[2.0, 1.2], [1.2, 2.0]])
process.set_tran((q, r))
lambda_delta = lambda : process.delta(x=scipy.stats.norm.rvs(loc=-2, scale=prop_stdv),
                                      y=scipy.stats.norm.rvs(loc=0., scale=prop_stdv))
process.set_delta(lambda_delta)
process.set_scores('hastings')
process.set_update('metropolis')
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
