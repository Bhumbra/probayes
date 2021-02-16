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
  
  Redo Problem 2, but now with an input density that
  is a function of two variables (x, y). For the density 
  function use two different functions. (b) The second 
  density function is a rectangular top-hat function that is 
  uniform on the joint constraint 3<x<7 and
  1<y<9 and zero everywhere else. For the proposal
  distribution q (x',y'∣x, y) use a two-dimensional Gaussian
  density with mean at [x, y] and variance tensor set to the
  two-dimensional identity matrix. 

"""

import probayes as pb
import numpy as np
import scipy.stats
from pylab import *; ion()

n_steps = 12288
prop_stdv = np.sqrt(1)

def p(**kwds):
  return scipy.stats.uniform.pdf(kwds['x'], loc=3., scale=4.) * \
         scipy.stats.uniform.pdf(kwds['y'], loc=1., scale=8.)

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  y, yprime = kwds['y'], kwds["y'"]
  return scipy.stats.norm.pdf(yprime, loc=y, scale=prop_stdv) * \
         scipy.stats.norm.pdf(xprime, loc=x, scale=prop_stdv)

x = pb.RV('x', vtype=float, vset=(-np.inf, np.inf))
y = pb.RV('y', vtype=float, vset=(-np.inf, np.inf))
process = pb.SP(x & y)
process.set_prob(p)
process.set_tran(q)
lambda_delta = lambda : process.Delta(x=scipy.stats.norm.rvs(loc=0., scale=prop_stdv),
                                      y=scipy.stats.norm.rvs(loc=0., scale=prop_stdv))
process.set_delta(lambda_delta)
process.set_scores('hastings')
process.set_update('metropolis')
sampler = process.sampler({'x': 5., 'y': 5.}, stop=n_steps)
samples = [sample for sample in sampler]
summary = process(samples)
n_accept = summary.u.count(True)
inference = summary.v.rescaled()
xvals, yvals, post = inference['x'], inference['y'], inference.prob

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
