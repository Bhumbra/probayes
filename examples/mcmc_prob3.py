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
  is uniform on 3<x<7 and zero everywhere else. The plot
  should look like Figure 2. What change did you have to make
  to the initialization, and why?


"""

import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 12288

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  return scipy.stats.norm.pdf(xprime, loc=x, scale=1.)

x = prob.RV('x', (-np.inf, np.inf), vtype=float)
process = prob.SP(x)
process.set_prob(scipy.stats.uniform.pdf, loc=3., scale=4.,
                 order={'x': 0})
process.set_tran(q)
lambda_delta = lambda : process.delta(x=scipy.stats.norm.rvs(loc=0., scale=1.))
process.set_delta(lambda_delta)
process.set_scores('hastings')
process.set_update('metropolis')
sampler = process.sampler({'x': 5.}, stop=n_steps)
samples = [sample for sample in sampler]
summary = process(samples)
xvals = summary.v.vals['x']

figure()
xbins = np.linspace(np.min(xvals), np.max(xvals), 128)
xhist, _ = np.histogram(xvals, xbins)
xprop = xhist / (n_steps * (xbins[1]-xbins[0]))
step(xbins[:-1], xprop, 'b')
norm_x = scipy.stats.uniform.pdf(xbins, loc=3., scale=4.) 
plot(xbins, norm_x, 'r')
