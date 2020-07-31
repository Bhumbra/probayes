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

"""

import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 10000
max_step_size = 1.

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  return scipy.stats.norm.pdf(xprime, loc=x, scale=1.)

x = prob.RV('x', (-np.inf, np.inf), vtype=float)
process = prob.SP(x)
process.set_prob(scipy.stats.norm.pdf, loc=2, scale=np.sqrt(2),
                 order={'x': 0})
process.set_tran(q)
process.set_delta([max_step_size])
process.set_scores('hastings')
process.set_update('metropolis')
sampler = process.sampler({'x': 0.}, stop=n_steps)
samples = [sample for sample in sampler]
summary = process(samples)
x = summary.p.vals['x']

figure()
xbins = np.linspace(np.min(x), np.max(x), 100)
xhist, _ = np.histogram(x, xbins)
xprop = xhist / (n_steps * (xbins[1]-xbins[0]))
step(xbins[:-1], xprop, 'b')
norm_x = scipy.stats.norm.pdf(xbins, loc=2, scale=np.sqrt(2)) 
plot(xbins, norm_x, 'r')
