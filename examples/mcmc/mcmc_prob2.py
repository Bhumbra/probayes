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

import probayes as pb
import numpy as np
import scipy.stats
from pylab import *; ion()

n_steps = 12288

def q(**kwds):
  x, xprime = kwds['x'], kwds["x'"]
  return scipy.stats.norm.pdf(xprime, loc=x, scale=1.)

x = pb.RV('x', vtype=float, vset=(-np.inf, np.inf))
process = pb.SP(x)
process.set_prob(scipy.stats.norm.pdf, loc=2, scale=np.sqrt(2),
                 order={'x': 0})
process.set_tran(q)
lambda_delta = lambda : process.Delta(x=scipy.stats.norm.rvs(loc=0., scale=1.))
process.set_delta(lambda_delta)
process.set_scores('hastings')
process.set_update('metropolis')
sampler = process.sampler({'x': 0.}, stop=n_steps)
samples = [sample for sample in sampler]
summary = process(samples)
xvals = summary.v['x']

figure()
xbins = np.linspace(np.min(xvals), np.max(xvals), 128)
xhist, _ = np.histogram(xvals, xbins)
xprop = xhist / (n_steps * (xbins[1]-xbins[0]))
step(xbins[:-1], xprop, 'b')
norm_x = scipy.stats.norm.pdf(xbins, loc=2, scale=np.sqrt(2)) 
plot(xbins, norm_x, 'r')
