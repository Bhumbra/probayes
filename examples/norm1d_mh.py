"""
Simple Metropolis Hastings sampler taken from Hogg and Foreman-MacKey(2018):

  From Problem 2:
  Sample in a single parameter x and give the sampler 
  as its density function p(x) a Gaussian density with 
  mean 2 and variance 2. Make the proposal distribution 
  q (x'∣x ) a Gaussian pdf for x′ with mean x and 
  variance 1. Initialize the sampler with x = 0 and run 
  the sampler for more than 104 steps.Plot the results as
  a histogram with the true density overplotted sensibly.

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
sampler = prob.SC(x)
sampler.set_prob(scipy.stats.norm.pdf, loc=2, scale=np.sqrt(2),
                 order={'x': 0})
sampler.set_tran(q)
x = np.zeros(n_steps+1, dtype=float)
accept = np.ones(n_steps+1, dtype=bool)

for i in range(n_steps+1):
  if i == 0:
    x_old = x[i]
    p_old = sampler({'x': x_old}).prob
  else:
    x_new = x_old + 2.*max_step_size*(np.random.uniform()-0.5)
    q_xx = sampler.step({'x': x_old}, {"x'": x_new})
    p_new = sampler({'x': x_new}).prob
    hastings_ratio = p_new / p_old
    accept[i] = hastings_ratio > np.random.uniform() 
    if accept[i]:
      p_old = p_new
      x[i] = x_new
    else:
      x[i] = x[i-1]
    x_old = x[i]

figure()
xbins = np.linspace(np.min(x), np.max(x), 100)
xhist, _ = np.histogram(x, xbins)
xprop = xhist / (n_steps * (xbins[1]-xbins[0]))
step(xbins[:-1], xprop, 'b')
f_x = scipy.stats.norm.pdf(xbins, loc=2, scale=np.sqrt(2)) 
plot(xbins, f_x, 'r')
