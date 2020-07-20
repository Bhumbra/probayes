""" Example of a Markov chain random walk simulation 
using a continuous transition function.
"""
import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 2000
set_lims = {-np.pi, np.pi}

def tran(succ, pred):
  loc = -np.sin(pred)
  scale = 1. + 0.5 * np.cos(pred)
  return scipy.stats.norm.pdf(succ, loc=loc, scale=scale)

def tcdf(succ, pred):
  loc = -np.sin(pred)
  scale = 1. + 0.5 * np.cos(pred)
  return scipy.stats.norm.cdf(succ, loc=loc, scale=scale)

def ticdf(succ, pred):
  loc = -np.sin(pred)
  scale = 1. + 0.5 * np.cos(pred)
  return scipy.stats.norm.ppf(succ, loc=loc, scale=scale)

x = prob.RV('x', set_lims)
x.set_tran(tran, order={'x': 'pred', "x'": 'succ'})
x.set_tfun((tcdf, ticdf), order={'x': 'pred', "x'": 'succ'})

cond = [None] * n_steps
pred = np.empty(n_steps, dtype=float)
succ = np.empty(n_steps, dtype=float)
prob = np.empty(n_steps, dtype=float)
print('Simulating...')
for i in range(n_steps):
  if i == 0:
    cond[i] = x.step(0.)
  else:
    dx = x.delta(({np.random.uniform(-0.1, 0.1)},))
    cond[i] = x.step(succ[i-1], dx)
  pred[i] = cond[i].vals['x']
  succ[i] = cond[i].vals["x'"]
  prob[i] = cond[i].prob
print('...done')


# PLOT DATA
figure()
c_norm = Normalize(vmin=np.min(prob), vmax=np.max(prob))
c_map = cm.jet(c_norm(prob))
scatter(pred, succ,  color=c_map, marker='.')
xlabel('Predecessor')
ylabel('Successor')
