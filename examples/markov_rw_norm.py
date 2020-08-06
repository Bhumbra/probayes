""" Example of a Markov chain random walk simulation 
using a continuous transition function.
"""
import probayes as pb
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 10000
set_lims = [-np.pi, np.pi]

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

x = pb.RV('x', set_lims)
x.set_tran(tran, order={'x': 'pred', "x'": 'succ'})
x.set_tfun((tcdf, ticdf), order={'x': 'pred', "x'": 'succ'})

steps = [None] * n_steps
pred = np.empty(n_steps, dtype=float)
succ = np.empty(n_steps, dtype=float)
cond = np.empty(n_steps, dtype=float)
print('Simulating...')
for i in range(n_steps):
  if i == 0:
    steps[i] = x.step({0})
  else:
    steps[i] = x.step(succ[i-1])
  pred[i] = steps[i].vals['x']
  succ[i] = steps[i].vals["x'"]
  cond[i] = steps[i].prob
print('...done')


# PLOT DATA
figure()
c_norm = Normalize(vmin=np.min(cond), vmax=np.max(cond))
c_map = cm.jet(c_norm(cond))
scatter(pred, succ, color=c_map, marker='.')
xlabel('Predecessor')
ylabel('Succesor')
