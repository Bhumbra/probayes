""" Example of a Markov chain random walk simulation 
using a continuous transition function.
"""
import probayes as pb
import numpy as np
import scipy.stats
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
X = pb.SP(x)
sampler = X.sampler({0}, stop=n_steps)
samples = X.walk(sampler)
summary = X(samples)
cond = summary.q.prob
pred = summary.q["x"]
succ = summary.q["x'"]


# PLOT DATA
figure()
c_norm = Normalize(vmin=np.min(cond), vmax=np.max(cond))
c_map = cm.jet(c_norm(cond))
scatter(pred, succ, color=c_map, marker='.')
xlabel('Predecessor')
ylabel('Succesor')
