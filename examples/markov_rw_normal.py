""" Example of a Markov chain random walk simulation 
using a continuous transition function.
"""
import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 6
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
d = x.step({0})
