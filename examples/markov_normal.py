""" Example of a Markov Chain random walk conditioned by a
continuous transition function. 
"""
import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

n_steps = 6
set_lims = {-np.pi, np.pi}
set_size = {np.uint(200)} # np.uint encodes include limits

def tran(succ, pred):
  loc = -np.sin(pred)
  scale = 0.5 + 0.25 * np.cos(pred)
  dx = (max(set_lims) - min(set_lims)) / list(set_size)[0]
  return dx*scipy.stats.norm.pdf(succ, loc=loc, scale=scale)

x = prob.RV('x', set_lims)
x.set_tran(tran, order={'x': 'pred'})
cond = x.step(set_size)
steps = [None] * n_steps
for i in range(n_steps):
  recond = cond.rekey({'x': 'x_{}'.format(i), 
                       "x'": 'x_{}'.format(i+1)})
  if i == 0:
    steps[i] = recond
  else:
    last_cond = steps[i-1]
    joint = recond * last_cond
    steps[i] = joint.marginalise("x_{}".format(i))

# Plot conditionals
figure()
nr = int(np.floor(np.sqrt(n_steps)))
nc = int(np.ceil(n_steps / nr))
for i in range(n_steps):
  subplot(nr, nc, i+1)
  pcolor(
         np.ravel(cond.vals["x'"]), 
         np.ravel(cond.vals['x']), 
         steps[i].prob[:-1, :-1], cmap=cm.jet,
        )
  colorbar()
  xlabel(r'$x_{}$'.format(0))
  ylabel(r'$x_{}$'.format(i+1))
