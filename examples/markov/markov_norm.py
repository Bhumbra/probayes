""" Example of a Markov chain evolution from a discretised
transition matrix defined using a continuous transition 
function. Probability multiplications are performed using 
summations in log-space.
"""
import probayes as pb
import numpy as np
import scipy.stats
from pylab import *; ion()

n_steps = 6
set_lims = [-np.pi, np.pi]
set_size = {200}

def tran(succ, pred):
  loc = -np.sin(pred)
  scale = 1. + 0.5 * np.cos(pred)
  logdx = np.log((max(set_lims) - min(set_lims)) / list(set_size)[0])
  return logdx+scipy.stats.norm.logpdf(succ, loc=loc, scale=scale)

x = pb.RV('x', set_lims, pscale='log')
x.set_tran(tran, order={"x'": 0, 'x': 1})
cond = x.step(set_size)
conds = [None] * n_steps
for i in range(n_steps):
  recond = cond.rekey({'x': 'x_{}'.format(i), 
                       "x'": 'x_{}'.format(i+1)})
  if i == 0:
    last_cond = recond
  else:
    joint = recond * last_cond
    last_cond = joint.marginalise("x_{}".format(i))
  conds[i] = last_cond.rescaled()

# Plot conditionals
figure()
nr = int(np.floor(np.sqrt(n_steps)))
nc = int(np.ceil(n_steps / nr))
for i in range(n_steps):
  subplot(nr, nc, i+1)
  pcolor(
         np.ravel(cond.vals["x'"]), 
         np.ravel(cond.vals['x']), 
         conds[i].prob[:-1, :-1], cmap=cm.jet,
        )
  colorbar()
  xlabel(r'$x_{}$'.format(0))
  ylabel(r'$x_{}$'.format(i+1))
