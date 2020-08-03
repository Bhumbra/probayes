"""
Simple Metropolis Hastings sampler for evaluating posterior for mean and stdv
"""

import prob
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

# PARAMETERS
rand_size = 60
rand_mean = 50.
rand_stdv = 10.
n_steps = 1000
step_size = (0.05,)
mu_lims = (40, 60)
sigma_lims = (5, 20.)

# SIMULATE DATA
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# SET UP MODEL AND SAMPLER
mu = prob.RV('mu', mu_lims, vtype=float)
sigma = prob.RV('sigma', sigma_lims, vtype=float)
x = prob.RV('x', (-np.inf, np.inf), vtype=float)
sigma.set_vfun((np.log, np.exp))
params = prob.SJ(mu, sigma)
stats = prob.SJ(x)
process = prob.SP(stats, params)
params.set_tran(lambda x: 1.)
params.set_delta(step_size)
process.set_tran(params)
process.set_delta(params)
process.set_scores('hastings')
process.set_update('metropolis')
init_state = {'x': data, 'mu': mu_lims[0], 'sigma': sigma_lims[0]}
sampler = process.sampler(init_state, stop=n_steps, iid=True)
samples = [sample for sample in sampler]
summary = process(samples)
mu, sigma = summary.p.vals['mu'], summary.p.vals['sigma']

