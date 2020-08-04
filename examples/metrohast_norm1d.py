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
n_steps = 5000
step_size = (0.002,)
mu_lims = (40, 60)
sigma_lims = (5, 20.)

# SIMULATE DATA
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# SET UP MODEL AND SAMPLER
mu = prob.RV('mu', mu_lims, vtype=float, pscale='log')
sigma = prob.RV('sigma', sigma_lims, vtype=float, pscale='log')
x = prob.RV('x', (-np.inf, np.inf), vtype=float)
sigma.set_vfun((np.log, np.exp))
params = prob.SJ(mu, sigma)
stats = prob.SJ(x)
process = prob.SP(stats, params)
process.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'})
params.set_tran(lambda **x: 1.)
params.set_delta(step_size, scale=True)
process.set_tran(params)
process.set_delta(params)
process.set_scores('hastings')
process.set_update('metropolis')
init_state = {'mu': np.mean(mu_lims), 'sigma': np.mean(sigma_lims)}
sampler = process.sampler(init_state, {'x': data}, stop=n_steps, iid=True, joint=True)
samples = [sample for sample in sampler]
summary = process(samples)
inference = summary.v.rescaled()
mu, sigma, post = inference.vals['mu'], inference.vals['sigma'], inference.prob

# PLOT DATA
figure()
c_norm = Normalize(vmin=np.min(post), vmax=np.max(post))
c_map = cm.jet(c_norm(post))
plot(mu, sigma, '-', color=(0.7, 0.7, 0.7, 0.3))
scatter(mu, sigma, color=c_map, marker='.', alpha=1.)
xlabel(r'$\mu$')
ylabel(r'$\sigma$')
yscale('log')
