"""
Simple Metropolis Hastings sampler for evaluating posterior for mean and stdv
"""

import probayes as pb
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
step_size = (0.005,)
mu_lims = (40, 60)
sigma_lims = (5, 20.)

# SIMULATE DATA
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# SET UP MODEL AND SAMPLER
mu = pb.RV('mu', mu_lims, vtype=float, pscale='log')
sigma = pb.RV('sigma', sigma_lims, vtype=float, pscale='log')
x = pb.RV('x', (-np.inf, np.inf), vtype=float)
sigma.set_mfun((np.log, np.exp))
paras = pb.RF(mu, sigma)
stats = pb.RF(x)
process = pb.SP(stats, paras)
process.set_prob(scipy.stats.norm.logpdf,
                 order={'x':0, 'mu':'loc', 'sigma':'scale'})
tran = lambda **x: 1.
paras.set_tran((tran, tran))
paras.set_delta(step_size, scale=True)
process.set_tran(paras)
process.set_delta(paras)
process.set_scores('hastings')
process.set_update('metropolis')
init_state = {'mu': np.mean(mu_lims), 'sigma': np.mean(sigma_lims)}
sampler = process.sampler(init_state, {'x': data}, stop=n_steps, iid=True, joint=True)
#samples = [sample for sample in sampler] # <- use list comprehension or process.walk
samples = process.walk(sampler)
summary = process(samples)
inference = summary.v.rescaled()
n_accept = summary.u.count(True)
mu, sigma, post = inference.vals['mu'], inference.vals['sigma'], inference.prob
hat_mu = np.median(mu)
hat_sigma = np.median(sigma)
hat_mu_str = '{:.2f}'.format(hat_mu)
hat_sigma_str = '{:.2f}'.format(hat_sigma)


# PLOT DATA
figure()
c_norm = Normalize(vmin=np.min(post), vmax=np.max(post))
c_map = cm.jet(c_norm(post))
plot(mu, sigma, '-', color=(0.7, 0.7, 0.7, 0.3))
scatter(mu, sigma, color=c_map, marker='.', alpha=1.)
xlabel(r'$\mu$')
ylabel(r'$\sigma$')
title(r'$\hat{\mu}=' + hat_mu_str + r',\hat{\sigma}=' + hat_sigma_str + r'$')
yscale('log')
