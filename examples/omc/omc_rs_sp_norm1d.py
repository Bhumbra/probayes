""" Example of ordinary Monte Carlo random sampling a 1-dimensional gaussian model """
import numpy as np
import scipy.stats
from matplotlib.colors import Normalize
from pylab import *; ion()
import probayes as pb

# Settings
rand_size = 60
rand_mean = 50.
rand_stdv = 10.
mu_lims = (40, 60)
sigma_lims = (5, 20.)
n_samples = 5000

# Generate data
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# Declare RVs
mu = pb.RV('mu', vtype=float, vset=mu_lims)
sigma = pb.RV('sigma', vtype=float, vset=sigma_lims)
x = pb.RV('x', vtype=float, vset=[-np.inf, np.inf])

# Set reciprocal prior for  sigma
sigma.set_ufun((np.log, np.exp))

# Set up params and models
paras = pb.RF(mu, sigma)
stats = pb.RF(x)
process = pb.SP(stats, paras)
process.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'},
               pscale='log')

# SAMPLE AND SUMMARISE
sampler = process.sampler({'mu': {0}, 'sigma': {0}, 'x': data}, 
                          iid=True, joint=True, stop=n_samples)
samples = [sample for sample in sampler]
summary = process(samples)

# DETERMINE HAT VALUES
inference = summary.rescaled()
mu, sigma, post = inference['mu'], inference['sigma'], inference.prob
mu_sort = inference.sorted('mu')
sigma_sort = inference.sorted('sigma')
hat_mu = mu_sort.quantile(0.5)['mu']
hat_sigma = sigma_sort.quantile(0.5)['sigma']
hat_mu_str = '{:.2f}'.format(hat_mu)
hat_sigma_str = '{:.2f}'.format(hat_sigma)


# Plot posterior
figure()
c_norm = Normalize(vmin=np.min(post), vmax=np.max(post))
c_map = cm.jet(c_norm(post))
scatter(mu, sigma, color=c_map, marker='.', alpha=1.)
xlabel(r'$\mu$')
ylabel(r'$\sigma$')
title(r'$\hat{\mu}=' + hat_mu_str + r',\hat{\sigma}=' + hat_sigma_str + r'$')
yscale('log')
