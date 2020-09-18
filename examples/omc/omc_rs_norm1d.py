# Example OMC random sampling to fit a 1D gaussian model

import numpy as np
import scipy.stats
import probayes as pb
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

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
mu = pb.RV('mu', mu_lims, vtype=float)
sigma = pb.RV('sigma', sigma_lims, vtype=float)
x = pb.RV('x', {-np.inf, np.inf}, vtype=float)

# Set reciprocal prior for  sigma
sigma.set_mfun((np.log, np.exp))

# Set up params and models
paras = pb.RJ(mu, sigma)
stats = pb.RJ(x)
model = pb.RF(stats, paras)
model.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'},
               pscale='log')

# Evaluate log probabilities
prior_x_likelihood = model({'x': data, 'mu,sigma': {-n_samples}}, 
                           iid=True, joint=True)
posterior = prior_x_likelihood.conditionalise('x')

# Return posterior probability mass and infer hat values using median
inference = posterior.rescaled()
mu_vals, sigma_vals, post = inference.vals['mu'], inference.vals['sigma'], inference.prob
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
scatter(mu_vals, sigma_vals, color=c_map, marker='.', alpha=1.)
xlabel(r'$\mu$')
ylabel(r'$\sigma$')
title(r'$\hat{\mu}=' + hat_mu_str + r',\hat{\sigma}=' + hat_sigma_str + r'$')
yscale('log')
