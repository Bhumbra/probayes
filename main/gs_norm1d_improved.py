# Example program to fit 1-dimensional gaussian data using grid search

import numpy as np
import scipy.stats
import prob
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

# Settings
rand_size = 60
rand_mean = 50.
rand_stdv = 10.
mu_lims = {40, 60}
sigma_lims = {5, 20.}
resolution = {'mu': {128}, 'sigma': {192}}

# Generate data
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# Declare RVs
mu = prob.RV('mu', mu_lims, vtype=float)
sigma = prob.RV('sigma', sigma_lims, vtype=float)
x = prob.RV('x', {-np.inf, np.inf}, vtype=float)

# Set reciprocal prior for  sigma
sigma.set_vfun((np.log, np.exp))

# Set up params and models
params = prob.SJ(mu, sigma)
model = prob.SC(x, params)
model.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'},
               pscale='log')

# Evaluate log probabilities
likelihood = model({'x': data, **resolution}, iid=True)
param_vals = likelihood.ret_cond_vals()
prior = params(param_vals)
posterior = prob.product(prior, likelihood).conditionalise('x')

# Return posterior probability mass
inference = posterior.rescaled().prob

# Plot posterior
figure()
pcolor(
       np.ravel(posterior.vals['sigma']), 
       np.ravel(posterior.vals['mu']), 
       inference[:-1, :-1]
      )
colorbar()
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
xscale('log')
