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
resolution = {'mu': {100}, 'sigma': {200}}

# Generate data
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# RVs
mu = prob.RV('mu', mu_lims, vtype=float)
sigma = prob.RV('sigma', sigma_lims, vtype=float)
x = prob.RV('x', {-np.inf, np.inf}, vtype=float)

# Set reciprocal prior for  sigma
sigma.set_prob(lambda x: 1./x)
#sigma.set_vfun((np.log, np.exp))

# Set up params and models
params = prob.SJ(mu, sigma)
#params.set_use_vfun(False)
model = prob.SC(x, params)
model.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'},
               pscale='log')

# Evaluate log probabilities
likelihood = model({'x': data, **resolution}, iid=True)
prior = params(likelihood.ret_vals(params.ret_keys()))
prior_x_likelihood = prior * likelihood
evidence = prior_x_likelihood.marginal('x')
posterior = prior_x_likelihood / evidence
#posterior = (prior * likelihood).conditionalise('x')

# Exponentialise log probabilities
post_prob = posterior.rescale()

# Plot posterior
figure()
pcolor(
       np.ravel(posterior.vals['sigma']), 
       np.ravel(posterior.vals['mu']), 
       post_prob[:-1, :-1]
      )
colorbar()
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
xscale('log')
