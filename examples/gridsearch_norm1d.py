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
mu_lims = (40, 60)
sigma_lims = (5, 20.)
resolution = {'mu': {128}, 'sigma': {192}}

# Generate data
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# Declare RVs
mu = pb.RV('mu', mu_lims, vtype=float)
sigma = pb.RV('sigma', sigma_lims, vtype=float)
x = pb.RV('x', (-np.inf, np.inf), vtype=float)

# Set reciprocal prior for  sigma
sigma.set_prob(lambda x: 1./x)

# Set up params and models
params = mu * sigma
model = x / params
model.set_prob(scipy.stats.norm.pdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'})

# Evaluate log probabilities
likelihood = model({'x': data, **resolution}, iid=True)
param_vals = likelihood.ret_cond_vals()
prior = params(param_vals)
prior_x_likelihood = prior * likelihood
evidence = prior_x_likelihood.marginal('x')
posterior = prior_x_likelihood / evidence

# Return posterior probability mass
inference = posterior.prob

# Plot posterior
figure()
pcolor(
       np.ravel(posterior.vals['sigma']), 
       np.ravel(posterior.vals['mu']), 
       inference[:-1, :-1], cmap=cm.jet,
      )
colorbar()
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
