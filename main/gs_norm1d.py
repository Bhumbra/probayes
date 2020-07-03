# Example program to fit 1-dimensional gaussian data using grid search

import numpy as np
import scipy.stats
import prob
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

# Settings
rand_size = 20
rand_mean = 50.
rand_stdv = 10.
mu_lims = {30, 70}
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
model.set_prob(scipy.stats.norm.pdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'})

# Evaluate probabilities
likelihood = model({'x': data, **resolution}, iid=True)
prior = params(likelihood.ret_vals(params.ret_keys()))
joint = prior * likelihood
posterior = joint.conditionalise('x')

# Plot posterior
pcolor(
       np.ravel(posterior.vals['sigma']), 
       np.ravel(posterior.vals['mu']), 
       posterior.prob[:-1, :-1]
      )
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
colorbar()
