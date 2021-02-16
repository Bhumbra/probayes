# Example to fit 1-dimensional gaussian data using discrete grid exact inference.

import numpy as np
import scipy.stats
import probayes as pb
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
mu = pb.RV('mu', vtype=float, vset=mu_lims)
sigma = pb.RV('sigma', vtype=float, vset=sigma_lims)
x = pb.RV('x', vtype=float, vset=(-np.inf, np.inf))

# Set reciprocal prior for  sigma
sigma.set_prob(lambda x: 1./x)

# Set up params and models
paras = mu & sigma
model = x | paras
model.set_prob(scipy.stats.norm.pdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'})

# Evaluate probabilities
likelihood = model({x: data, **resolution}, iid=True)
para_vals = likelihood('cond')
prior = paras(para_vals)
prior_x_likelihood = prior * likelihood
evidence = prior_x_likelihood.marginal('x')
posterior = prior_x_likelihood / evidence

# Return posterior probability mass
post_expt = posterior.expectation()
post_expt.pop('x')
post_mean = posterior.marginal('mu')
post_stdv = posterior.marginal('sigma')
post_mean_medn = post_mean.quantile()
post_stdv_medn = post_stdv.quantile()
post_prob = posterior.prob

# Plot posterior
figure()
pcolor(
       np.ravel(posterior['sigma']), 
       np.ravel(posterior['mu']), 
       post_prob[:-1, :-1], cmap=cm.jet,
      )
colorbar()
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
title(str(dict(post_expt)))
