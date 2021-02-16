# Improved example to fit 1-dimensional gaussian data using discrete grid exact inference.

import numpy as np
import sympy
import sympy.stats
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
x = pb.RV('x', vtype=float, vset={-pb.OO, pb.OO})

# Set reciprocal prior for  sigma
sigma.set_ufun(sympy.log(sigma[:]))

# Set up params and models
paras = pb.RF(mu, sigma)
stats = pb.RF(x)
model = pb.SD(stats, paras)
model.set_prob(sympy.stats.Normal(x[:], mean=mu[:], std=sigma[:]),
               pscale='log')

# Evaluate log probabilities
joint = model({x: data, **resolution}, iid=True, joint=True)
posterior = joint.conditionalise('x')

# Return posterior probability mass
post_expt = posterior.expectation()
post_expt.pop('x')
post_mean = posterior.marginal('mu')
post_stdv = posterior.marginal('sigma')
post_mean_medn = post_mean.quantile()
post_stdv_medn = post_stdv.quantile()
post_prob = posterior.rescaled().prob

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
xscale('log')
