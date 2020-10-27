# Improved example to fit 1-dimensional gaussian data using discrete grid exact inference.

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
resolution = {'mu': {128}, 'sigma': {192}}

# Generate data
data = np.random.normal(loc=rand_mean, scale=rand_stdv, size=rand_size)

# Declare RVs
mu = pb.RV('mu', mu_lims, vtype=float)
sigma = pb.RV('sigma', sigma_lims, vtype=float)
x = pb.RV('x', {-np.inf, np.inf}, vtype=float)

# Set reciprocal prior for  sigma
sigma.set_mfun((np.log, np.exp))

# Set up params and models
paras = pb.RF(mu, sigma)
stats = pb.RF(x)
model = pb.SD(stats, paras)
model.set_prob(scipy.stats.norm.logpdf,
               order={'x':0, 'mu':'loc', 'sigma':'scale'},
               pscale='log')

# Evaluate log probabilities
posterior = model({'x': data, **resolution}, 
                  iid=True, joint=True).conditionalise('x')

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
       np.ravel(posterior.vals['sigma']), 
       np.ravel(posterior.vals['mu']), 
       post_prob[:-1, :-1], cmap=cm.jet,
      )
colorbar()
xlabel(r'$\sigma$')
ylabel(r'$\mu$')
title(str(dict(post_expt)))
xscale('log')
