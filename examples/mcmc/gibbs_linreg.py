""" 
Example of linear regression using Gibbs taken from Radford Neil's slides at:
http://www.cs.toronto.edu/~radford/csc2541.S11/week3.pdf
https://kieranrcampbell.github.io/blog/2016/05/15/gibbs-sampling-bayesian-linear-regression.html
""" 
import numpy as np
import scipy.stats
import probayes as pb

# Simulation settings
rand_size = 60
x_range = [-3, 3]
slope = 1.5
intercept = -1.
y_noise = 0.5
x_obs = np.random.uniform(low=x_range[0], high=x_range[1], size=rand_size)
y_obs = np.random.normal(slope*x_obs + intercept, y_noise)

# Set up RVs, RFs, and SP
x = pb.RV('x', x_range, vtype=float)
y = pb.RV('y', [-np.inf, np.inf], vtype=float) 
beta_0 = pb.RV('beta_0', [-np.inf, np.inf], vtype=float) 
beta_1 = pb.RV('beta_1', [-np.inf, np.inf], vtype=float) 
y_sigma = pb.RV('y_sigma', [(0.), np.inf], vtype=float) 
stats = x * y
paras = beta_0 * beta_1 * y_sigma
process = pb.SP(stats, paras)

# Set up likelihoods and conditionals

def norm_reg(x, y, beta_0, beta_1, y_sigma):
  return scipy.stats.norm.pdf(y, loc=beta_0 + beta_1*x, scale=y_sigma)

def cond_reg(x, y, beta_0, beta_1, y_sigma, unknown,
             beta_0_mu=0, beta_0_sigma=1, beta_1_mu=0, beta_1_sigma,
             y_sigma_alpha=1., y_sigma_beta=1.):
  if unknown == 'y_sigma':
    alpha_prime = alpha + rand_size/2



process.set_prob(norm_reg)
