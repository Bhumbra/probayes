""" 
Example of linear regression using Gibbs taken from Radford Neil's slides at:
http://www.cs.toronto.edu/~radford/csc2541.S11/week3.pdf

p(y|x, beta_0, beta_1, y_sigma) = N(beta_1*x + beta_0, y_sigma)
p(beta_0) = N(beta_0_mu, beta_0_sigma)
p(beta_1) = N(beta_1_mu, beta_1_sigma)
p(1/y_sigma^2) = Gamma(y_sigma_alpha, 1/y_sigma_beta)
""" 
import numpy as np
import scipy.stats
import probayes as pb
from pylab import *; ion()
from mpl_toolkits.mplot3d import Axes3D # import needed for 3D projection

n_steps = 1000

# Simulate data
rand_size = 60
x_range = [-3, 3]
slope = 1.5
intercept = -1.
y_noise = 0.5
x_obs = np.random.normal(0, 1, size=rand_size)
y_obs = np.random.normal(slope*x_obs + intercept, y_noise)

# Set up RVs, RFs, and SP
x = pb.RV('x', vtype=float, vset=x_range)
y = pb.RV('y', vtype=float, vset=[-np.inf, np.inf]) 
beta_0 = pb.RV('beta_0', vtype=float, vset=[-6., 6.]) 
beta_1 = pb.RV('beta_1', vtype=float, vset=[-6., 6.]) 
y_sigma = pb.RV('y_sigma', vtype=float, vset=[(0.001), 10.]) 

# Define likelihood and conditional functions
def norm_reg(x, y, beta_0, beta_1, y_sigma):
  return scipy.stats.norm.logpdf(y, loc=beta_0 + beta_1*x, scale=y_sigma)

def cond_reg(x, y, beta_0, beta_1, y_sigma, unknown,
             beta_0_mu=0, beta_0_sigma=1, beta_1_mu=0, beta_1_sigma=1.,
             y_sigma_alpha=1., y_sigma_beta=1.):
  if unknown == 'y_sigma':
    cond_alpha = y_sigma_alpha + 0.5*rand_size
    cond_beta = y_sigma_beta + 0.5*np.sum((y - beta_0 - beta_1*x)**2)
    y_sigma = 1 / np.sqrt(np.random.gamma(cond_alpha, 1/cond_beta))
    return y_sigma

  y_prec = 1 / (y_sigma**2)

  if unknown == 'beta_0':
    beta_0_prec = 1/(beta_0_sigma**2)
    cond_var = 1 / (beta_0_prec + rand_size*y_prec)
    cond_mu = (beta_0_prec*beta_0_mu + y_prec*np.sum(y - beta_1*x)) * cond_var
    cond_sigma = np.sqrt(cond_var)
    beta_0 = np.random.normal(cond_mu, cond_sigma)
    return beta_0

  if unknown == 'beta_1':
    beta_1_prec = 1/(beta_1_sigma**2)
    cond_var = 1 / (beta_1_prec + y_prec*np.sum(x**2))
    cond_mu = (beta_1_prec*beta_1_mu + y_prec*np.sum(x*(y - beta_0))) * cond_var
    cond_sigma = np.sqrt(cond_var)
    beta_1 = np.random.normal(cond_mu, cond_sigma)
    return beta_1 
  
  raise ValueError("Unknown unknown: {}".format(unknown))

# Setup up RFs and SP
stats = x & y
paras = beta_0 & beta_1 & y_sigma
paras.set_tfun(cond_reg, tsteps=1, x=x_obs, y=y_obs)
process = pb.SP(stats, paras)
process.set_tfun(paras)
process.set_prob(norm_reg, pscale='log')
process.set_scores('gibbs')
lr = scipy.stats.linregress(x_obs, y_obs)
init_state = {'beta_0': lr.intercept, 'beta_1': lr.slope, 'y_sigma': np.sqrt(lr.stderr)}
sampler = process.sampler(init_state, {'x,y': [x_obs,y_obs]}, stop=n_steps, iid=True, joint=True)
samples = [sample for sample in sampler]
summary = process(samples)
n_accept = summary.u.count(True)
inference = summary.v.rescaled()
b0, b1, ys, post = inference['beta_0'], inference['beta_1'], \
                   inference['y_sigma'], inference.prob
hat_beta_0 = np.median(b0)
hat_beta_1 = np.median(b1)
hat_y_sigma = np.median(ys)
hat_beta_0_str = '{:.2f}'.format(hat_beta_0)
hat_beta_1_str = '{:.2f}'.format(hat_beta_1)
hat_y_sigma_str = '{:.2f}'.format(hat_y_sigma)

# PLOT DATA
fig = figure()
ax = fig.add_subplot(111, projection='3d')
c_norm = Normalize(vmin=np.min(post), vmax=np.max(post))
c_map = cm.jet(c_norm(post))
ax.plot(b0, b1, ys, '-', color=(0.7, 0.7, 0.7, 0.3))
ax.scatter(b0, b1, ys, color=c_map, marker='.', alpha=1.)
ax.set_xlabel(r'$\beta_0$')
ax.set_ylabel(r'$\beta_1$')
ax.set_zlabel(r'$\sigma_y$')
ax.set_title(r'$\hat{\beta_0}=' + hat_beta_0_str + r',\hat{\beta_1}=' + hat_beta_1_str + \
             r',\hat{\sigma_y}=' + hat_y_sigma_str + r'$')
