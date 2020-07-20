""" Example of ordinary Monte Carlo integration with rejection sampling """
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.colors import Normalize
from pylab import *; ion()
import prob

# PARAMETERS
radius = 1.
set_size = {-10000}

# SETUP CIRCLE FUNCTION AND RVs
def inside(o, x, y): 
  return o == (x**2 + y**2 <= radius**2)

def norm2d(x, y, loc=0., scale=radius):
  return scipy.stats.norm.pdf(x, loc=loc, scale=scale) * \
         scipy.stats.norm.pdf(y, loc=loc, scale=scale)

xy_range = [-radius, radius]
x = prob.RV("x", xy_range)
y = prob.RV("y", xy_range)
o = prob.RV("o")

# DEFINE STOCHASTIC CONDITION
xy = x * y
oxy = o / xy
oxy.set_prob(inside)

# DEFINE PROPOSAL DENSITY AND COEFFICIENT VARIABLE
prop_xy = x * y
prop_xy.set_prob(norm2d)
coef_max = float(norm2d(radius, 1.))
coef = prob.RV('coef', {0., coef_max})
coefs = coef(set_size)
p_prop = prop_xy({'x,y': set_size})
thresholds = coefs.vals['coef'] * p_prop.prob

# CALL TARGET DENSITY AND APPLY REJECTION SAMPLING
xy_vals = p_prop.ret_marg_vals()
p_oxy = oxy(xy_vals)
p_oxy_true = p_oxy({'o': True})
accept = p_oxy_true.prob >= thresholds
reject = np.logical_not(accept)
expt = np.mean(accept)
square_area = 4. * radius**2
circle_area = square_area * expt

# PLOT DATA
figure()
x_accept, x_reject = xy_vals['x,y'][0][accept], xy_vals['x,y'][0][reject]
y_accept, y_reject = xy_vals['x,y'][1][accept], xy_vals['x,y'][1][reject]
c_norm = Normalize(vmin=np.min(p_prop.prob), vmax=np.max(p_prop.prob))
c_map = cm.jet(c_norm(p_prop.prob))
c_accept, c_reject = c_map[accept], c_map[reject]
scatter(x_accept, y_accept, color=c_accept, marker='.', linewidths=0.5)
scatter(x_reject, y_reject, color=c_reject, marker='x', linewidths=0.5)
xlabel('x')
ylabel('y')
title('Est. circle area={}'.format(circle_area))
