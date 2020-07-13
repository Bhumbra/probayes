""" Example of ordinary Monte Carlo integration with rejection sampling """
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Qt5Agg")
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

xy_range = {-radius, radius}
x = prob.RV("x", xy_range)
y = prob.RV("y", xy_range)
o = prob.RV("o")

# DEFINE STOCHASTIC CONDITION
xy = prob.SJ(x, y)
oxy = prob.SC(o, xy)
oxy.set_prob(inside)

# DEFINE PROPOSAL DENSITY AND COEFFICIENT VARIABLE
prop_xy = prob.SJ(x, y)
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
expt = np.mean(accept)
square_area = 4. * radius**2
circle_area = square_area * expt

# PLOT DATA
"""
figure()
true = np.nonzero(omc_true.prob)[0]
false = np.nonzero(np.logical_not(omc_true.prob))[0]
x, y = omc_true.vals['x'], omc_true.vals['y']
plot(x[true], y[true], 'b.')
plot(x[false], y[false], 'r.')
xlabel('x')
ylabel('y')
title('Est. circle area={}'.format(circle_area))
"""
