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

xy_range = {-radius, radius}
x = prob.RV("x", xy_range)
y = prob.RV("y", xy_range)
o = prob.RV("o")

# DEFINE STOCHASTIC CONDITION
xy = prob.SJ(x, y)
oxy = prob.SC(o, xy)
def norm2d(x, y, loc=0., scale=radius):
  return scipy.stats.norm.pdf(x, loc=loc, scale=scale) * \
         scipy.stats.norm.pdf(y, loc=loc, scale=scale)
oxy.set_prop(norm2d)
oxy.set_prob(inside)

# CALL PROBABILITIES AND EVALUATE EXPECTATION AND AREA
prop_omc = oxy.propose({'x,y': set_size})
xy_vals = prop_omc.ret_marg_vals()
cond_omc = oxy(xy_vals)
"""
omc_true = cond_omc({'o': True})
marg_omc = xy(xy_vals)
joint_omc = marg_omc * cond_omc
joint_expt = joint_omc.expectation()
square_area = 4. * radius**2
circle_area = square_area * joint_expt['o']
"""

# PLOT DATA FROM CONDITIONAL DISTRIBUTION
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
