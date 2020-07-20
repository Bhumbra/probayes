""" Example of ordinary Monte Carlo integration with uniform sampling """
import numpy as np
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
xy = x * y
oxy = o / xy
oxy.set_prob(inside)

# CALL PROBABILITIES AND EVALUATE EXPECTATION AND AREA
p_cond_omc = oxy({'x,y': set_size})
p_omc_true = p_cond_omc({'o': True})
xy_vals = p_cond_omc.ret_cond_vals()
p_marg_omc = xy(xy_vals)
p_joint_omc = p_marg_omc * p_cond_omc
joint_expt = p_joint_omc.expectation()
square_area = 4. * radius**2
circle_area = square_area * joint_expt['o']

# PLOT DATA FROM CONDITIONAL DISTRIBUTION
figure()
true = np.nonzero(p_omc_true.prob)[0]
false = np.nonzero(np.logical_not(p_omc_true.prob))[0]
x_all, y_all = p_omc_true.vals['x'], p_omc_true.vals['y']
plot(x_all[true], y_all[true], 'b.')
plot(x_all[false], y_all[false], 'r.')
xlabel('x')
ylabel('y')
title('Est. circle area={}'.format(circle_area))
