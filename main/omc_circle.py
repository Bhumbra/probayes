""" Example of ordinary Monte Carlo integration """
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import prob

radius = 1.
def inside(o, x, y): 
  return o == (x**2 + y**2 <= radius**2)
set_size = {-10000}
xy_range = {-radius, radius}
x = prob.RV("x", xy_range)
y = prob.RV("y", xy_range)
xy = prob.SJ(x, y)
o = prob.RV("o")
oxy = prob.SC(o, xy)
oxy.set_prob(inside)
cond_omc = oxy({'x,y': set_size})
omc_true = cond_omc({'o': True})
xy_vals = cond_omc.ret_cond_vals()
marg_omc = xy(xy_vals)
joint_omc = marg_omc * cond_omc
joint_expt = joint_omc.expectation()
square_area = 4. * radius**2
circle_area = square_area * joint_expt['o']
true = np.nonzero(omc_true.prob)[0]
false = np.nonzero(np.logical_not(omc_true.prob))[0]
x, y = omc_true.vals['x'], omc_true.vals['y']
figure()
plot(x[true], y[true], 'b.')
plot(x[false], y[false], 'r.')
xlabel('x')
ylabel('y')
title('Area={}'.format(circle_area))
