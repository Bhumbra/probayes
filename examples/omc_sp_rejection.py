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
steps = 10000

# SETUP CIRCLE FUNCTION AND RVs
def inside(x, y): 
  return np.array(x**2 + y**2 <= radius**2, dtype=float)

def norm2d(x, y, loc=0., scale=radius):
  return scipy.stats.norm.pdf(x, loc=loc, scale=scale) * \
         scipy.stats.norm.pdf(y, loc=loc, scale=scale)

xy_range = [-radius, radius]
x = prob.RV("x", xy_range)
y = prob.RV("y", xy_range)

# DEFINE STOCHASTIC PROCESS
xy = x * y
process = prob.SP(xy)
process.set_prob(inside)

# DEFINE PROPOSAL DENSITY AND COEFFICIENT VARIABLE
process.set_prop(norm2d)
process.set_scores(lambda opqr: opqr.p.prob)
coef_max = float(norm2d(radius, 1.))
process.set_thresh(np.random.uniform, low=0., high=coef_max)
process.set_update(lambda stu: stu.s >= stu.t)

# SAMPLE AND SUMMARISE
sampler = process.sampler({0}, stop=steps)
samples = [sample for sample in sampler]
summary = process(samples)
expectation = summary.p.size / steps
square_area = 4. * radius**2
circle_area = square_area * expectation

# PLOT DATA
figure()
xy_vals = np.array([(sample.p.vals['x'], sample.p.vals['y']) \
                    for sample in samples])
p_prop = np.array([sample.q.prob for sample in samples])
accept = np.array([sample.u for sample in samples])
reject = np.logical_not(accept)
x_accept, x_reject = xy_vals[accept, 0], xy_vals[reject, 0]
y_accept, y_reject = xy_vals[accept, 1], xy_vals[reject, 1]
c_norm = Normalize(vmin=np.min(p_prop), vmax=np.max(p_prop))
c_map = cm.jet(c_norm(p_prop))
c_accept, c_reject = c_map[accept], c_map[reject]
scatter(x_accept, y_accept, color=c_accept, marker='.', linewidths=0.5)
scatter(x_reject, y_reject, color=c_reject, marker='x', linewidths=0.5)
xlabel('x')
ylabel('y')
title('Est. circle area={}'.format(circle_area))
