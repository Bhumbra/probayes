# Remarginalisation example

import collections
import probayes as pb
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

num_samples = 500
num_resamples = 50

x = pb.RV('x', [0, 1], vtype=float)
y = pb.RV('y', [0, 1], vtype=float)
xy = x * y
p_xy = xy({num_samples})

xpy = np.linspace(-0.001, 2.001, num_resamples)
xmy = np.linspace(-1.001, 1.001, num_resamples)
manifold_vals = collections.OrderedDict({'p': xpy, 'm': xmy})
manifold = pb.Manifold(manifold_vals)
vals = p_xy.vals
mapping = {'p': vals['x'] + vals['y'],
           'm': vals['x'] - vals['y']}
p_pm = p_xy.remarginalise(manifold, mapping)
pmf = p_pm.prob[:-1, :-1]

figure()
pcolor(np.ravel(p_pm.vals['m']), 
       np.ravel(p_pm.vals['p']), 
       pmf, cmap=cm.jet)
colorbar()



