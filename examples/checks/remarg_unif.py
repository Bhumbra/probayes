# Remarginalisation example

import collections
import probayes as pb
import numpy as np
from pylab import *; ion()

num_samples = 500
num_resamples = 50

x = pb.RV('x', vset=[0, 1], vtype=float)
y = pb.RV('y', vset=[0, 1], vtype=float)
xy = x & y
p_xy = xy({num_samples})

xpy = np.linspace(-0.001, 2.001, num_resamples)
xmy = np.linspace(-1.001, 1.001, num_resamples)
distribution_vals = collections.OrderedDict({'p': xpy, 'm': xmy})
distribution = pb.Distribution('p,m', distribution_vals)
mapping = {'p': p_xy['x'] + p_xy['y'],
           'm': p_xy['x'] - p_xy['y']}
p_pm = p_xy.remarginalise(distribution, mapping)
pmf = p_pm.prob[:-1, :-1]

figure()
pcolor(np.ravel(p_pm['m']), 
       np.ravel(p_pm['p']), 
       pmf, cmap=cm.jet)
colorbar()



