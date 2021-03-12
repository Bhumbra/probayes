# Change of variables example

import collections
import probayes as pb
import numpy as np
from pylab import *; ion()

num_samples = 500

x = pb.Variable('x', vtype=float, vset=[0, 1])
y = pb.Variable('y', vtype=float, vset=[0, 1])
s = pb.Variable('s', vtype=float)
d = pb.Variable('d', vtype=float)
xy = x & y
sd = s & d

sd_from_xy = sd | xy
sd_from_xy.func.add_func(s[:], x[:] + y[:])
sd_from_xy.func.add_func(d[:], x[:] - y[:])


