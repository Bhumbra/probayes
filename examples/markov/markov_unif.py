# Example (fairly boring one) of a Markov chain sampling from a uniform rv
# defined over a logarithmic scale with no transitional dependencies.
# This example also includes a (fairly redundant) reverse step.

import probayes as pb
import numpy as np

set_lims = [np.e, np.e**3] # returns a scalar probability of 0.5
set_sizes = [{0}, {-5}]

x = pb.RV('x', set_lims)
x.set_ufun((np.log, np.exp))
x_x = x.step(set_sizes[0], set_sizes[1])
xx_ = x_x.rekey({"x'": 'x', 'x': "x'"})
_xx = x.step(xx_['x'], xx_["x'"], reverse=True)
print((x_x, x_x.prob))
print((_xx, _xx.prob))
