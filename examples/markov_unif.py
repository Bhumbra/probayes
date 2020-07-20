# Example (fairly boring one) of a Markov chain sampling from a uniform rv
# defined over a logarithmic scale with no transitional dependencies.
# This example also includes a (fairly redundant) reverse step.

import prob
import numpy as np

set_lims = (np.e, np.e**3) # returns a scalar probability of 0.5
set_sizes = [{-5}, {0}]

x = prob.RV('x', set_lims)
x.set_vfun((np.log, np.exp))
x_x = x.step(set_sizes[1], set_sizes[0])
xx_ = x_x.rekey({"x'": 'x', 'x': "x'"})
vals = xx_.ret_vals()
_xx = x.step(vals['x'], vals["x'"], reverse=True)
print((x_x.vals, x_x.prob))
print((_xx.vals, _xx.prob))
