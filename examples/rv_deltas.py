import prob
import numpy as np
set_lims = [-0.5, 0.5]
num_deltas = 1000
delta_spec = 0.1

x = prob.RV('x', set_lims, vtype=float)
y = prob.RV('y', set_lims, vtype=float)
z = prob.RV('z', set_lims, vtype=float)

xyz = x * y * z
delta = xyz.Delta(delta_spec)
deltas = [xyz.eval_delta(delta) for _ in range(num_deltas)]
dxdydz = np.array([np.array(list(_delta)) for _delta in deltas])
means = np.mean(dxdydz, axis=0)
stdvs = np.std(dxdydz, axis=0)
lengths = np.sqrt(np.sum(dxdydz**2, axis=1))
