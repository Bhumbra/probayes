""" Simple example program using deltas in 3 dimensions """
import probayes as pb
import numpy as np
set_lims = [-0.5, 0.5]
num_deltas = 20000
delta = (0.1,)

x = pb.RV('x', vtype=float, vset=set_lims)
y = pb.RV('y', vtype=float, vset=set_lims)
z = pb.RV('z', vtype=float, vset=set_lims)

xyz = x & y & z
xyz.set_delta(delta, scale=False)
deltas = [xyz.eval_delta() for _ in range(num_deltas)]
dxdydz = np.array([np.array(list(_delta)) for _delta in deltas])
means = np.mean(dxdydz, axis=0)
stdvs = np.std(dxdydz, axis=0)
lengths = np.sqrt(np.sum(dxdydz**2, axis=1)) 
print(lengths) # should all be 0.1 if scale is False
