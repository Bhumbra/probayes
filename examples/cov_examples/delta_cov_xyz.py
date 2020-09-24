""" Simple example program using deltas in 3 dimensions """
import probayes as pb
import numpy as np
set_lims = [-0.5, 0.5]
num_deltas = 20000
delta = [0.1]

x = pb.RV('x', set_lims, vtype=float)
y = pb.RV('y', set_lims, vtype=float)
z = pb.RV('z', set_lims, vtype=float)

xyz = x * y * z
cov_mat = np.array([1., .5, 0.,
                    .5, 1., .5,
                    0., .5, 1.]).reshape([3,3])

xyz.set_tran(cov_mat)
xyz.set_delta(delta)
deltas = [xyz.eval_delta() for _ in range(num_deltas)]
dxdydz = np.array([np.ravel(np.array(list(_delta))) \
                   for _delta in deltas])
means = np.mean(dxdydz, axis=0)
stdvs = np.std(dxdydz, axis=0)
lengths = np.sqrt(np.sum(dxdydz**2, axis=1))
mean_lengths = np.mean(lengths)
covar = np.cov(dxdydz.T)
print(3.*covar/(delta[0]**2))
