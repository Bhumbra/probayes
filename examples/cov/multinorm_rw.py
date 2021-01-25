# Example of a 2D multivariate random walk
import scipy.stats
from pylab import *; ion()
import probayes as pb

set_lims = (-10., 10.)
nsteps = 2000
means = [0.5, -0.5]
covar = [[1.5, -1.0], [-1.0, 2.]]

x = pb.RV('x', vtype=float, vset=set_lims)
y = pb.RV('y', vtype=float, vset=set_lims)
xy = x & y
xy.set_prob(scipy.stats.multivariate_normal, means, covar)
xy.set_tran(scipy.stats.multivariate_normal, means, covar)
x_t = np.empty(nsteps, dtype=float)
y_t = np.empty(nsteps, dtype=float)
p_t = np.empty(nsteps, dtype=float)
for i in range(nsteps):
  if i == 0:
    cond = xy.step({'x':0., 'y':0.}, {0})
  else:
    cond = xy.step({'x': x_t[i-1], 'y': y_t[i-1]}, {0})
  x_t[i], y_t[i] = cond.vals["x'"], cond.vals["y'"]
  p_xy = xy({'x': x_t[i], 'y': y_t[i]})
  p_t[i] = p_xy.prob

xy_t = np.array([x_t, y_t])
amn_xy = np.mean(xy_t, axis=1)
cov_xy = np.cov(xy_t)

figure()
c_norm = Normalize(vmin=np.min(p_t), vmax=np.max(p_t))
c_map = cm.jet(c_norm(p_t))
plot(x_t, y_t, '-', color=(0.7, 0.7, 0.7, 0.3))
scatter(x_t, y_t, color=c_map, marker='.')
xlabel(r'$x$')
ylabel(r'$y$')
