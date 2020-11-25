""" Example of ordinary Monte Carlo random walk sampling to integrate circle area """
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
import probayes as pb

# PARAMETERS
radius = 1.
n_steps = 6000
cols = {False: 'r', True: 'b'}

# SETUP CIRCLE FUNCTION AND RVs
def inside(x, y): 
  return x**2 + y**2 <= radius**2

xy_range = (-radius, radius)
x = pb.RV("x", xy_range)
y = pb.RV("y", xy_range)

# DEFINE RANDOM FIELD
xy = x & y
xy.set_delta((0.15*radius,), bound=True)
xy.set_prop(inside, order={'x': None, 'y': None, "x'": 0, "y'": 1})
steps = [None] * n_steps
pred = [None] * n_steps
succ = [None] * n_steps
cond = np.empty(n_steps, dtype=float)

print('Simulating...')
for i in range(n_steps):
  if i == 0:
    steps[i] = xy.step({0})
  else:
    steps[i] = xy.step(succ[i-1])
  x_pred, y_pred = steps[i]["x"], steps[i]["y"]
  x_succ, y_succ = steps[i]["x'"], steps[i]["y'"]
  pred[i] = {'x': x_pred, 'y': y_pred}
  succ[i] = {'x': x_succ, 'y': y_succ}
  cond[i] = inside(x_pred, y_pred)
print('...done')
xy_pred = np.array([np.array([xy['x'], xy['y']]) for xy in pred])
xy_succ = np.array([np.array([xy['x'], xy['y']]) for xy in succ])
square_area = 4. * radius**2
circle_area = square_area * np.mean(cond)

# PLOT DATA FROM CONDITIONAL DISTRIBUTION
figure()
state = None
for i in range(n_steps):
  x_pred, y_pred = xy_pred[i, 0], xy_pred[i, 1]
  x_succ, y_succ = xy_succ[i, 0], xy_succ[i, 1]
  current_state = inside(x_pred, y_pred)
  plot_xy = i < n_steps - 1
  if state is None:
    xx = [x_pred]
    yy = [y_pred]
    state = current_state
  if state == current_state:
    xx.append(x_succ)
    yy.append(y_succ)
  if state != current_state:
    plot_xy = True
  if plot_xy:
    plot(xx, yy, cols[state]+'-')
    xx = [xx[-1]]
    yy = [yy[-1]]
    state = current_state
xlabel('x')
ylabel('y')
title('Est. circle area={}'.format(circle_area))
