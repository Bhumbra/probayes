# Overkill program to test 1-D sampling
import prob
x = prob.RV('x', vtype=float)
c = prob.SC(x)
samples = c.sample({-10})
