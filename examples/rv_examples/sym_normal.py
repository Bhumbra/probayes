""" Example of normally distributed variable specified using sympy.stats """
import sympy
import sympy.stats
import probayes as pb
from pylab import *; ion()

x = pb.RV('x', vtype=float, vset=[-2, 2])
x.set_prob(sympy.stats.Normal(x[:], mean=0, std=1), pscale='log')
fx = x({1000}).rescaled()
rx = x({-1000})

figure()
subplot(2, 1, 1)
plot(fx['x'], fx.prob)
xlabel('x')
ylabel('Prob / density')
gca().set_ylim(0., 1.02*max(fx.prob))
title("{}".format(x.prob))
subplot(2, 1, 2)
hist(rx['x'], 50)
xlabel('x (random samples)')
ylabel('Freq / counts')
