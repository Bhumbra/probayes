""" 
Example of a RV with an implicit probability distribution defined according
to a functional transformation:

if:
    x' ~ uniform
    f(x) = arcsin(sqrt(x'))
    then x ~ arcsine distirbution
"""

import probayes as pb
import sympy
from pylab import *; ion()

x = pb.RV('x', vtype=float, vset = (0,1))
x.set_ufun(sympy.asin(sympy.sqrt(x[:])), no_ucov=True)
fx = x({200})
x.set_ufun(sympy.asin(sympy.sqrt(x[:])), no_ucov=False)
rx = x({-10000})

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




