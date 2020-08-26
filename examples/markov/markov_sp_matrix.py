""" Example of a Discrete Markov Chain random walk conditioned by a
transition matrix. What is the probability after n steps that the
Markov train goes from state 0 to state 0? This simulation is based
on Example 1.1.6 from J.R. Norris (1997): Markov chains. CUP. p.6-8.
"""
import probayes as pb
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

# Prob convention: successor dimension (row) > predecessor dimension (col)
tran = np.array(
                [0., 0., .5,
                 1., .5, 0.,
                 0., .5, .5],
               ).reshape([3,3])
n_sims = 2000
m_steps = 12 # max_steps

# Analytical solution (obtained from the eigenvalues of tran)
m = np.arange(1, m_steps+1)
mpi_2 = m * np.pi / 2
hatp = 0.2 + 0.5**m * (0.8 * np.cos(mpi_2) - 0.4 * np.sin(mpi_2))

# Simulation
x = pb.RV('x', range(3))
x.set_tran(tran)
X = pb.SP(x)

cond = [None] * n_sims
succ = np.empty([n_sims, m_steps], dtype=int)
print('Simulating...')
for i in range(n_sims):
  cond[i] = [None] * m_steps
  sampler = X.sampler({'x': 0}, stop=m_steps)
  samples = X.walk(sampler)
  summary = X(samples)
  cond[i] = summary.q.prob
  succ[i] = summary.q.vals["x'"]
print('...done')
obsp = np.sum(succ==0, axis=0) / n_sims

# Plot
figure()
plot(m, hatp, 'r', label='Expected probability')
plot(m, obsp, 'b', label='Simulated proportion')
xlabel(r'$n$')
ylabel(r'$P$')
legend()
