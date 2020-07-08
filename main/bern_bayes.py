import prob
import numpy as np
"""
Consider a disease with a prevalence 1\% in a given population. 
Of those with the disease, 98\% manifest a particular symptom,
that is present only in 10\% of those without the disease. 
What is the probability someone with symptoms has the disease?
Answer: \approx 9%
"""

# PARAMETERS
prevalence = 0.01
sym_if_dis = 0.98
sym_if_undis = 0.1

# SET UP RANDOM VARIABLES
dis = prob.RV('dis', prob=prevalence)
sym = prob.RV('sym')

# SET UP STOCHASTIC CONDITION
sym_given_dis = prob.SC(sym, dis)
sym_given_dis.set_prob(np.array([1-sym_if_undis, 1-sym_if_dis, \
                                 sym_if_undis,   sym_if_dis]).reshape((2,2)))

# CALL PROBABILITIES
prior = dis()
likelihood = sym_given_dis()
prior_x_likelihood = prior * likelihood
evidence = prior_x_likelihood.marginal('sym')
posterior = prior_x_likelihood / evidence
inference = posterior({'dis': True, 'sym': True})
print(inference)
