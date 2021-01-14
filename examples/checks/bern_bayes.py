import probayes as pb
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
dis = pb.RV('dis', prob=prevalence)
sym = pb.RV('sym')

# SET UP STOCHASTIC CONDITION
sym_given_dis = sym | dis
sym_given_dis.set_prob(np.array([1-sym_if_undis, 1-sym_if_dis, \
                                 sym_if_undis,   sym_if_dis]).reshape((2,2)))

# APPLY BAYES' RULE
p_dis = dis()
p_sym_given_dis = sym_given_dis()
p_dis_and_sym = p_dis * p_sym_given_dis
p_sym = p_dis_and_sym.marginal('sym')
p_dis_given_sym = p_dis_and_sym / p_sym
inference = p_dis_given_sym({'dis': True, 'sym': True})
print(inference)
assert abs(inference.prob-0.09) < 0.01, \
    "Expected around 0.09 but evaluated {}".format(inference.prob)
