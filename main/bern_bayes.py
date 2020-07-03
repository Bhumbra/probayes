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
sym = prob.RV('sym', vtype=bool)

# SET UP STOCHASTIC CONDITION AND CONDITIONAL PROBABILITY
sym_given_dis = prob.SC(sym, dis)
conditional_prob = np.array([1-sym_if_undis, sym_if_undis, \
                             1-sym_if_dis,   sym_if_dis]).reshape((2,2))
sym_given_dis.set_prob(conditional_prob)

# CALL THE PROBABILITIES
p_dis = dis()
p_sym_given_dis = sym_given_dis()
p_sym_and_dis = p_dis * p_sym_given_dis
p_dis_given_sym = p_sym_and_dis.conditionalise('sym')
p_dis_true_given_symptom_true = p_dis_given_sym({'dis': True, 'sym': True})
print(p_dis_true_given_symptom_true)
