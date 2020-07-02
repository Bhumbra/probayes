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
symptoms_if_diseased = 0.98
symptoms_if_undiseased = 0.1

# SET UP RANDOM VARIABLES
diseased = prob.RV('diseased', prob=prevalence)
symptoms = prob.RV('symptoms', vtype=bool)

# SET UP CONDITIONAL PROBABILITY AND STOCHASTIC CONDITION
conditional_prob = np.array([1-symptoms_if_undiseased, symptoms_if_undiseased, \
                             1-symptoms_if_diseased,   symptoms_if_diseased]).reshape((2,2))
symptoms_given_disease = prob.SC(symptoms, diseased)
symptoms_given_disease.set_prob(conditional_prob)

# CALL THE PROBABILITIES
p_diseased = diseased()
p_symptoms_given_diseased = symptoms_given_disease()
p_symptoms_and_diseased = p_diseased * p_symptoms_given_diseased
p_diseased_given_symptoms = p_symptoms_and_diseased.conditionalise('symptoms')
p_diseased_true_given_symptom_true = p_diseased_given_symptoms({'diseased': True, 'symptoms': True})
print(p_diseased_true_given_symptom_true)
