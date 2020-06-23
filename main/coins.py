# Example of a joint PMF for two coins
import prob
heads_0 = prob.RV('coin_0', prob=0.7)
heads_1 = prob.RV('coin_1', prob=0.4)
Heads = prob.JC(heads_0, heads_1)
print(Heads())
