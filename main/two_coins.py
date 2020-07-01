# Example of a joint PMF for two coins
import prob
h0 = prob.RV('c0', prob=0.7)
h1 = prob.RV('c1', prob=0.4)
hh = prob.SJ(h0, h1)
HH = hh()
M0 = HH.marginalise('c1')
M1 = HH.marginalise('c0')
#C0 = HH.conditionalise('c0')
C1 = HH.conditionalise('c1')
print((HH, HH.vals, HH.prob))
print((C1, C1.vals, C1.prob))
"""
print((M1, M1.vals, M1.prob))
print((C0, C0.vals, C0.prob))
print((C1, C1.vals, C1.prob))
"""
