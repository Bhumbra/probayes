# Sampling program to test simple 1-D sampling
import prob
x = prob.RV('x', vtype=float)
c = prob.SP(x)
sampler = c.sampler(10)
samples = [sample for sample in sampler]
print("Found {} samples".format(len(samples)))
