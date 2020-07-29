# Sampling program to test simple 1-D sampling
import prob
num_samples = 10

x = prob.RV('x', vtype=float)
p = prob.SP(x)

# Method one
sampler_0 = p.sampler()
samples = []
while len(samples) < num_samples:
  samples.append(next(sampler_0))
print("Number of samples: {}".format(len(samples)))

# Method two
sampler_1 = p.sampler(num_samples)
samples = [sample for sample in sampler_1]
print("Number of samples: {}".format(len(samples)))
