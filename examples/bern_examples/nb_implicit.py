"""
Examples of naive 'Bayes' implicit example where: 
p(z | x, y) = p(z | x) * p(z | y)
- assumes RHS conditionls are mutually independent
"""
import numpy as np
import probayes as pb

# Simulation settings
sim_size = 1000 # Number of observations to simulate

# Simulate data
x_obs = np.random.choice([False, True], size=sim_size)
y_obs = np.random.choice([False, True], size=sim_size)
z_prob = 0.1 + 0.2*x_obs + 0.4*y_obs
z_obs = np.array([np.random.choice([False, True],
                    p=[1-z_p, z_p]) for z_p in z_prob])
zx_obs = np.hstack([z_obs.reshape([sim_size, 1]),
                    x_obs.reshape([sim_size, 1])])
zy_obs = np.hstack([z_obs.reshape([sim_size, 1]),
                    y_obs.reshape([sim_size, 1])])
zx_lhood, zx_rfreq = pb.bool_perm_freq(zx_obs, ['z', 'x'])
zy_lhood, zy_rfreq = pb.bool_perm_freq(zy_obs, ['z', 'y'])
 
# Naive Bayes
x = pb.RV('x', vtype=bool)
y = pb.RV('y', vtype=bool)
z = pb.RV('z', vtype=bool)
zx = z / x
zy = z / y
zx.set_prob(zx_lhood, passdims=True)
zy.set_prob(zy_lhood, passdims=True)
zxy = pb.SD(zx, zy)
p_zxy = zxy()
p_zxy_false = zxy({'x': False, 'y': False})
p_zxy_true = zxy({'x': True, 'y': True})
