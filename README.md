# probayes
Probability package supporting multiple Bayesian methods including MCMC

Unlike existing libraries, probayes adopts a model-driven approach with full flexibility constrained only by the rules
of probability. Since probayes is its infancy and in a state of flux, there is no manual. Currently probayes supports
the following:

1. Multiple random variable sampling in untransformed and transformed domain space. 
2. Transitional simulation, including random walks, using Markov chain conditionals.
3. Discrete grid exact inference.
4. Ordinary Monte Carlo random sampling.
5. Ordinary Monto Carlo rejection sampling.
6. Metropolis-Hastings MCMC sampling.
7. Limited support for multivariate normal-covariance Gibbs sampling.

In the near-future, it is intended to expand the scope of probayes to include:

1. Support Gibbs sampling using semi-conjugacy.
2. Code initial support for approximate inference using using dense mean field messaging.
3. Support derivative-based updates (HMC, gradient ascent/descent optimisation).

A quickstart is also intended, but for now there are examples in the examples/ subdirectories:

1. tests/        Simple test scripts
2. rv_examples/  Random variable examples
3. markov/       Markov chain examples
4. cov_examples/ Examples of using covariance matrices
5. dgei/         Discrete grid exact inference examples
6. omc/          Ordinary Monte-Carlo examples
7. mcmc/         Markov chain Monte Carlo examples (Metropolis-Hastings, Gibbs...)
