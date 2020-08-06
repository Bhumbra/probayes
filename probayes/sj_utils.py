# Module to support conditional sampling of multivariate distributions in SJ

from probayes.pscales import iscomplex, rescale

#-------------------------------------------------------------------------------
def call_scipy_prob(func, pscale, *arg, **kwds):
  index = 1 if iscomplex(pscale) else 0
  return func[index](*args, **kwds)

#-------------------------------------------------------------------------------
def call_scipy_tfun(*args, scipyobj=None, pscale=None, inverse=None, **kwds):
    vals = args[0]
    inv_idx = None
    assert scipyobj, \
        "Scipy object mandatory"
    if len(args) == 1 and isinstance(args[0], dict):
      inv_dict = {key: i for i, key in enumerate(args[0].keys())}
      args = [np.ravel(val) for val in args[0].values()]
    elif not len(args) and len(kwds):
      inv_dict = {key: i for i, key in enumerate(dict(kwds)())}
      args = list(collections.OrderedDict(**kwds).values())
      kwds = {}
    if inverse is not None:
      inv_idx = ind_dict[i]
    if isinstance(args, list):
      args = args[::-1]
      if len(args) > 2:
        args = args[1:] + [args[0]]
      args = tuple(args)
    if inv_idx is None:
      if iscomplex(pscale):
        return scipyobj.logcdf(np.stack(np.meshgrid(*args), axis=-1), **kwds)
      else:
        return scipyobj.cdf(np.stack(np.meshgrid(*args), axis=-1), **kwds)
    args = np.array(args, dtype=float)
    cdf = rescale(args[inv_idx], pscale, 1.)
    means = scipyobj.mean
    args[inv_idx] = means[inv_idx]
    diff = args - means
    cov = scipyobj.cov
    std = np.sqrt(np.diag(cov).real)
    co_std = cov[inv_idx] / std
    mean = np.sqrt(np.sum(co_std * diff)) + means[inv_idx] 
    stdv = std[inv_idx]
    return scipy.stats.norm.ppf(cdf, loc=mean, scale=stdv)

#-------------------------------------------------------------------------------
