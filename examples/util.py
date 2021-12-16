import json
import numpy as np
from scipy.stats import multivariate_normal, uniform


class NumpyEncoder(json.JSONEncoder):
    """Ensures json.dumps doesn't crash on numpy types
    See: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class EnvParamDist():
    """Environment parameter p is a k-dimensional random variable within a given range.
    
    """
    def __init__(self, param_start=[0], param_end=[10], dist_type='gaussian'):
        self.start = np.array(param_start)
        self.end = np.array(param_end)
        if dist_type == 'gaussian':
            mu = (self.start + self.end) / 2
            sigma = (mu - self.start) / 3
            cov = np.diag(sigma)**2
            self.param_dist = multivariate_normal(mean=mu, cov=cov)
        elif dist_type == 'uniform':
            self.param_dist = uniform(loc=self.start, scale=self.end-self.start)
        else:
            raise NotImplementedError

    def sample(self, size=(1, 2)):
        # size = num x k
        tmp = self.param_dist.rvs(size=size)
        min_param = self.start.reshape(1, -1).repeat(size[0], axis=0)
        max_param = self.end.reshape(1, -1).repeat(size[0], axis=0)
        return np.clip(tmp, min_param, max_param)

    def integral(self, left, right):
        return self.param_dist.cdf(right) - self.param_dist.cdf(left)
