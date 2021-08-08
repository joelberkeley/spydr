from jax import numpy as np

from spydr.bayesian_optimization import unary
from spydr.bayesian_optimization.acquisition import negative_lower_confidence_bound, use_pair, Acquisition
from spydr.model.gaussian_process import GaussianProcess, ConjugateGPRegression
from spydr.model.kernel import rbf, matern52
from spydr.model.mean_function import zero
from spydr.optimize import bfgs


query_points = np.array([[-1.0], [-0.6], [0.1], [0.5], [0.9]])
data = (query_points, (query_points - 1) ** 2 - 0.5)


def mk_gp(params: np.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, matern52(params[0], params[1]))


gpr = ConjugateGPRegression(mk_gp, np.array([1.0, 0.3]), np.array(0.2)).fit(bfgs, data)


def acquisition_optimizer(f: Acquisition) -> np.ndarray:
    return bfgs(np.array([-0.5]))(lambda x: f(x[None]))


neg_lcb = use_pair(lambda x: x, negative_lower_confidence_bound(np.array(1.0)))
acquisition = unary.map(acquisition_optimizer, neg_lcb)

print(acquisition((data, gpr)))
