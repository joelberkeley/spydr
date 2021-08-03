from jax import numpy as np
from jax.scipy.optimize import minimize

from spydr.bayesian_optimization import unary
from spydr.bayesian_optimization.acquisition import negative_lower_confidence_bound, use
from spydr.distribution import Gaussian
from spydr.model.gaussian_process import GaussianProcess, fit, marginalise
from spydr.model.kernel import rbf
from spydr.model.mean_function import zero
from spydr.optimize import Optimizer

from spydr.util import assert_shape


def bfgs(initial_guess: np.ndarray) -> Optimizer[np.ndarray]:
    def optimizer(f):
        res = minimize(f, initial_guess, method="BFGS")

        if not res.success:
            raise ValueError

        return res.x

    return optimizer


query_points = np.array([[-1.0], [-0.6], [0.1], [0.5], [0.9]])
observations = query_points ** 2 - 0.5


def prior(len_and_noise: np.ndarray) -> GaussianProcess:
    assert_shape(len_and_noise, [2])
    return GaussianProcess(zero, rbf(len_and_noise[0]))


def likelihood(len_and_noise: np.ndarray) -> Gaussian:
    assert_shape(len_and_noise, [2])
    return Gaussian(
        np.zeros_like(observations),
        len_and_noise[1] * np.ones_like(observations)
    )


posterior = fit(bfgs(np.array([0.3, 0.1])), prior, likelihood, (query_points, np.squeeze(observations)))


def model(x: np.ndarray) -> Gaussian:
    assert_shape(x, [None, 1])
    marginal = marginalise(posterior, x)
    return Gaussian(
        assert_shape(marginal.mean[None], (len(x), 1)),
        assert_shape(marginal.cov[None], (len(x), len(x), 1))
    )


neg_lcb = use(lambda x: x, negative_lower_confidence_bound(np.array(1.0)))
acquisition = unary.map(lambda f: bfgs(np.array([-0.05]))(lambda x: f(x[None])), neg_lcb)
print(acquisition(((query_points, observations), model)))
