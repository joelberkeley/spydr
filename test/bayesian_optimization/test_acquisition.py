import jax
import jax.numpy as jnp

from spydr.bayesian_optimization.acquisition import expected_improvement_by_model
from spydr.bayesian_optimization.model import Env
from spydr.data import Dataset
from spydr.distribution import Gaussian
from spydr.model.gaussian_process import ConjugateGPRegression, GaussianProcess, fit, predict_latent
from spydr.model.kernel import rbf
import numpy.testing as jnpt

from spydr.model.mean_function import zero
from spydr.optimize import bfgs


def test_expected_improvement_by_model() -> None:
    def y(x_: jnp.ndarray) -> jnp.ndarray:
        return x_ ** 2

    def model(x_: jnp.ndarray) -> Gaussian:
        return Gaussian(jnp.array(y(x_)), jnp.ones_like(x_)[..., None])

    x = jnp.arange(20.0)[:, None] / 10 - 1
    ei = expected_improvement_by_model().run(Env(Dataset(x, y(x)), model))

    jnpt.assert_equal(x[jnp.argmax(jax.vmap(ei)(x[..., None]), 0)], jnp.array([0.0]))
    jnpt.assert_array_less(ei(jnp.array([[-0.5]])), ei(jnp.array([[0.0]])))
    jnpt.assert_array_less(ei(jnp.array([[1.0]])), ei(jnp.array([[0.5]])))
