import jax.numpy as jnp

from spydr.data import Dataset
from spydr.model.gaussian_process import ConjugateGPRegression, GaussianProcess, fit, predict_latent
from spydr.model.kernel import rbf, matern52
import numpy.testing as jnpt

from spydr.model.mean_function import zero
from spydr.optimize import bfgs


def test_gp() -> None:
    def gp(params: jnp.ndarray) -> GaussianProcess:
        return GaussianProcess(
            zero, matern52(params[0], params[1]), feature_shape=[1], target_shape=[]
        )

    def y(x_: jnp.ndarray) -> jnp.ndarray:
        return x_ ** 2 + jnp.sin(x_)

    def mse(gpr_: ConjugateGPRegression, x_: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(predict_latent(gpr_)(x_).mean - y(x_)) ** 2

    gpr = ConjugateGPRegression(Dataset.empty([1], [1]), gp, jnp.array([0.1, 0.1]), jnp.array(0.1))
    x_train = jnp.arange(20.0)[:, None] / 10 - 1
    gpr_trained = fit(gpr, bfgs, Dataset(x_train, y(x_train)))
    x_test = x_train - 0.3
    jnpt.assert_array_less(mse(gpr_trained, x_test), mse(gpr, x_test))
