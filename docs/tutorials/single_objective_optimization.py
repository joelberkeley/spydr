# %load_ext autoreload
# %autoreload 2

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from spydr.bayesian_optimization.loop import loop
from spydr.bayesian_optimization.acquisition import (
    expected_improvement_by_model, acquisition_shapes,
)
from spydr.bayesian_optimization.model import Env, map_model
from spydr.bayesian_optimization.optimizer import multistart_bfgs
from spydr.data import Dataset
from spydr.model.gaussian_process import GaussianProcess, ConjugateGPRegression, fit, predict_latent
from spydr.model.kernel import rbf, matern52
from spydr.model.mean_function import zero
from spydr.objectives import unit_branin
from spydr.optimize import bfgs
from spydr.types.stream import take

jax.config.update("jax_enable_x64", True)
# chex.disable_asserts()
x = y = jnp.linspace(0, 1, 5)
xx, yy = jnp.meshgrid(x, y)
query_points = jnp.vstack((xx.flatten(), yy.flatten())).T

query_points = jnp.array([
    [0.2, 0.3],
    [0.8, 0.7],
    [0.5, 0.8],
    [0.4, 0.2],
    [0.7, 0.2],
], dtype=jnp.float64)
data = Dataset(query_points, unit_branin(query_points))
chex.assert_shape(query_points, [None, 2])


feature_shape = [2]
query_points


def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, matern52(params[0], params[1]), feature_shape=feature_shape)


# problem is that length scale is being fit to 12705

gpr = fit(ConjugateGPRegression(
    Dataset.empty(feature_shape, [1]),
    gp,
    jnp.array([0.5, 0.4], dtype=jnp.float64),
    jnp.array(0.1, dtype=jnp.float64)
), bfgs, data)
print(gpr.gp_params)


def trace(x, msg=""):
    print(msg, x)
    return x


acquisition = (
    expected_improvement_by_model()
    # def don't want to write this like this
    # .map(lambda acq: acquisition_shapes(acq, batch_size=1, feature_shape=[2]))
    .contramap(map_model(predict_latent))
    .map(lambda f: trace(multistart_bfgs(jnp.array([0, 0.0]), jnp.array([1, 1.0]))(f), "acquisition optimizer"))
)


def observer(points: jnp.ndarray, env: Env[ConjugateGPRegression]) -> Env[ConjugateGPRegression]:
    new_data = Dataset(points, unit_branin(points))
    return Env(env.data.op(new_data), fit(env.model, lambda x: lambda f: trace(bfgs(x)(f), "gp optimizer"), new_data))


# +
points = loop(acquisition, observer, Env(data, gpr))
y = take(10, points)[-1].data.targets
print(y)

# we're getting nan because we're using the same multistart points each time and we're using the same jitter each time, so we get 0 
# -




