from __future__ import annotations

import chex
import jax.numpy as jnp

from spydr.bayesian_optimization.loop import loop
from spydr.bayesian_optimization.acquisition import (
    expected_improvement_by_model,
)
from spydr.bayesian_optimization.model import Env, map_model
from spydr.bayesian_optimization.optimizer import multistart_bfgs
from spydr.data import Dataset
from spydr.model.gaussian_process import GaussianProcess, ConjugateGPRegression, fit, predict_latent
from spydr.model.kernel import rbf
from spydr.model.mean_function import zero
from spydr.objectives import unit_branin
from spydr.optimize import bfgs
from spydr.types.stream import take

# chex.disable_asserts()
x = y = jnp.linspace(-2, 2, 5)
xx, yy = jnp.meshgrid(x, y)
query_points = jnp.vstack((xx.flatten(), yy.flatten())).T
data = Dataset(query_points, unit_branin(query_points))
chex.assert_shape(query_points, [25, 2])


feature_shape = [2]


def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]), feature_shape=feature_shape, target_shape=[])


gpr = fit(ConjugateGPRegression(
    Dataset.empty(feature_shape, [1]), gp, jnp.array([1.0]), jnp.array(0.2)
), bfgs, data)

acquisition = (
    expected_improvement_by_model()
    .contramap(map_model(predict_latent))
    .map(multistart_bfgs(query_points[0], query_points[-1]))
)


def observer(points: jnp.ndarray, env: Env[ConjugateGPRegression]) -> Env[ConjugateGPRegression]:
    new_data = Dataset(points, unit_branin(points))
    return Env(env.data.op(new_data), fit(env.model, bfgs, new_data))


points = loop(acquisition, observer, Env(data, gpr))
y = take(10, points)[-1].data.features
print(y)
