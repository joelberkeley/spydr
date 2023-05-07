from __future__ import annotations

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
from spydr.optimize import bfgs
from spydr.types.stream import take


def objective(x: jnp.ndarray) -> jnp.ndarray:
    return (x - 1) ** 2


query_points = jnp.array(
    [[-1.0], [-0.9], [-0.6], [0.0], [0.1], [0.5], [0.9], [1.1], [1.2], [1.4], [1.7], [2.0], [2.3]]
)
data = Dataset(query_points, objective(query_points))


def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]))


gpr = fit(ConjugateGPRegression(
    Dataset.empty([1], [1]), gp, jnp.array([1.0]), jnp.array(0.2)
), bfgs, data)

acquisition = (
    expected_improvement_by_model()
    .contramap(map_model(predict_latent))
    .map(multistart_bfgs(query_points[0], query_points[-1]))
)


def observer(points: jnp.ndarray, env: Env[ConjugateGPRegression]) -> Env[ConjugateGPRegression]:
    new_data = Dataset(points, objective(points))
    return Env(env.data.op(new_data), fit(env.model, bfgs, new_data))


points = loop(acquisition, observer, Env(data, gpr))
y = take(10, points)[-1].data.features
print(y)
