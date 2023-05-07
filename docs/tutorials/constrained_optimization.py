from collections.abc import Mapping
from operator import itemgetter

import jax.numpy as jnp

from spydr.bayesian_optimization.loop import loop
from spydr.bayesian_optimization.acquisition import (
    expected_constrained_improvement,
    probability_of_feasibility,
)
from spydr.bayesian_optimization.model import Env, map_model
from spydr.bayesian_optimization.optimizer import multistart_bfgs
from spydr.data import Dataset
from spydr.model.gaussian_process import ConjugateGPRegression, fit, GaussianProcess, predict_latent
from spydr.model.kernel import rbf
from spydr.model.mean_function import zero
from spydr.optimize import bfgs
from spydr.types.stream import take


def objective(x: jnp.ndarray) -> jnp.ndarray:
    return (x - 1) ** 2


def constraint(x: jnp.ndarray) -> jnp.ndarray:
    return (jnp.cos(x) * jnp.sin(x)) ** 2


query_points = jnp.array(
    [[-1.0], [-0.9], [-0.6], [0.0], [0.1], [0.5], [0.9], [1.1], [1.2], [1.4], [1.7], [2.0], [2.3]]
)
data = Dataset(query_points, objective(query_points))


def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]))


gpr = ConjugateGPRegression(
    Dataset(jnp.empty([0, 1]), jnp.empty([0, 1])), gp, jnp.array([1.0]), jnp.array(0.2)
)
gpr = fit(gpr, bfgs, data)


def observer_(
        points: jnp.ndarray, env: Mapping[str, Env],
) -> Mapping[str, Env]:
    new_obj_data = Dataset(points, objective(points))
    new_con_data = Dataset(points, constraint(points))
    return {
        OBJECTIVE: Env(
            env[OBJECTIVE].data.concat(new_obj_data),
            fit(env[OBJECTIVE].model, bfgs, new_obj_data)),
        CONSTRAINT: Env(
            env[CONSTRAINT].data.concat(new_con_data),
            fit(env[CONSTRAINT].model, bfgs, new_con_data)
        ),
    }


constraint_query_points = jnp.array([[-1.0], [-0.6], [0.1], [0.3], [0.5], [0.9]])
constraint_data = Dataset(constraint_query_points, constraint(constraint_query_points))

constraint_gpr = ConjugateGPRegression(
    Dataset(jnp.empty([0, 1]), jnp.empty([0, 1])), gp, jnp.array([0.5]), jnp.array(0.2)
)
constraint_gpr = fit(constraint_gpr, bfgs, constraint_data)

OBJECTIVE, CONSTRAINT = "OBJECTIVE", "CONSTRAINT"

everything = {
    OBJECTIVE: Env(data, gpr),
    CONSTRAINT: Env(constraint_data, constraint_gpr),
}

eci = (
    expected_constrained_improvement(jnp.array(0.1))
    .contramap(map_model(predict_latent))
    .contramap(itemgetter(OBJECTIVE))
)
pof = (
    probability_of_feasibility(jnp.array([0.1]))
    .contramap(map_model(predict_latent))
    .contramap(itemgetter(CONSTRAINT))
)
acquisition_ = pof.apply(eci).map(multistart_bfgs(query_points[0], query_points[-1]))
points = loop(acquisition_, observer_, everything)
y = take(10, points)[-1][OBJECTIVE].data.features
print(y)
