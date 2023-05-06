from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from operator import itemgetter

import jax
from jax import jit
import jax.numpy as jnp

from spydr.bayesian_optimization import loop
from spydr.bayesian_optimization.acquisition import (
    expected_constrained_improvement, expected_improvement_by_model,
    negative_lower_confidence_bound,
    probability_of_feasibility, DataAndModel,
)
from spydr.bayesian_optimization.optimizer import multistart_bfgs
from spydr.distribution import Gaussian
from spydr.model import ProbabilisticModel
from spydr.data import Dataset
from spydr.model.gaussian_process import GaussianProcess, ConjugateGPRegression, fit, predict_latent
from spydr.model.kernel import matern52, rbf
from spydr.model.mean_function import zero
from spydr.optimize import bfgs
from spydr.util import compose


def objective(x: jnp.ndarray) -> jnp.ndarray:
    return (x - 1) ** 2


query_points = jnp.array(
    [[-1.0], [-0.9], [-0.6], [0.0], [0.1], [0.5], [0.9], [1.1], [1.2], [1.4], [1.7], [2.0], [2.3]]
)
data = Dataset(query_points, objective(query_points))

def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]))

gpr = fit(ConjugateGPRegression(data, gp, jnp.array([1.0]), jnp.array(0.2)), bfgs, data)

# this, and therefore `GaussianProcess`, have to be pytrees for this to work.
# that may mean rewriting `GaussianProcess` to not work recursively (which is correct anyway).
# we could look alternatively at using gpjax
@dataclass(frozen=True, eq=False)
class Env:
    data: Dataset
    model: ConjugateGPRegression

acquisition = (
    expected_improvement_by_model()
    .contramap(lambda e: DataAndModel(e.data, predict_latent(e.model)))
    .map(lambda f: multistart_bfgs(f, query_points[0], query_points[-1]))
)

def concat(this: Dataset, that: Dataset) -> Dataset:
    return Dataset(
        jnp.concatenate([this.features, that.features]),
        jnp.concatenate([this.targets, that.targets])
    )

##################################

# constraint_query_points = jnp.array([[-1.0], [-0.6], [0.1], [0.3], [0.5], [0.9]])
# constraint_data = (constraint_query_points, jnp.sin(constraint_query_points - 1) ** 2)
#
# constraint_gpr = fit(ConjugateGPRegression(gp, jnp.array([0.5]), jnp.array(0.2)), bfgs, data)
#
# OBJECTIVE, CONSTRAINT = "OBJECTIVE", "CONSTRAINT"
#
# all_data = {OBJECTIVE: data, CONSTRAINT: constraint_data}
# all_models = {OBJECTIVE: gpr, CONSTRAINT: constraint_gpr}
#
# objective = itemgetter(OBJECTIVE)
# constraint = itemgetter(CONSTRAINT)
#
# eci = expected_constrained_improvement(jnp.array(0.5)) << objective >> compose(predict_latent, objective)
# pof = probability_of_feasibility(jnp.array(0.5)) << constraint >> compose(predict_latent, constraint)
# acquisition_ = pof.apply(eci).map(multistart_bfgs)
# print(acquisition_(all_data, all_models))

#################################

def observer(
        points: jnp.ndarray, env: Env,
) -> Env:
    new_data = Dataset(points, objective(points))
    return Env(concat(env.data, new_data), fit(env.model, bfgs, new_data))

def take(n, xs):
    return [next(xs) for _ in range(n)]

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
points = loop(acquisition, observer, Env(data, gpr))
y = take(10, points)[-1].data.features
print(y)
    # y.block_until_ready()
