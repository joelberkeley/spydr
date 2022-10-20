from __future__ import annotations

from itertools import islice
from operator import itemgetter

import jax
import jax.numpy as jnp

from spydr.bayesian_optimization import loop
from spydr.bayesian_optimization.acquisition import (
    expected_constrained_improvement, expected_improvement_by_model, negative_lower_confidence_bound,
    probability_of_feasibility,
)
from spydr.bayesian_optimization.optimizer import acquisition_optimizer
from spydr.distribution import Gaussian
from spydr.model import ProbabilisticModel
from spydr.data import Dataset
from spydr.model.gaussian_process import GaussianProcess, ConjugateGPRegression, fit, predict_latent
from spydr.model.kernel import matern52, rbf
from spydr.model.mean_function import zero
from spydr.optimize import bfgs
from spydr.util import compose


def objective(x: jnp.ndarray) -> jnp.ndarray:
    return (x / 2 + jnp.sin(x)) ** 2


query_points = jnp.array(
    [[-1.0], [-0.9], [-0.6], [0.0], [0.1], [0.5], [0.9], [1.1], [1.2], [1.4], [1.7], [2.0], [2.3]]
)
data = (query_points, objective(query_points))

def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]))

gpr = fit(ConjugateGPRegression(gp, jnp.array([1.0]), jnp.array(0.2)), bfgs, data)

ei = expected_improvement_by_model() >> predict_latent
acquisition = ei.map(lambda f: acquisition_optimizer(f, query_points[0], query_points[-1]))

def concat(this: Dataset, that: Dataset) -> Dataset:
    return tuple(map(jnp.concatenate, zip(this, that)))

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
next_point = acquisition(data, gpr)
data1 = concat(data, (next_point, objective(next_point)))
gpr1 = fit(gpr, bfgs, data1)
next_point1 = acquisition(data1, gpr1)
data2 = concat(data1, (next_point1, objective(next_point1)))
gpr2 = fit(gpr1, bfgs, data2)
next_point2 = acquisition(data2, gpr2)
data3 = concat(data2, (next_point2, objective(next_point2)))
gpr3 = fit(gpr2, bfgs, data3)
next_point3 = acquisition(data3, gpr3)
data4 = concat(data3, (next_point3, objective(next_point3)))
gpr4 = fit(gpr3, bfgs, data4)
next_point4 = acquisition(data4, gpr4)
data5 = concat(data4, (next_point4, objective(next_point4)))
gpr5 = fit(gpr4, bfgs, data5)
next_point5 = acquisition(data5, gpr5)
data6 = concat(data5, (next_point5, objective(next_point5)))
gpr6 = fit(gpr5, bfgs, data6)
next_point6 = acquisition(data6, gpr6)

print(next_point)
print(next_point1)
print(next_point2)
print(next_point3)
print(next_point4)
print(next_point5)
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
# acquisition_ = pof.apply(eci).map(acquisition_optimizer)
# print(acquisition_(all_data, all_models))

#################################

def observer(
        points: jnp.ndarray, dataset: Dataset, model: ConjugateGPRegression
) -> tuple[Dataset, ConjugateGPRegression]:
    print("step")
    new_data = (points, objective(points))
    return concat(dataset, new_data), fit(model, bfgs, new_data)


ei = expected_improvement_by_model() >> predict_latent
#
# ei(data, )

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     points = loop(ei.map(lambda f: acquisition_optimizer(f, query_points[0], query_points[-1])), observer, (data, gpr))
#     _ = next(points)
#     # _ = next(points)
#     # _ = next(points)
#     # _ = next(points)
#     res = next(points)
#     y = res[0][0][-10:]
#     y.block_until_ready()
#
# print(y)

# points = loop(ei.map(acquisition_optimizer), observer, (data, gpr))
# first_five = islice(points, 10)
# print(list(first_five)[-1][0][0][-10:])
