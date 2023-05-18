import chex
import jax
import jax.numpy as jnp

from spydr.bayesian_optimization.acquisition import Acquisition
from spydr.optimize import bfgs, Optimizer


def multistart_bfgs(low: jnp.ndarray, high: jnp.ndarray) -> Optimizer[jnp.ndarray]:
    def optimizer(acquisition: Acquisition) -> jnp.ndarray:
        start_candidates = (jnp.broadcast_to(jnp.arange(10.0).reshape([10, 1]) / 10, (10,) + low.shape)) * (high - low) + low
        print("start_candidates", start_candidates)
        chex.assert_shape(start_candidates, [10] + list(low.shape))
        start_candidate_evaluations = jax.vmap(acquisition)(start_candidates[:, None, :])
        print("start_candidate_evaluations", start_candidate_evaluations)
        start = start_candidates[jnp.argmax(start_candidate_evaluations)]

        chex.assert_equal_shape([start, low])

        def reshaped_acquisition(x: jnp.ndarray) -> jnp.ndarray:
            chex.assert_equal_shape([x, low])
            return acquisition(x[None])

        print("start", start)
        res = bfgs(start)(reshaped_acquisition)[None]
        print("optimum", res)
        return res

    return optimizer
