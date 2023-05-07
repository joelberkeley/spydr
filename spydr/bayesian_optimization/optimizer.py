import jax
import jax.numpy as jnp

from spydr.bayesian_optimization.acquisition import Acquisition
from spydr.optimize import bfgs, Optimizer


def multistart_bfgs(low: jnp.ndarray, high: jnp.ndarray) -> Optimizer[jnp.ndarray]:
    def optimizer(f: Acquisition) -> jnp.ndarray:
        x = (jnp.arange(10.0) / 10) * (high - low) + low
        start = x[jnp.argmax(jax.vmap(f)(x[:, None, None]))][None]
        return bfgs(start)(lambda x: f(x[None]))[None]

    return optimizer
