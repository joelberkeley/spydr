import jax
import jax.numpy as jnp

from spydr.bayesian_optimization.acquisition import Acquisition
from spydr.optimize import bfgs


def multistart_bfgs(f: Acquisition, low: jnp.array, high: jnp.array) -> jnp.ndarray:
    x = (jnp.arange(10.0) / 10) * (high - low) + low
    start = x[jnp.argmax(jax.vmap(f)(x[:, None, None]))][None]
    return bfgs(start)(lambda x: f(x[None]))[None]
