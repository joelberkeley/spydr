import jax
import jax.numpy as jnp
from jax import jit

from spydr.bayesian_optimization.acquisition import Acquisition
from spydr.optimize import bfgs


def acquisition_optimizer(f: Acquisition, low: jnp.array, high: jnp.array) -> jnp.ndarray:
    f = jit(f)
    x = (jnp.arange(10.0) / 10) * (high - low) + low
    start = x[jnp.argmax(jax.vmap(f)(x[:, None, None]))][None]
    # print(start)
    return bfgs(start)(lambda x: f(x[None]))[None]
