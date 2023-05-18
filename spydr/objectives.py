import math
from typing import Final

import jax.numpy as jnp


def unit_branin(x: jnp.ndarray) -> jnp.ndarray:
    x0 = x[..., :1] * 15.0 - 5.0
    x1 = x[..., 1:] * 15.0

    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return 1 / 51.95 * ((x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * jnp.cos(x0) - 44.81)


UNIT_BRANIN_MINIMUM: Final = -1.0473
