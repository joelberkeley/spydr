import jax.numpy as jnp

from spydr.distribution import Gaussian
import numpy.testing as jnpt


def test_gaussian_cdf() -> None:
    dist = Gaussian(jnp.array([[0.0]]), jnp.array([[[1.0]]]))
    jnpt.assert_allclose(dist.cdf(jnp.array([[-1.4]])), 0.0808, rtol=0.04)
    jnpt.assert_allclose(dist.cdf(jnp.array([[0.0]])), 0.5)
    jnpt.assert_allclose(dist.cdf(jnp.array([[1.0]])), 0.8413, rtol=0.04)
