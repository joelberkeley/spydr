import jax.numpy as jnp
from spydr.model.kernel import rbf
import numpy.testing as jnpt


def test_rbf() -> None:
    actual = rbf(jnp.array(1.0))(jnp.array([[1.0]]), jnp.array([[0.0]]))
    jnpt.assert_almost_equal(actual, jnp.exp(jnp.array([[-0.5]])))
