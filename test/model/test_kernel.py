import jax.numpy as np
from spydr.model.kernel import rbf
import numpy.testing as npt


def test_rbf() -> None:
    actual = rbf(np.array(1.0))(np.array([[1.0]]), np.array([[0.0]]))
    npt.assert_almost_equal(actual, np.exp(np.array([[-0.5]])))
