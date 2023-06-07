# %%
# %load_ext autoreload
# %autoreload 2

# %%
import jax.numpy as jnp
import jax

from spydr.data import Dataset
from spydr.model.gaussian_process import ConjugateGPRegression, GaussianProcess, fit, predict_latent
from spydr.model.kernel import rbf, matern52
from spydr.model.mean_function import zero
from spydr.optimize import bfgs

import seaborn as sns

# %%
def f(x: jnp.ndarray) -> jnp.array:
    return jnp.sin(4 * x) + jnp.cos(2 * x)


key = jax.random.PRNGKey(0)
x = jax.random.uniform(key, [20, 1], minval=-3.0, maxval=3.0).sort()
dataset = Dataset(x, f(x) + jax.random.normal(key, shape=[20, 1]))


def gp(params: jnp.ndarray) -> GaussianProcess:
    return GaussianProcess(zero, rbf(params[0]), feature_shape=[1])


gpr = ConjugateGPRegression(Dataset.empty([1], [1]), gp, jnp.array([0.5]), jnp.array(0.5))

gpr = fit(gpr, bfgs, dataset)

key = jax.random.PRNGKey(2)
x_ = jax.random.uniform(key, [20, 1], minval=-3.0, maxval=3.0).sort()
predictions = predict_latent(gpr)(x_)

import pandas as pd

training = pd.DataFrame({'x': dataset.features.squeeze(axis=-1), 'y': dataset.targets.squeeze(axis=-1)})
predictions_ = pd.DataFrame({'x': x_.squeeze(axis=-1), 'y': predictions.mean.squeeze(axis=-1)})
# all_ = jnp.concatenate([
#     jnp.concatenate([dataset.targets, predictions.mean]),
#     jnp.concatenate([x, x_])
# ], axis=0)

# %%
sns.lineplot(data=training, x='x', y='y')

# %%
sns.lineplot(data=predictions_, x='x', y='y')

# %%

# %%

# %%
