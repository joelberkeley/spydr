# spydr

spydr is a Python port of the machine learning research library [spidr](https://github.com/joelberkeley/spidr). While spidr has its own tensor API that uses [XLA](https://www.tensorflow.org/xla) directly for its backend, spydr uses [JAX](https://github.com/google/jax) which provides its own [NumPy](https://numpy.org/) array API powered by XLA (and [Autograd](https://github.com/hips/autograd)). Otherwise, spidr and spyder are largely identical.

## Installation

Clone this repository, then run
```bash
$ pip install .
```
in the repository root.
