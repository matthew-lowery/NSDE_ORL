

import jax
import equinox as eqx
from jax import numpy as jnp

def GaussianKernel(x,y,r):

    ### one pair of pts
    def eval(x, y):
        sq_norm = jnp.linalg.norm(x-y)**2
        return jnp.exp(-sq_norm / (2*r**2))

    if x.ndim == 1 or y.ndim == 1:
        ndims = 1
    else:
        ndims = x.shape[-1]
    X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
    k_xy = jax.vmap(jax.vmap(eval, (0, None)), (None, 0))(Y,X)
    return k_xy