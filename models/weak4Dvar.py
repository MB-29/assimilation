import numpy as np
import jax
import jaxopt
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random
import optax
from jaxopt import LevenbergMarquardt

from collections.abc import Callable

class Weak4DVar(nn.Module):
    d: int
    r: float
    model: Callable

    def setup(self):
        pass

    def interpolate(self, x_init, y, H, **kwargs):
        
        @jax.jit
        def residual_function(x):
            data_residual = (1/self.r)*(H@x - y)
            prior_residual = jnp.stack(self.model(x))
            residual = jnp.concatenate([data_residual, prior_residual])
            return residual

        optimizer = LevenbergMarquardt(residual_fun=residual_function, **kwargs)
        output = optimizer.run(x_init)
        x_hat = output.params

        return x_hat, output
    


# def lorenz_model()