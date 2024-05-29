import jax.experimental
import jax.experimental.sparse
import numpy as np
import jax
import jaxopt
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random
import functools
import optax

class UnconditionalPoint(nn.Module):
    d: int
    r: float
    n_blur: int
    assimilation: str = 'unconditional'
    # HH: jax.Array

    def setup(self):


        self.mu = nn.NeuralAssimilation([
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(self.d)
        ])
        sin_embedding = self.sinusoidal_embedding(feature_dimension=100)
        self.blur_embedding = nn.Embed(
            self.n_blur, 100, embedding_init=lambda rng, shape, dtype: sin_embedding)


    
    def embed(self, x, blur_index):
        blur_embedding = self.blur_embedding(blur_index-1)
        x_blur = jnp.concatenate((x, blur_embedding), axis=-1)
        return x_blur
    
    
    def __call__(self, x, HTy, blur_index, HTH):

        x_blur = self.embed(x, blur_index) 
        mu = self.mu(x_blur) 
        M = jnp.zeros((self.d, self.d))
        x_hat = mu

        return x_hat, M

    
    def sinusoidal_embedding(self, feature_dimension):
        d = feature_dimension

        # Returns the standard positional embedding
        embedding = np.array([[i / 10_000 ** (2 * j / d)
                               for j in range(d)] for i in range(self.n_blur)])
        sin_mask = np.arange(0, self.n_blur, 2)

        embedding[sin_mask] = np.sin(embedding[sin_mask])
        embedding[1 - sin_mask] = np.cos(embedding[sin_mask])

        return jnp.array(embedding)

    def reconstruct_multi(self, parameter_state, X_init, HTY, HTH_values):
        n_blur = self.n_blur
        blur_values = np.arange(1, self.n_blur+1) 
        batch_size, n_process, d = X_init.shape
        X0 = X_init.reshape((batch_size, n_process, 1, self.d))
        Xs = X0.copy()
        X_hat_values = np.zeros((n_blur+1, batch_size, n_process, self.d))
        X_hat_values[n_blur] = X_init
        for iteration in range(n_blur):
            # print(f'iteration {iteration}')
            # blur_index = n_blur - iteration - 1
            blur_index = n_blur - iteration - 1
            weight = ((blur_index+1)/n_blur)
            weight_ = (blur_index/n_blur)
            # print(f'weight : {weight}, weight_:{weight_}')
            blur = jnp.array([blur_values[blur_index]])
            # print(f'iteration {iteration},\niteration index {iteration_index},\nn iterations = {n_blur}\nblur index {blur_index}')
            # blur = blur_index/ n_blur
            # blur_ = jnp.array([blur])
            # blur_values = jnp.array([blur]).reshape((1, 1))
            # print(f'X_hat = {X_hat.shape}, HYY = {HTY.shape}, blur = {blur}, HTH = {HTH_values.shape}')
            mu, cov = self.apply(parameter_state, Xs, HTY, blur, HTH_values)
            # cov = self.apply(parameter_state, x_hat, blur_index, method=self.compute_posterior)
            x_hat = mu
            # x_hat = jnp.stack([np.random.multivariate_normal(mu[i, j], cov[]) 
            Ds = weight*X0 + (1-weight)*x_hat
            Ds_ = weight_*X0 + (1-weight_)*x_hat
            Xs = Xs - Ds + Ds_
            # Xs = weight_*X0 + (1-weight_)*x_hat
            # self.init(random.key(0), X_hat, Y, blur, F, H)
            # X_hat_value = X_hat.squeeze().copy()

            X_hat_values[blur_index] = np.array(x_hat.squeeze())
            # X_hat_values[blur_index] = np.array(Xs.squeeze())
        # for i in range(10):
        #     x_hat = self.apply(parameter_state, x_hat, HTY, blur, HTH_values)
        #     X_hat_values[blur_index] = np.array(x_hat.squeeze())
        result = Xs if n_blur == 1 else X_hat_values
        return result
    
    def sample(self, parameter_state, X_init, HTY, HTH_values, test_index, process_index):
        n_blur = self.n_blur
        blur_values = np.arange(1, self.n_blur+1) 
        batch_size, n_process, d = X_init.shape
        X0 = X_init.reshape((batch_size, n_process, 1, self.d))
        Xs = X0.copy()
        X_hat_values = np.zeros((n_blur+1, self.d))
        X_hat_values[n_blur] = X_init[test_index, process_index]
        for iteration in range(n_blur):
            # print(f'iteration {iteration}')
            # blur_index = n_blur - iteration - 1
            blur_index = n_blur - iteration - 1
            weight = ((blur_index+1)/n_blur)
            weight_ = (blur_index/n_blur)
            # print(f'weight : {weight}, weight_:{weight_}')
            blur = jnp.array([blur_values[blur_index]])
            # print(f'iteration {iteration},\niteration index {iteration_index},\nn iterations = {n_blur}\nblur index {blur_index}')
            # blur = blur_index/ n_blur
            # blur_ = jnp.array([blur])
            # blur_values = jnp.array([blur]).reshape((1, 1))
            # print(f'X_hat = {X_hat.shape}, HYY = {HTY.shape}, blur = {blur}, HTH = {HTH_values.shape}')
            mu, cov = self.apply(parameter_state, Xs, HTY, blur, HTH_values)
            # cov = self.apply(parameter_state, x_hat, blur_index, method=self.compute_posterior)
            x_hat = mu
            mu_sample = mu[test_index, process_index].squeeze()
            sample = mu[test_index, process_index]
            # cov_sample = 0.000001*np.eye(d)
            cov_sample = cov[test_index, process_index].squeeze()
            # print(f'mu {mu_sample}x')
            # print(f'cov {cov_sample}')
            sample = np.random.multivariate_normal(mean=mu_sample, cov=cov_sample)
            x_hat = x_hat.at[test_index, process_index].set(sample)
            # x_hat = jnp.stack([np.random.multivariate_normal(mu[i, j], cov[]) 
            Ds = weight*X0 + (1-weight)*x_hat
            Ds_ = weight_*X0 + (1-weight_)*x_hat
            Xs = Xs - Ds + Ds_
            # Xs = weight_*X0 + (1-weight_)*x_hat
            # self.init(random.key(0), X_hat, Y, blur, F, H)
            # X_hat_value = X_hat.squeeze().copy()

            X_hat_values[blur_index] = np.array(sample.squeeze())
            # X_hat_values[blur_index] = np.array(Xs.squeeze())
        # for i in range(10):
        #     x_hat = self.apply(parameter_state, x_hat, HTY, blur, HTH_values)
        #     X_hat_values[blur_index] = np.array(x_hat.squeeze())
        result = Xs if n_blur == 1 else X_hat_values
        return result



# lifted_methods = ['__call__', 'embed']
UnconditionalGranularity = nn.vmap(
    UnconditionalPoint,
    in_axes=(0, None, 0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
UnconditionalMulti = nn.vmap(
    UnconditionalGranularity,
    in_axes=(0, 0, None, 0), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
Unconditional = nn.vmap(
    UnconditionalMulti,
    in_axes=(0, 0, None, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
# NeuralAssimilation = nn.vmap(
#   NeuralAssimilationPoint,
#   in_axes=(0, 0, None, None, None), out_axes=0,
#   variable_axes={'params': None},
#   split_rngs={'params': False}
# )
