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
from jax.experimental.sparse import BCOO, bcoo_dot_general, bcoo_fromdense
from jax.scipy.linalg import solve

class NeuralAssimilationPoint(nn.Module):
    d: int
    r: float
    blur_max: int
    k: int = 5


    def setup(self):

        self.L = nn.Sequential([
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            # nn.Dense(self.d*6-15),
            nn.Dense(self.d*self.k),
            nn.tanh
        ])
        self.mu = nn.Sequential([
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
            self.blur_max, 100, embedding_init=lambda rng, shape, dtype: sin_embedding)
        # self.q = self.param('q', lambda rng, shape: 0.001, (1,))


    
    def embed(self, x, blur_index):
        blur_embedding = self.blur_embedding(blur_index-1)
        x_blur = jnp.concatenate((x, blur_embedding), axis=-1)
        return x_blur
    
    def compute_prior(self, x, blur_index):
        x_blur = self.embed(x, blur_index)
        # obs_embedding = jnp.diag(HTH)
        # x_blur_obs = jnp.concatenate((x_blur, obs_embedding), axis=-1)
        L = self.L(x_blur)
        index_values = [[i, i] for i in range(self.d)]
        index = self.d
        data = [L[:index]**2]
        for j in range(1, 6):
            index_values += [[i, i+j] for i in range(self.d-j)]
            index_values += [[i+j, i] for i in range(self.d-j)]
            length = self.d-j
            values = L[index:index+length]
            # if j%3 == 0:
            #     values = values**2
            # values = 1
            data += 2*[values]
            index += length
        # for j in range(1, 6):
        indices = jnp.array(index_values)
        data = jnp.concatenate(data)
        # data.at[:self.d:3*self.d-2].set(0.9)
        # data[:self.d:2*self.d-1] += 0.5
        M = BCOO((data, indices), shape=(self.d, self.d))

        return M
    
    def __call__(self, x, HTy, HTH, blur):

        x_blur = self.embed(x, blur) 
        mu = self.mu(x_blur) 
        M = jnp.zeros((self.d, self.d))
        x_hat = mu
        # x_blur_obs = jnp.concatenate((x_blur, obs_embedding), axis=-1)
        
        L = self.L(x_blur).reshape((self.d, self.k))
        innovation = HTy - HTH@(mu)
        prior_precision = jnp.eye(self.d) + L@L.T
        # prior_precision = jnp.eye(self.d) + self.compute_prior(x, blur).todense()
        posterior_precision = (self.r**2)*prior_precision + HTH
        # posterior_precision = 0.1*jnp.eye(self.d) + (1/self.r**2)*HTH
        # update = posterior @ innovation
        # obs_embedding = jnp.diag(HTH)
        # L = self.L(x_blur).reshape((self.d, self.k))
        # P = L@L.T
        # prior_precision = jnp.eye(self.d) + P + (1/self.r**2)*HTH
        # update = posterior @ innovation
        update = solve(posterior_precision, innovation, assume_a='pos')

        x_hat =  mu + update
        # M = jnp.linalg.inv(prior_precision)
        # cov = prior.todense()
        # cov = posterior

        return x_hat, M

    
    def sinusoidal_embedding(self, feature_dimension):
        d = feature_dimension

        # Returns the standard positional embedding
        embedding = np.array([[i / 10_000 ** (2 * j / d)
                               for j in range(d)] for i in range(self.blur_max)])
        sin_mask = np.arange(0, self.blur_max, 2)

        embedding[sin_mask] = np.sin(embedding[sin_mask])
        embedding[1 - sin_mask] = np.cos(embedding[sin_mask])

        return jnp.array(embedding)

    def reconstruct_multi(self, parameter_state, X_init, HTY, HTH_values):
        n_blur = self.blur_max
        blur_values = np.arange(1, self.blur_max+1) 
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
            mu, cov = self.apply(parameter_state, Xs, HTY, HTH_values, blur)
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
        n_blur = self.blur_max
        blur_values = np.arange(1, self.blur_max+1) 
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
            mu, cov = self.apply(parameter_state, Xs, HTY, HTH_values, blur)
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
NeuralAssimilationBlur = nn.vmap(
    NeuralAssimilationPoint,
    in_axes=(0, None, None, 0), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
NeuralAssimilationObs = nn.vmap(
    NeuralAssimilationBlur,
    in_axes=(0, 0, 0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
NeuralAssimilation = nn.vmap(
    NeuralAssimilationObs,
    in_axes=(0, 0, None, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    # methods=lifted_methods
)
