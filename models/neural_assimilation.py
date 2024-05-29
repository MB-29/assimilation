import jax.experimental.sparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental.sparse import BCOO
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


    
    def embed(self, x, blur_index):
        blur_embedding = self.blur_embedding(blur_index-1)
        x_blur = jnp.concatenate((x, blur_embedding), axis=-1)
        return x_blur
    
    def compute_prior(self, x, blur_index):
        x_blur = self.embed(x, blur_index)
        L = self.L(x_blur)
        index_values = [[i, i] for i in range(self.d)]
        index = self.d
        data = [L[:index]**2]
        for j in range(1, 6):
            index_values += [[i, i+j] for i in range(self.d-j)]
            index_values += [[i+j, i] for i in range(self.d-j)]
            length = self.d-j
            values = L[index:index+length]
            data += 2*[values]
            index += length
        indices = jnp.array(index_values)
        data = jnp.concatenate(data)
        M = BCOO((data, indices), shape=(self.d, self.d))

        return M
    
    def __call__(self, x, HTy, HTH, blur):

        x_blur = self.embed(x, blur) 
        mu = self.mu(x_blur) 
        M = jnp.zeros((self.d, self.d))
        x_hat = mu
        
        L = self.L(x_blur).reshape((self.d, self.k))
        innovation = HTy - HTH@(mu)
        prior_precision = jnp.eye(self.d) + L@L.T
        posterior_precision = (self.r**2)*prior_precision + HTH
        update = solve(posterior_precision, innovation, assume_a='pos')

        x_hat =  mu + update

        return x_hat, M

    
    def sinusoidal_embedding(self, feature_dimension):
        d = feature_dimension

        embedding = np.array([[i / 10_000 ** (2 * j / d)
                               for j in range(d)] for i in range(self.blur_max)])
        sin_mask = np.arange(0, self.blur_max, 2)

        embedding[sin_mask] = np.sin(embedding[sin_mask])
        embedding[1 - sin_mask] = np.cos(embedding[sin_mask])

        return jnp.array(embedding)

    def reconstruct(self, parameter_state, Z_batch, Y_batch, H_batch):
        n_blur = self.blur_max
        blur_values = np.arange(1, self.blur_max+1) 
        batch_size, n_process, d = Z_batch.shape
        X0 = Z_batch.reshape((batch_size, n_process, 1, self.d))
        Xs = X0.copy()
        X_hat_values = np.zeros((n_blur+1, batch_size, n_process, self.d))
        X_hat_values[n_blur] = Z_batch
        for iteration in range(n_blur):
            blur_index = n_blur - iteration - 1
            weight = ((blur_index+1)/n_blur)
            weight_ = (blur_index/n_blur)
            blur = jnp.array([blur_values[blur_index]])
            mu, cov = self.apply(parameter_state, Xs, Y_batch, H_batch, blur)
            x_hat = mu
            Ds = weight*X0 + (1-weight)*x_hat
            Ds_ = weight_*X0 + (1-weight_)*x_hat
            Xs = Xs - Ds + Ds_

            X_hat_values[blur_index] = np.array(x_hat.squeeze())
        result = Xs if n_blur == 1 else X_hat_values
        return result


NeuralAssimilationBlur = nn.vmap(
    NeuralAssimilationPoint,
    in_axes=(0, None, None, 0), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
)
NeuralAssimilationObs = nn.vmap(
    NeuralAssimilationBlur,
    in_axes=(0, 0, 0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
)
NeuralAssimilation = nn.vmap(
    NeuralAssimilationObs,
    in_axes=(0, 0, None, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
)
