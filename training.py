import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from tqdm import tqdm
from flax import linen as nn, traverse_util
from flax.training import train_state
import optax
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import wandb

from processing import interpolate


def create_train_step(key, model, lr, d):

    Y_init = jnp.zeros((1, 1, d))


    X_init = jnp.zeros((1, 1, 1, d))
    H_init = jnp.zeros((1, d, d))
    blur_init = jnp.array([1])

    parameter_state = model.init(key, X_init, Y_init, H_init, blur_init)
    partition_optimizers = {'trainable': optax.adam(lr), 'frozen': optax.set_to_zero()}
    param_partitions = traverse_util.path_aware_map(
        lambda path, v: 'frozen' if 'blur_embedding' in path else 'trainable', parameter_state)
    optimizer = optax.multi_transform(partition_optimizers, param_partitions)
    opt_state = optimizer.init(parameter_state)

    def reconstruction_loss(parameter_state, Z_batch, X_batch, Y_batch, H_batch, blur_batch):

        reconstruction_batch, posterior_batch = model.apply(parameter_state, Z_batch, Y_batch, H_batch, blur_batch)
        squared_residual = optax.l2_loss(reconstruction_batch, X_batch)
        normalized_residual = squared_residual
        loss = normalized_residual.mean() 
        return loss

    @jax.jit
    def train_step(parameter_state, opt_state, Z_batch, X_batch, Y_batch, H_batch, blur_batch):
        loss_grad_fn = jax.value_and_grad(reconstruction_loss)

        loss, grads = loss_grad_fn(parameter_state, Z_batch, X_batch,
                                   Y_batch, H_batch, blur_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        parameter_state = optax.apply_updates(parameter_state, updates)

        return parameter_state, opt_state, loss
    
    return train_step, parameter_state, opt_state

def training_loop(
        model,
        X,
        Z0,
        Y_train,
        H_values,
        n_epochs,
        batch_size,
        blur_batch_size=None,
        test_function=None,
        lr=1e-3,
        ):
    
    print(f'n_epochs = {n_epochs}')

    n_train, d = X.shape
    n_train, n_processes, d = Y_train.shape

    key = random.key(0)
    train_step, parameter_state, opt_state = create_train_step(key, model, lr, d)

    blur_batch_size = model.blur_max if blur_batch_size is None else blur_batch_size
    process_batch_size = 1
    n_grad = n_train//batch_size
    loss_values = np.zeros(n_epochs)
    error_values = []
    for epoch in tqdm(range(n_epochs)):
        X_epoch, Y_epoch, Z0_epoch = shuffle(X, Y_train, Z0)
        for grad_index in range(n_grad):

            batch_blur_values = 1 + np.random.choice(model.blur_max, size=blur_batch_size, replace=False)
            H_indices = np.random.choice(n_processes, size=process_batch_size)

            X_batch = X_epoch[grad_index*batch_size:(grad_index+1)*batch_size]
            Z0_batch = Z0_epoch[grad_index*batch_size:(grad_index+1)*batch_size][:, H_indices]
            Y_batch = Y_epoch[grad_index*batch_size:(grad_index+1)*batch_size, H_indices]
            Hbatch = H_values[H_indices]

            X_batch_process = jnp.repeat(
                jnp.expand_dims(X_batch, 1), process_batch_size, axis=1)
            Z_batch = interpolate(X_batch_process, Z0_batch, batch_blur_values, model.blur_max)
            X_batch_multi = jnp.repeat(
                jnp.expand_dims(X_batch_process, 2), blur_batch_size, axis=2)

            parameter_state, opt_state, loss = train_step(parameter_state, opt_state, Z_batch, X_batch_multi, Y_batch, Hbatch, batch_blur_values)
        loss_values[epoch] = loss

        X_hat_values, test_error_values = test_function(model, parameter_state)
        error_values.append(test_error_values)

    return parameter_state, loss_values, error_values

