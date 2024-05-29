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

from processing import add_noise


def create_train_step(key, model, lr, d):

    HTy_init = jnp.zeros((1, 1, d))


    x_init = jnp.zeros((1, 1, 1, d))
    HH_init = jnp.zeros((1, d, d))
    blur_init = jnp.array([1])
    L0 = jnp.eye(d)
    # HH = jnp.diagonal(H.T@H)

    parameter_state = model.init(key, x_init, HTy_init, blur_init, HH_init)
    partition_optimizers = {'trainable': optax.adam(lr), 'frozen': optax.set_to_zero()}
    param_partitions = traverse_util.path_aware_map(
        lambda path, v: 'frozen' if 'blur_embedding' in path else 'trainable', parameter_state)
    optimizer = optax.multi_transform(partition_optimizers, param_partitions)
    # optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(parameter_state)

    def reconstruction_loss(parameter_state, x_blurred_values, x_values, HTy_values, blur_values, HTH_values):
        x_hat_values, posterior_values = model.apply(
            parameter_state, x_blurred_values, HTy_values, blur_values, HTH_values)
        squared_residual = optax.l2_loss(x_hat_values, x_values).mean(axis=(0,1))
        weight = (1/(blur_values+1))**2
        weight /= weight.sum()
        weighted_residual = jnp.diag(weight) @ squared_residual
        x_normalization = (x_values**2).mean(axis=2)
        normalization = jnp.maximum(x_normalization, 1.)
        normalized_residual = squared_residual
        trace = jnp.trace(posterior_values, axis1=3, axis2=4)
        loss = normalized_residual.mean() + trace.mean()
        return loss

    @jax.jit
    def train_step(parameter_state, opt_state, X_init_batch, X_batch, HTY_batch, HTH_values, restoration_indices):
        loss_grad_fn = jax.value_and_grad(reconstruction_loss)

        loss, grads = loss_grad_fn(parameter_state, X_init_batch, X_batch,
                                   HTY_batch, restoration_indices, HTH_values)
        updates, opt_state = optimizer.update(grads, opt_state)
        parameter_state = optax.apply_updates(parameter_state, updates)

        return parameter_state, opt_state, loss
    
    return train_step, parameter_state, opt_state

def train_multi(
        model,
        X_train_multi,
        HTY_train,
        HTH_values,
        n_epochs,
        batch_size,
        test_function=None,
        lr=1e-3,
        plot_reconstructions=None
        ):
    
    print(f'n_epochs = {n_epochs}')



    n_train, n_process, n_iterations, d = X_train_multi.shape
    key = random.key(0)

    train_step, parameter_state, opt_state = create_train_step(key, model, lr, d)
    # n_epochs = 500
    # batch_size = 256

    restoration_indices = np.arange(n_iterations-1)
    blur_values = np.arange(n_iterations)/(n_iterations-1)
    # n_blur = 5
    n_grad = n_train//batch_size
    loss_values = np.zeros(n_epochs)
    error_values = []
    for epoch in tqdm(range(n_epochs)):
        X_epoch_blurred, HTY_epoch = shuffle(X_train_multi, HTY_train)
        X_epoch_true = X_epoch_blurred[:, :, 0, :] 
        for grad_index in range(n_grad):
            n_blur = n_iterations - 1
            # n_blur = 3

            restoration_indices = np.random.choice(n_iterations-1, size=n_blur, replace=False)
            blur_indices = restoration_indices+1
            H_indices = np.random.choice(n_process, size=1)

            X_true = X_epoch_true[grad_index*batch_size:(grad_index+1)*batch_size]
            X_batch = jnp.transpose(np.array([X_true]*(n_blur)), (1, 2, 0, 3))[:, H_indices]
            HTY_batch = HTY_epoch[grad_index*batch_size:(grad_index+1)*batch_size, H_indices]

            X_init_batch = X_epoch_blurred[grad_index * batch_size:(grad_index+1)*batch_size:, :, blur_indices, :][:, H_indices]

            parameter_state, opt_state, loss = train_step(
                parameter_state, opt_state, X_init_batch, X_batch, HTY_batch, HTH_values[H_indices], restoration_indices)
        loss_values[epoch] = loss

        X_hat_values, test_error_values = test_function(model, parameter_state)
        error_values.append(test_error_values)
        # print(f'loss = {loss}, error = {test_error_values[0]}')

        # wandb.log(
        #     {"error": test_error_values[0], 'one-step-error': test_error_values[-2], "loss": loss})
       
        if epoch % 10 != 0:
            continue
        # continue

        test_index = 1

        # plt.figure()
        # iteration_values = np.arange(0, len(X_hat_values))
        # X_hat_train_values = model.apply(parameter_state, X_train_multi[:, :, 1:], HTY_train, deblurring_values, HTH_values).transpose((2, 0, 1, 3))
        # print(f'X-hat_train {X_hat_train_values.shape}')
        # system.test_plot_blurred(X_hat_train_values[:, :, 0], X_train_multi[:, 0, :-1] , test_index=test_index, color='red')

        # plt.figure()
        # iteration_values = np.arange(len(X_hat_values))
        plt.figure(figsize=(12, 6))
        # print(f'Xhat shape {X_hat_values.shape}')
        # print(f'Xtest shape {X_hat_values.shape}')
        # system.test_plot_blurred(X_hat_values[:, :, 0], X_test_multi[:, 0], test_index=test_index, color='red')
        # system.test_plot_blurred(X_hat_values[:, :, 0], X_test_multi[:, 0], test_index=test_index, color='red')
        # plt.show()
        plot_reconstructions(X_hat_values)
        # system.plot_observations(y, sample_observed_indices)
        plt.pause(0.5)
        plt.close('all')

    return parameter_state, loss_values, error_values

