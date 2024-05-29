import os
import numpy as np
import scipy
import jax
import jaxopt
import jax.numpy as jnp
import optax
from jax import random
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp

from models.neural import NeuralAssimilation
from filters.lorenz import generate_lowpass_filters
from training import train_multi
from processing import interpolate

from systems.pendulum import Pendulum as System

np.random.seed(0)


T, dt = 100, 0.05
system = System(T, dt)


d = system.d
n_samples = 500
string = f'_multi_d={system.d}_n_samples={n_samples}'
data = system.load_data(string=string)


X = data['X']
X_gaussian = data['X_gaussian']
Y_values = data['Y']
HTY_values = data['HTY']
H_values = data['H_values']
HTH_values = data['HTH_values']
rho = data['rho']
observed_indices = data['observed_indices']
n_samples = data['n_samples']
n_train = data['n_train']
n_processes = data['n_processes']
n_processes_train = data['n_processes_train']
n_processes_test = n_processes - n_processes_train

# sample_observed_indices = observed_indices[0]
# y = Y_values[0][0]
# x = X[0]
# x_gaussian = X_gaussian[0][0]
# plt.figure(figsize=(16, 4))
# system.plot_trajectory(x, color='black')
# system.plot_trajectory(x_gaussian, color='blue')
# system.plot_observations(y, sample_observed_indices)
# plt.show()

n_interpolations = 5
blur_values = np.arange(n_interpolations+1)
X_multi = jnp.array([interpolate(X, X_gaussian[index], n_interpolations) for index in range(n_processes)])
X_multi = X_multi.transpose(1, 0, 2, 3)

deblurring_values = blur_values[1:]

# X_train = X[:n_train]
# X_train_gaussian = X_gaussian[:n_processes_train, :n_train]

# X_train_multi = jnp.array([[X_train, (1/2)*(X_train+X_train_gaussian[index]), X_train_gaussian[index]] for index in range(n_processes_train)])
# X_train_multi = X_train_multi.transpose(2, 0, 1, 3)
# X_train_multi = interpolate(X_train, X_train_gaussian, n_interpolations)
X_train_multi = X_multi[:n_train, :n_processes_train] 

HTY_train_multi = HTY_values[:n_train, :n_processes_train]
HTH_train_values = HTH_values[:n_processes_train]

# X_test = X[n_train:]
# X_test_gaussian = X_gaussian[n_processes_train:, n_train:]
# jnp.array([[X_test, (1/2)*(X_test+X_test_gaussian[index]), X_test_gaussian[index]] for index in range(n_processes_train, n_processes)])
# X_test_multi = X_test_multi.transpose(2, 0, 1, 3)
X_test = X[n_train:]
X_init_test = X_gaussian.transpose(1, 0, 2)[n_train:, n_processes_train:]

HTY_test_multi = HTY_values[n_train:, n_processes_train:]
HTH_test_values = HTH_values[n_processes_train:]
# X_train_blurred = system.interpolate(X_train, X_gaussian_train, n_interpolations)




print('model')
HTy_init = jnp.zeros((1, 1, d))
x_init = jnp.zeros((1, 1, 1, d))
HH_init = jnp.zeros((1, d, d))
blur_init = jnp.array([1])
L0 = jnp.eye(d)
# HH = jnp.diagonal(H.T@H)

model = NeuralAssimilation(d, rho, n_interpolations)
key = random.key(0)
parameter_state = model.init(key, x_init, HTy_init, blur_init, HH_init)

print('training')

# n_epochs = 100
# n_epochs = 2_000
n_epochs = 200
batch_size = 32
# batch_size = 64
n_grad = n_train//batch_size
loss_values = np.zeros(n_epochs)
error_values = []
lr = 5e-4


def evaluate_model(model, parameter_state):
    X_hat_values = model.reconstruct_multi(
        parameter_state, X_init_test, HTY_test_multi, HTH_test_values)
    X_test_multi = jnp.array(
        [X_test for i in range(n_processes_test)]).transpose(1, 0, 2)
    test_error_values = [optax.l2_loss(
        X_hat.squeeze(), X_test_multi).mean() for X_hat in X_hat_values]
    # test_error_values = [optax.l2_loss(
    #     X_hat[:, 0].squeeze(), X_test[:, 0]).mean() for X_hat in X_hat_values]

    return X_hat_values, test_error_values


def plot_reconstructions(X_hat_values, **kwargs):
    test_index, process_index = 0, 0
    # iteration_indices = np.arange(0, n_interpolations, 5)
    iteration_indices = np.arange(n_interpolations)
    n_rows = len(iteration_indices)
    for row_index, iteration in enumerate(iteration_indices):
        blur_index = n_interpolations-iteration-1
        x_hat = X_hat_values[blur_index, test_index, process_index]
        x = X_test[test_index]
        y = Y_values[n_processes_train+process_index][n_train+test_index]
        x_gaussian = X_init_test[test_index, process_index]
        system.plot_trajectory(x, n_rows=n_rows,
                               row_index=row_index, color='black', lw=2)
        system.plot_trajectory(x_gaussian, n_rows=n_rows,
                               row_index=row_index, color='blue', ls='--', lw=2)
        system.plot_trajectory(x_hat, n_rows=n_rows,
                               row_index=row_index, color='red', ls='--', lw=2)
        sample_observed_indices = observed_indices[n_processes_train+process_index]
        system.plot_observations(
            y, sample_observed_indices, n_rows=n_rows, row_index=row_index)


plt.figure(figsize=(12, 6))



trained_parameter_state, loss_values, error_values = train_multi(
    model,
    parameter_state,
    X_train_multi,
    HTY_train_multi,
    HTH_train_values,
    n_epochs,
    batch_size,
    test_function=evaluate_model,
    lr=lr,
    plot_reconstructions=plot_reconstructions
    )
model_name = (f'{model.assimilation}_{n_epochs}')
# model_name = (f'noisy')
path = os.path.abspath(f'output/checkpoints/pendulum/{model_name}')
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path, trained_parameter_state)

# gaussian_error = optax.l2_loss(X_gaussian_test[:, :, 0], X_test_multi[:, :, -1]).mean()

plt.subplot(2, 1, 1)
plt.plot(loss_values)
plt.yscale('log')
plt.subplot(2, 1, 2)
# for iteration in 
plt.yscale('log')
for blur_index in range(n_interpolations+1):
    plt.plot(np.array(error_values)[:, blur_index], color='red', alpha=1-0.9*blur_index/n_interpolations)
# plt.plot(np.array(error_values)[:, 0])
# plt.axhline(gaussian_error, ls='--', color='blue')
plt.show()

