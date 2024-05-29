import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp

from models.neural import NeuralAssimilation as Model
# from models.unconditional import Unconditional as Model

from training import train_multi
from processing import interpolate, build_observation_matrix

from systems.lorenz import Lorenz as System

np.random.seed(0)


T, dt = 32, 0.05
system = System(T, dt)


d = system.d
n_samples = 2_000
string = f'_multi_d={system.d}_n_samples={n_samples}'
data = system.load_data(string=string)


X = data['X']
X_gaussian = data['X_gaussian']
Y_values = data['Y']
HTY_values = data['HTY']
H_values = data['H_values']
HTH_values = data['HTH_values']
posterior_values = data['posterior_values']
rho = data['rho']
observed_indices = data['observed_indices']
n_samples = data['n_samples']
n_train = data['n_train']
n_processes = data['n_processes']
n_processes_train = data['n_processes_train']
n_processes_test = n_processes - n_processes_train



n_interpolations = 3
blur_values = np.arange(n_interpolations+1)
X_multi = jnp.array([interpolate(X, X_gaussian[index], n_interpolations) for index in range(n_processes)])
X_multi = X_multi.transpose(1, 0, 2, 3)

deblurring_values = blur_values[1:]

X_train_multi = X_multi[:n_train, :n_processes_train] 

HTY_train_multi = HTY_values[:n_train, :n_processes_train]
HTH_train_values = HTH_values[:n_processes_train]

X_test = X[n_train:]
X_init_test = X_gaussian.transpose(1, 0, 2)[n_train:, n_processes_train:]

HTY_test_multi = HTY_values[n_train:, n_processes_train:]
HTH_test_values = HTH_values[n_processes_train:]




# HH = jnp.diagonal(H.T@H)

model = Model(d, rho, n_interpolations)

print('training')

n_epochs = 300
# n_epochs = 200
# n_epochs = 40
# n_epochs = 1_000
batch_size = 32
# batch_size = 64
n_grad = n_train//batch_size
loss_values = np.zeros(n_epochs)
error_values = []
lr = 5e-4

# @jax.jit
def evaluate_model(model, parameter_state):
    X_hat_values = model.reconstruct_multi(
        parameter_state, X_init_test, HTY_test_multi, HTH_test_values)
    X_test_multi = jnp.array([X_test for i in range(n_processes_test)]).transpose(1, 0, 2)
    test_error_values = [optax.l2_loss(
        X_hat.squeeze(), X_test_multi).mean() for X_hat in X_hat_values]
    # test_error_values = [optax.l2_loss(
    #     X_hat[:, 0].squeeze(), X_test[:, 0]).mean() for X_hat in X_hat_values]

    return X_hat_values, test_error_values


test_index, process_index = 45, 0
x = X_test[test_index]
y = Y_values[n_processes_train+process_index][n_train+test_index]
sample_observed_indices = observed_indices[n_processes_train+process_index]
H = build_observation_matrix(sample_observed_indices, d)
x_gaussian = X_init_test[test_index, process_index]
def plot_reconstructions(X_hat_values, **kwargs):
    # iteration_indices = np.arange(0, n_interpolations, 5)
    iteration_indices = np.arange(n_interpolations)
    n_rows = len(iteration_indices)
    for row_index, iteration in enumerate(iteration_indices):
        blur_index = n_interpolations-iteration-1
        x_hat = X_hat_values[blur_index, test_index, process_index]
        system.plot_trajectory(x, n_rows=n_rows,
                               row_index=row_index, color='black', lw=2)
        system.plot_trajectory(x_gaussian, n_rows=n_rows,
                               row_index=row_index, color='blue', ls='--', lw=2)
        system.plot_trajectory(x_hat, n_rows=n_rows,
                               row_index=row_index, color='red', ls='--', lw=2)
        system.plot_observations(
            y, sample_observed_indices, n_rows=n_rows, row_index=row_index)


trained_parameter_state, loss_values, error_values = train_multi(
    model,
    X_train_multi,
    HTY_train_multi,
    HTH_train_values,
    n_epochs,
    batch_size,
    test_function=evaluate_model,
    lr=lr,
    plot_reconstructions=plot_reconstructions
    )

# model_name = (f'unconditional')
# model_name = (f'{model.assimilation}_{n_epochs}')
# # model_name = (f'noisy')
# path = os.path.abspath(f'output/checkpoints/lorenz/{model_name}')
# checkpointer = ocp.StandardCheckpointer()
# checkpointer.save(path, trained_parameter_state)

plt.subplot(2, 1, 1)
plt.plot(loss_values)
plt.yscale('log')
plt.subplot(2, 1, 2)
# for iteration in 
plt.yscale('log')
for blur_index in range(n_interpolations+1):
# blur_index = 0
    plt.plot(np.array(error_values)[:, blur_index], color='red', alpha=1-0.9*blur_index/n_interpolations)
# plt.plot(np.array(error_values)[:, 0])
# plt.axhline(gaussian_error, ls='--', color='blue')
plt.show()

