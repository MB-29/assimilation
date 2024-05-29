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

from systems.lorenz import Lorenz as System

np.random.seed(0)


T, dt = 32, 0.05
system = System(T, dt)


d = system.d
n_samples = 1_000
n_samples = 2_000
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

sample_observed_indices = observed_indices[0]
y = Y_values[0][0]
x = X[0]
x_gaussian = X_gaussian[0][0]


X_test = X[n_train:]
X_init_test = X_gaussian.transpose(1, 0, 2)[n_train:, n_processes_train:]
HTY_test_multi = HTY_values[n_train:, n_processes_train:]
HTH_test_values = HTH_values[n_processes_train:]

print('model')
HTy_init = jnp.zeros((1, 1, d))
x_init = jnp.zeros((1, 1, 1, d))
HH_init = jnp.zeros((1, d, d))
blur_init = jnp.array([1])
L0 = jnp.eye(d)
# HH = jnp.diagonal(H.T@H)

n_interpolations = 5
model = NeuralAssimilation(d, rho, n_interpolations)
key = random.key(0)
parameter_state = model.init(key, x_init, HTy_init, blur_init, HH_init)


model_name = (f'unconditional')
model_name = (f'autocorrelation')
# model_name = (f'noisy')
model_name = (f'conditional_1000')
model_name = (f'conditional_1000')
model_name = (f'conditional_300')

path = os.path.abspath(f'output/checkpoints/lorenz/{model_name}')
checkpointer = ocp.StandardCheckpointer()
 
abstract_parameter_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, parameter_state)
trained_parameter_state = checkpointer.restore(
    path,
    # target = tree
    args=ocp.args.StandardRestore(abstract_parameter_state)
)



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


test_index, process_index = 105, 4
test_index, process_index = 105, 0
test_index, process_index = 20, 0
test_index, process_index = 18, 0
# test_index, process_index = 100, 0
n_sampling = 5
sample_values = np.zeros((n_sampling, n_interpolations+1, d))
for sample_index in range(n_sampling):
    reconstruction_values = model.sample(trained_parameter_state, X_init_test, HTY_test_multi, HTH_test_values, test_index, process_index)
    sample_values[sample_index] = reconstruction_values
def plot_reconstructions(X_hat_values, **kwargs):
    # iteration_indices = np.arange(0, n_interpolations, 5)
    iteration_indices = np.arange(n_interpolations) 
    n_rows = len(iteration_indices)
    # n_rows = 1
    for row_index, iteration in enumerate(iteration_indices):
    # for row_index, iteration in enumerate([n_interpolations-1]):
        blur_index = n_interpolations-iteration-1
        x_hat = X_hat_values[blur_index, test_index, process_index]
        reconstruction_samples = sample_values[:, blur_index]
        x = X_test[test_index]
        y = Y_values[n_processes_train+process_index][n_train+test_index]
        x_gaussian = X_init_test[test_index, process_index]
        
        system.plot_trajectory(x, n_rows=n_rows,
                               row_index=row_index, color='black', lw=2)
        system.plot_trajectory(x_gaussian, n_rows=n_rows,
                               row_index=row_index, color='blue', ls='--', lw=2)
        system.plot_trajectory(x_hat, n_rows=n_rows,
                               row_index=row_index, color='red', ls='--', lw=2)
        for sample_index in range(n_sampling):
            sample = reconstruction_samples[sample_index]
            system.plot_trajectory(sample, n_rows=n_rows,
                                row_index=row_index, color='green', alpha=.4, lw=2)

        sample_observed_indices = observed_indices[n_processes_train+process_index]
        system.plot_observations(
            y, sample_observed_indices, n_rows=n_rows, row_index=row_index)
        
# gaussian_error = optax.l2_loss(X_gaussian_test[:, :, 0], X_test_multi[:, :, -1]).mean()

X_hat_values, test_error_values = evaluate_model(model, trained_parameter_state)

plot_reconstructions(X_hat_values)
plt.show()
