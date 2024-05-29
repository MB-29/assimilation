# Neural Incremental Data Assimilation

Neural Incremental Data Assimilation method for learning a neural data assimilation operator, built with JAX.

## Example

```python
import numpy as np
from models.neural_assimilation import NeuralAssimilation
from training import training_loop

n_epochs = 300
batch_size = 32
loss_values = np.zeros(n_epochs)
error_values = []
lr = 1e-3


def evaluate_model(model, parameter_state):
    X_hat_values = model.reconstruct(
        parameter_state, Z_test, Y_test, H_test)
    X_test_multi = jnp.array(
        [X_test for i in range(n_processes_test)]).transpose(1, 0, 2)
    test_error_values = [optax.l2_loss(
        X_hat.squeeze(), X_test_multi).mean() for X_hat in X_hat_values]
    return X_hat_values, test_error_values

d = 32 #state dimension
rho = 0.01 #observation noise size
n_increments = 3
model = NeuralAssimilation(d, rho, n_increments)

trained_parameter_state, loss_values, error_values = training_loop(
    model,
    X_train, # states, of shape (N, d)
    Z_train, # state estimates, of shape (N, n_processes, d)
    Y_train, # reshaped observation = H.T@y, of shape (N, n_processes, d)
    H_train, # reshaped observation matrix = H@H.T, of shape (n_processes, d, d)
    n_epochs,
    batch_size,
    test_function=evaluate_model,
    lr=lr,
)

```