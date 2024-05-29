import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
import pickle 


class Rosenbrock:

    d = 2
    r = 10
    n_obs = 1

    def __init__(self) -> None:
        pass

    def load_data(self, d):
        file_name = f'data/rosenbrock_d={d}.pkl'

        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        
        return data

    def test_plot_blurred(self, X_hat_values, X_test_blurred, test_index=0, **kwargs):

        # T = d//3
        n_test = 3
        n_snaps = 5
        # n_snaps = len(iteration_values)
        n_snaps = len(X_hat_values)
        iteration_values = np.arange(len(X_hat_values))

        for column_index, iteration in enumerate(iteration_values):
            X_hat = X_hat_values[iteration] 
            for test_index in range(n_test):
                x_hat = X_hat[5+test_index]
                x_test = X_test_blurred[5+test_index][iteration]
                
                plt.subplot(n_test, n_snaps, (test_index*n_snaps)+column_index+1)
                plt.scatter(*x_test, color='black')
                plt.scatter(*x_hat, **kwargs, marker='x', ls='--')

                plt.xlim((-1, 1))
                plt.ylim((0, 1))

    def interpolate(self, X, X_target, n_iterations):
        batch_size, d = X.shape
        X_interpolation = np.zeros((batch_size, n_iterations+1, d))
        for iteration in range(n_iterations+1):
            t = iteration/n_iterations

            X_interpolation_iteration = t*X_target + (1-t)*X
            X_interpolation[:, iteration, :] = X_interpolation_iteration
        return jnp.array(X_interpolation)



def model(x):
    equation = x[1] - x[0]**2
    location = 0.1*x[0]
    return jnp.array([location, equation])

if __name__ == '__main__':

    d = 2
    n_obs = 1

    n_samples = 1_000
    n_train = 800

    # u0 = np.array([7., 1., 33.])

    X = np.zeros((n_samples, d))
    y_values = np.zeros((n_samples, n_obs))
    H = np.array([[1, 0]])
    obs_noise_size = 0.1

    X[:, 0] = np.random.randn(n_samples)
    X[:, 1] = X[:, 0]**2

        # y_values[sample_index] = u_values[sample_index, ::8, 0] + obs_noise_size*np.random.randn(n_obs)


    # trajectory = odeint(lorenz_dynamics, u0, time_values)

    # x_values, y_values = trajectory[:, :2].T
    # x_values, y_values = u_values[:, :, :2].T
    # plt.plot(x_values, y_values, color='blue', alpha=.6)
    # X = u_values.reshape((n_samples, d))

    plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=.6, lw=1)
    plt.show()

    Y = X@H.T + obs_noise_size*np.random.randn(n_samples, n_obs)
    X_train, X_test = X[:n_train, ], X[n_train:, ]
    Y_train, Y_test = Y[:n_train, ], Y[n_train:, ]
    r = obs_noise_size

    data = {
        'train': {
            'X': jnp.array(X_train),
            'Y': jnp.array(Y_train),
            'H': jnp.array(H),
            'r': r,
        },
        'test': {
            'X': jnp.array(X_test),
            'Y': jnp.array(Y_test),
            'H': jnp.array(H),
            'r': r
        }
    }
    file_name = f'data/rosenbrock_d={d}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
