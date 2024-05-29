import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from jax import numpy as jnp
import pickle
from scipy.linalg import dft

from systems.base_system import BaseSystem
from systems.spring import Spring
from processing import build_observation_matrix, create_dataset

np.random.seed(1)

omega0_sq = 2.
alpha = 0.1

class Pendulum(BaseSystem):
    n_dims = 2

    name = 'pendulum'

    def __init__(self, T, dt) -> None:
        super().__init__(T)
        self.dt = dt
        self.time_values  = dt*np.arange(0, T)


    def integrate(self, state):
        trajectory = odeint(pendulum_dynamics, state, self.time_values)
        return trajectory


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
                q_test = X_test_blurred[5+test_index][iteration][::2]
                p_test = X_test_blurred[5 +
                                         test_index][iteration][1::2]
                q_hat = x_hat[::2]
                p_hat = x_hat[1::2]

                plt.subplot(n_test, n_snaps,
                            (test_index*n_snaps)+column_index+1)
                plt.plot(q_test, p_test, color='black')
                plt.plot(q_hat, p_hat, **kwargs, ls='--')

                # plt.xlim((-15, 15))
                # plt.ylim((-15, 15))
                # plt.xlim((-2, 2))
                # plt.ylim((-2, 2))
    def plot_trajectory(self, x, n_rows=1, row_index=0, **kwargs):
        ylabel_values = [r'angle', r'momentum']
        for dim in range(self.n_dims):
            plt.subplot(n_rows, self.n_dims+1, row_index*(self.n_dims+1)+dim+1)
            plt.plot(self.time_values, x[dim::self.n_dims], **kwargs)
            plt.xticks([])
            plt.xlabel(r'time')
            plt.yticks([])
            plt.ylabel(ylabel_values[dim])

        plt.subplot(n_rows, self.n_dims+1, row_index *
                    (self.n_dims+1) + self.n_dims+1)
        plt.plot(x[::self.n_dims], x[1::self.n_dims], **kwargs)
        plt.xticks([])
        plt.xlabel(r'angle')
        plt.yticks([])  
        plt.ylabel(r'momentum')
def pendulum_dynamics(u, t):
    q, p = u
    q_dot = p
    p_dot = -omega0_sq*np.sin(q) - alpha*p
    u_dot = np.array([q_dot, p_dot])
    return u_dot


if __name__ == '__main__':

    n_samples = 500
    n_train = 400

    T = 100
    dt = 0.1

    system = Pendulum(T, dt)

    # u0_values = np.array([2.5, -1.]) + .1*np.random.randn(n_samples, 2)*np.array([1, 0.1])
    u0_values = np.array([0., 0.]) + np.random.uniform(-1., 1., size=(n_samples, 2))*np.array([3*np.pi/4, 1.])

    X = system.generate_trajectories(u0_values)
    # normalized_X = normalize(X)

    plt.figure(figsize=(16, 4))
    system.plot_trajectory(X[1], color='blue')
    plt.show()

    obs_noise_size = 0.001

    u1_values = X[:, ::2]
    u2_values = X[:, 1::2]

    plt.plot(u1_values.T, u2_values.T, color='blue', alpha=.6, lw=1)
    plt.show()

    n_obs = T//10
    n_obs = 10
    n_processes_train = T//n_obs
    n_processes = 2*n_processes_train

    model = Spring(T, dt, omega_sq=omega0_sq, alpha=alpha)
    mean, generator = model.compute_generators()
    permutation = np.zeros((system.d, system.d))
    for t in range(T):
        permutation[2*t, t] = 1
        permutation[2*t+1, T+t] = 1 
    generator = permutation@generator
    theoretical_cov = generator@generator.T

    observed_indices_values = 2*np.random.choice(T, size=(n_processes, n_obs))
    observed_indices_values = np.array([(np.arange(n_obs)*(n_processes)*2 + 2*i) % (2*T) for i in range(n_processes)])

    data = create_dataset(
        X,
        observed_indices_values,
        obs_noise_size,
        n_train,
        n_processes_train,
        # normalize=False,
        theoretical_mean=mean,
        theoretical_cov=theoretical_cov
        )
    
    X = data['X']
    X_gaussian = data['X_gaussian']
    Y = data['Y']
    H = data['H_values']
    rho = data['rho']
    observed_indices_train = data['observed_indices']
    observed_indices = observed_indices_train[0]
    y = Y[0][0]
    x = X[0]
    x_gaussian = X_gaussian[0][0]
    plt.figure(figsize=(16, 4))
    system.plot_trajectory(x, color='black')
    system.plot_trajectory(x_gaussian, color='blue', ls='--')
    system.plot_observations(y, observed_indices)
    plt.show()

    file_name = f'data/pendulum_multi_d={system.d}_n_samples={n_samples}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
