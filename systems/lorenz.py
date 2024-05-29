import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from jax import numpy as jnp
import pickle
from scipy.linalg import dft

from systems.base_system import BaseSystem
from processing import build_observation_matrix, create_dataset

np.random.seed(1)

rho, sigma, beta = 28, 10, 8/3


def lorenz_dynamics(u, t):
    x, y, z = u
    x_dot = sigma*(y-x)
    y_dot = x*(rho-z) - y
    z_dot = x*y - beta*z
    u_dot = np.array([x_dot, y_dot, z_dot])
    return u_dot

class Lorenz(BaseSystem):

    # n_obs = T//8
    # d = 3*T
    n_dims = 3

    name = 'lorenz'

    def __init__(self, T, dt) -> None:
        super().__init__(T)
        self.dt = dt
        self.time_values = dt*np.arange(0, T)

    def model_iteration(self, state, time_index, model_noise_size):
        integration_times = self.time_values[time_index:time_index+2]
        trajectory = odeint(lorenz_dynamics, state, integration_times)
        next_state = trajectory[-1]
        noise = model_noise_size*np.random.randn(self.n_dims)
        noisy_next_state = next_state + noise
        return noisy_next_state
    
    def integrate(self, u, **kwargs):
        trajectory = np.zeros((T, self.n_dims)) 
        for t in range(self.T):
                trajectory[t] = u
                # integration_times = time_values[t:t+2]
                # trajectory = odeint(lorenz_dynamics, u, integration_times)
                # noise = np.sqrt(dt)*np.random.randn(3)
                u = self.model_iteration(u, t, **kwargs)
        return trajectory
        # def plot_observations(self, observations, observed_indices):
        #     plt.subplot(1, 4, 1)
            
        #     plt.scatter(self.time_values[observed_indices//3], observations, color='blue', marker='x')
        
    # def plot_trajectory(self, x, n_rows=1, row_index=0, **kwargs):
    #     for dim in range(self.n_dims):
    #         plt.subplot(n_rows, self.n_dims, row_index*(self.n_dims+1)+dim+1)
    #         plt.plot(self.time_values, x[dim::self.n_dims], **kwargs)
    #         plt.xlabel(r'time')
    #         plt.ylabel(fr'$u_{dim+1}$')
    #         plt.xticks([])
    #         plt.yticks([])

    # def plot_observations(self, observations, observed_indices, n_rows=1, row_index=0, **kwargs):
    #     plt.subplot(n_rows, self.n_dims, row_index*(self.n_dims+1)+1)

    #     plt.scatter(self.time_values[observed_indices//self.n_dims],
    #                 observations, marker='x', s=100, **kwargs)

                




if __name__ == '__main__':

    n_samples = 2_000
    n_train = 1_800

    T = 32
    dt = 0.05

    system = Lorenz(T, dt)

    u0_values = np.array([0, 0, 25]) + 10*np.random.randn(n_samples, 3)

    model_noise_size = np.sqrt(dt)
    X = system.generate_trajectories(u0_values, model_noise_size=model_noise_size)
    # normalized_X = normalize(X)

    plt.figure(figsize=(16, 4))
    system.plot_trajectory(X[1], color='blue')
    plt.show()

    obs_noise_size = 0.05


    # u1_values = X[:, ::3]
    # u2_values = X[:, 1::3]
    # u3_values = X[:, 2::3]

    # plt.plot(u1_values.T, u2_values.T, color='blue', alpha=.6, lw=1)
    # plt.show()

    mean_values = np.zeros((system.n_dims, T))
    std_values = np.zeros((system.n_dims, T))
    normalized_X = X.copy()
    for i in range(3):
        ui_values = X[:, i::3]
        mean = ui_values.mean(axis=0)
        std = np.sqrt(ui_values.var(axis=0))
        mean_values[i] = mean
        std_values[i] = std
        normalized_X[:, i::3] = (1/std) * (ui_values - mean)

    n_obs = T//8
    n_processes = 16
    n_processes_train = 8

    
    # observed_indices_train = 3*np.random.choice(T, size=(n_processes_train, n_obs))
    observed_indices_values = np.array([(np.arange(n_obs)*8*3 + 3*i)%(3*T) for i in range(n_processes)])
    # observed_indices_test = np.array([(np.arange(n_obs)*8*3 + 3*i)%(3*T) for i in range(8)])
    # observed_indices_test = np.array([(3*i +3*8*np.arange(n_obs))%T for i in range(5)])
    # observed_indices_values = np.concatenate((observed_indices_train, observed_indices_test), axis=0)
   
    data = create_dataset(
        normalized_X,
        observed_indices_values,
        obs_noise_size,
        n_train,
        n_processes_train,
        )
    
    data['mean_values'] = mean_values
    data['std_values'] = std_values
    
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


    file_name = f'data/lorenz_multi_d={system.d}_n_samples={n_samples}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
