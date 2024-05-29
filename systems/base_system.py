import numpy as np
import pickle
import matplotlib.pyplot as plt

class BaseSystem:

    def __init__(self, T) -> None:
        self.T = T
        self.d = self.n_dims*T
    
    def model_iteration(self, state, time_index):
        return NotImplementedError
    
    def generate_trajectories(self, u0_values, **kwargs):
        n_samples, n_dims = u0_values.shape 
        trajectory_values = np.zeros((n_samples, self.T, self.n_dims))
        for sample_index in range(n_samples):
            u0 = u0_values[sample_index]
            u = u0.copy()
            trajectory = self.integrate(u, **kwargs)
            trajectory_values[sample_index] = trajectory
        return trajectory_values.reshape((n_samples, self.d))




    def plot_trajectory(self, x, n_rows=1, row_index=0, **kwargs):
        for dim in range(self.n_dims):
            plt.subplot(n_rows, self.n_dims+1, row_index*(self.n_dims+1)+dim+1)
            plt.plot(self.time_values, x[dim::self.n_dims], **kwargs)

        plt.subplot(n_rows, self.n_dims+1, row_index *
                    (self.n_dims+1) + self.n_dims+1)
        plt.plot(x[::self.n_dims], x[1::self.n_dims], **kwargs)
    def plot_observations(self, observations, observed_indices, n_rows=1, row_index=0, **kwargs):
        plt.subplot(n_rows, self.n_dims+1, row_index*(self.n_dims+1)+1)

        plt.scatter(self.time_values[observed_indices//self.n_dims], observations, marker='x', s=100, **kwargs)

    def load_data(self, string=''):

        file_name = f'data/{self.name}{string}.pkl'

        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        
        # self.n_obs = data['n_obs']

        return data