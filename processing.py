import numpy as np
from jax import numpy as jnp
from scipy.linalg import solve
from tqdm import tqdm

def build_observation_matrix(observed_indices, d):
    n_obs = len(observed_indices)
    H = np.zeros((n_obs, d))
    for obs_index in range(n_obs):
        H[obs_index, observed_indices[obs_index]] = 1
    return H
    

def normalize_trajectories(X):
    normalized_X = np.zeros_like(X)
    mean = X.mean(axis=0)
    std = np.sqrt(X.var(axis=0))
    normalized_X = (X - mean)/std
    return normalized_X


def rescale_trajectory(x, mean_values, std_values):
    n_dims, T = mean_values.shape
    rescaled_x = x.copy()
    for i in range(n_dims):
        ui_values = x[i::n_dims]
        mean = mean_values[i]
        std = std_values[i]
        rescaled_x[i::n_dims] = ui_values*std + mean
    return rescaled_x

def create_dataset(
        X,
        observed_indices_values,
        rho,
        n_train,
        n_processes_train,
        theoretical_mean=None,
        theoretical_cov=None,
        ):
    n_samples, d = X.shape
    X_train = X[:n_train]
    n_processes = len(observed_indices_values)
    H_values = [] 
    HTY_values = [] 
    HTH_values = [] 
    Y_values = []
    X_gaussian_values = np.zeros((n_samples, n_processes, d))
    posterior_values = np.zeros((n_processes, d, d))
    empirical_mean = np.mean(X_train, axis=0)

    mu_gaussian = theoretical_mean if theoretical_mean is not None else empirical_mean

    empirical_cov = (1/n_train)*(X_train - mu_gaussian).T@(X_train - mu_gaussian)
    cov_gaussian = theoretical_cov if theoretical_cov is not None else empirical_cov
    precision_gaussian = np.linalg.inv(cov_gaussian)

    for process_index, process_observed_indices in tqdm(enumerate(observed_indices_values)):
        n_obs = len(process_observed_indices) 
        H = build_observation_matrix(process_observed_indices, d)
        H_values.append(H)
        Y = X@H.T + rho*np.random.randn(n_samples, n_obs)
        Y_values.append(Y)
        HTY_values.append(Y@H)
        HTH_values.append(H.T@H)

        posterior_precision = precision_gaussian + (1/rho**2)*H.T@H
        K_gaussian = (1/rho**2)*solve(posterior_precision, H.T, assume_a='pos')
        X_gaussian_values[:, process_index, :] = mu_gaussian + (Y-H@mu_gaussian)@K_gaussian.T


    data = {
        # 'train': {
            'n_processes': n_processes,
            'n_processes_train': n_processes_train,
            'n_samples': n_samples,
            'n_train': n_train,
            'X': X,
            'Y': Y_values,
            'HTY': jnp.stack(HTY_values).transpose((1, 0, 2)),
            'X_gaussian': jnp.array(X_gaussian_values),
            'H_values': H_values,
            'HTH_values': jnp.stack(HTH_values),
            'rho': rho,
            'observed_indices': observed_indices_values,
    }
    return data

def interpolate(X, Z, blur_values, blur_max):
    batch_size, n_processes, d = X.shape
    n_blur = len(blur_values)
    X_interpolation = np.zeros((batch_size, n_processes, n_blur, d))
    for iteration, blur in enumerate(blur_values):
        weight = (blur/blur_max)
        interpolation = weight *Z + (1-weight)*X
        X_interpolation[:, :, iteration, :] = interpolation
    return jnp.array(X_interpolation)

def add_noise(X, posterior_values, blur_values):
    batch_size, n_processes, n_interpolations, d = X.shape 
    n_blur = len(blur_values)
    mean = np.zeros(d)
    noisy_X = np.zeros_like(X)
    for process_index in range(n_processes):
        posterior = posterior_values[process_index]
        noise = np.random.multivariate_normal(mean, posterior, size=(batch_size, n_interpolations))
        for blur_index in range(n_blur):
            blur = blur_values[blur_index]
            noise[:, blur_index, :] *= blur**2
        noisy_X[:, process_index] = X[:, process_index] + noise
    return jnp.array(noisy_X)

