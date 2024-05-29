import numpy as np
import matplotlib.pyplot as plt
import pickle
from jax import numpy as jnp

from systems.base_system import BaseSystem
from processing import build_observation_matrix, pad

class Spring(BaseSystem):

    n_dims = 2

    name = 'spring' 

    q0 = 0
    p0 = 0

    # sigma_q0 = .5
    # sigma_p0 = 0.2
    # sigma_dp = 0.1
    # sigma_dq = 0.05

    sigma_q0 = 1.
    sigma_p0 = 0.1
    sigma_dp = 0.01
    sigma_dq = 0.01


    def __init__(self, T, dt, omega_sq, alpha) -> None:
        print(f'T = {T}, dt = {dt}, omega = {omega_sq}, alpha = {alpha}')

        super().__init__(T)
        self.dt = dt
        self.time_values = dt*np.arange(T)
        # self.dt = 0.1
        # self.time_values = self.dt*np.arange(T)

        self.alpha = alpha
        self.omega_sq = omega_sq

    def build_matrices(self):
        qp_shift = np.diag((np.ones(self.d-1)), k=-1)
        qp_shift[self.T, :] = 0

        shift = np.diag((np.ones(self.T-1)), k=-1)



        dynamics_blocks = [[np.zeros((self.T, self.T)), shift],
                        [-self.omega_sq*shift, -self.alpha*shift]]
        dynamics_matrix = self.dt*np.block(dynamics_blocks)

        constraint = np.eye(self.d) - qp_shift - dynamics_matrix



        noise_weights = np.zeros((self.d, self.d))
        noise_weights[:self.T, :self.T] = self.dt*self.sigma_dq*np.diag((np.ones(self.T)))
        noise_weights[self.T:, self.T:] = self.dt*self.sigma_dp*np.diag((np.ones(self.T)))
        noise_weights[self.T, self.T] = self.sigma_p0
        noise_weights[0, 0] = self.sigma_q0

        return constraint, noise_weights

    
    def compute_generators(self):


        constraint, noise_weights = self.build_matrices()

        inv_constraint = np.linalg.inv(constraint)

        self.A = inv_constraint
        L = inv_constraint @ noise_weights
        x0 = np.zeros(self.d)
        x0[0] = self.q0
        x0[self.T] = self.p0
    # x = np.linalg.solve(generator, noise + initial_conditions)
        x_mean = self.A@x0

        return x_mean, L
    
    def generate(self, n_samples, seed=0):
        np.random.seed(seed)


        initial_conditions = np.zeros((n_samples, self.d))
        initial_conditions[:, 0] = self.q0
        initial_conditions[:, self.T] = self.p0

        self.x_mean, self.L = self.compute_generators()

        np.random.seed(seed)

        constraint, noise_weights = self.build_matrices()

        inv_constraint = np.linalg.inv(constraint)

        self.A = inv_constraint
        self.L = inv_constraint @ noise_weights
    # x = np.linalg.solve(generator, noise + initial_conditions)
        self.x_mean = (self.A@initial_conditions.T).T

        gaussian = np.random.randn(n_samples, self.d)
        variability = (self.L@gaussian.T).T
        X = self.x_mean + variability

        return X
    
    def test_plot(self, X_hat, X_test, Y_test, H, mu_gaussian, K_gaussian, test_index=0, **kwargs):

        # T = d//3
        n_test = 3

        X_hat_gaussian = mu_gaussian + (Y_test-H@mu_gaussian)@K_gaussian.T
        # n_snaps = len(iteration_values)

        for test_index in range(n_test):
            x_hat = X_hat[5+test_index]
            x_hat_gaussian = X_hat_gaussian[5+test_index]
            q_test = X_test[5+test_index][:self.T]
            p_test = X_test[5+test_index][self.T:2*self.T]
            q_hat = x_hat[:self.T]
            p_hat = x_hat[self.T:2*self.T]
            q_hat_gaussian = x_hat_gaussian[:self.T]
            p_hat_gaussian = x_hat_gaussian[self.T:2*self.T]

            plt.subplot(1, n_test, test_index+1)
            plt.plot(q_test, p_test, color='black')
            plt.plot(q_hat, p_hat, **kwargs, ls='--')
            plt.plot(q_hat_gaussian, p_hat_gaussian, color='blue', ls='--')

            plt.xlim((-1, 1))
            plt.ylim((-1, 1))


if __name__ == '__main__':
    T = 1000
    n_samples = 1200
    system = Spring(T)
    d = system.d

    X = system.generate(n_samples)
    n_obs = T//20


    theoretical_cov = system.L@system.L.T
    empirical_cov = (1/n_samples)*(X-system.x_mean).T@(X-system.x_mean)
    q = X[:, :T]

    n_train = n_samples//3
    X_train = X[:n_train]
    X_test = X[n_train:]


    plt.plot(system.time_values, q.T, color='blue', alpha=0.3)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(theoretical_cov)
    plt.subplot(1, 2, 2)
    plt.imshow(empirical_cov)
    plt.show()


    index_values = [
        np.arange(0, T, T//n_obs)[:n_obs],
        # np.arange(25, T+25, 5),
        # np.arange(100, T+100, 5),
        # np.arange(0, 2*T, 10),
    ]
    n_H = len(index_values)

    H_values = []
    X_train_values = []
    X_test_values = []
    Y_train_values = []
    Y_test_values = []
    r = 1e-2
    r_values = np.full(n_H, r)
    # r = 0
    for observed_indices in index_values:
        assert n_obs == len(observed_indices)
        H = build_observation_matrix(observed_indices, d)
        H_values.append(H)
        
        Y_train = X_train@H.T + r*np.random.randn(n_train, n_obs)
        X_train_values.append(X_train)
        Y_train_values.append(Y_train)

        Y_test = X_test@H.T + r*np.random.randn(n_samples - n_train, n_obs)
        X_test_values.append(X_test)
        Y_test_values.append(Y_test)

    data = {
        # 'n_obs': n_obs,
        'train': {
            'X': X_train,
            'Y': Y_train,
            'H': H,
            'r': r,
            'cov': jnp.array(theoretical_cov),
            'mean': jnp.array(system.x_mean)
        },
        'test': {
            'X': X_test,
            'Y': Y_test,
            # 'H_values': jnp.array(H_values),
            # 'r_values': jnp.array(r_values)
        }
    }
    file_name = f'data/spring_d={d}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

    # x_blank = pad(Y_train_values[0][0], index_values[0], d)
    q = X_test[20][:T]
    x_blank = H.T@Y_test[20]
    q_blank = x_blank[:T]
    plt.plot(q, color='black')
    plt.plot(q_blank, color='blue')
    plt.show()
