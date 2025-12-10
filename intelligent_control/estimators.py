# estimators.py
import numpy as np

def qrd_rls_first_order(y, u, ff, delta):
    N = len(y)
    n_params = 2
    R = np.sqrt(delta) * np.eye(n_params)
    z = np.zeros(n_params)
    th = np.zeros((N, n_params))

    for k in range(N):
        y1 = y[k - 1] if k - 1 >= 0 else 0.0
        u1 = u[k - 1] if k - 1 >= 0 else 0.0
        phi = np.array([-y1, u1])

        A = np.vstack((np.sqrt(ff) * R, phi.reshape(1, -1)))
        b = np.hstack((np.sqrt(ff) * z, np.array([y[k]])))

        Q, Rn = np.linalg.qr(A, mode='reduced')
        z_new = Q.T @ b
        R = Rn[:n_params, : ]
        z = z_new[:n_params]

        th[k, :] = np.linalg.solve(R, z)
    return th

def generate_data(n_samples, Ts=0.01):
    alpha_real, beta_real = 5.0, 15.0
    a_disc = 1.0 - (alpha_real * Ts)
    b_disc = beta_real * Ts
    u = np.zeros(n_samples)
    y = np.zeros(n_samples)
    current_u = 1.0
    
    for k in range(n_samples):
        if np.random.rand() < 0.05: current_u *= -1
        u[k] = current_u

    for k in range(1, n_samples):
        y[k] = a_disc * y[k - 1] + b_disc * u[k - 1] + np.random.normal(0, 0.05)
    return y, u, a_disc, b_disc, Ts