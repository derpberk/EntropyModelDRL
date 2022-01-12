from scipy.linalg import cholesky, cho_solve, solve_triangular, LinAlgError
import numpy as np

def conditioning_cov_matrix(X, X_train, kernel, alpha=1E-5):
	"""

	:param X: Base points
	:param X_train: Posterior points
	:param kernel: Kernel
	:return:
	"""

	K_xy = kernel(X, X_train)
	K_yy = kernel(X_train)
	K_yy[np.diag_indices_from(K_yy)] += alpha
	L = cholesky(K_yy, lower=True)
	V = cho_solve((L, True), K_xy.T)
	cov = kernel(X) - K_xy.dot(V)

	return cov

def conditioning_std(X, X_train, kernel, alpha=1E-5):
	K_xy = kernel(X, X_train)
	K_yy = kernel(X_train)
	K_yy[np.diag_indices_from(K_yy)] += alpha
	L = cholesky(K_yy, lower=True)
	L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
	K_inv = L_inv.dot(L_inv.T)
	y_var = kernel.diag(X)
	y_var -= np.einsum("ij,ij->i", np.dot(K_xy, K_inv), K_xy)

	return y_var

def conditioning_cov_matrix_with_time(X, X_train, kernel, sample_times, time, weights, alpha=1E-5):
	"""

	:param sample_times:
	:param time:
	:param alpha:
	:param X: Base points
	:param X_train: Posterior points
	:param kernel: Kernel
	:return:
	"""

	K_xy = kernel(X, X_train)
	K_yy = kernel(X_train)
	K_xx = kernel(X)


	K_yy[np.diag_indices_from(K_yy)] += weights*(time - weights*sample_times.squeeze(1)) ** 2

	K_yy[np.diag_indices_from(K_yy)] += alpha
	L = cholesky(K_yy, lower=True)
	V = cho_solve((L, True), K_xy.T)
	cov = K_xx - K_xy.dot(V)

	return cov

def conditioning_std_with_time(X, X_train, kernel, sample_times, time, weights, alpha=1E-5):
	K_xy = kernel(X, X_train)
	K_yy = kernel(X_train)
	K_yy[np.diag_indices_from(K_yy)] += alpha

	K_yy[np.diag_indices_from(K_yy)] += weights*(time - sample_times.squeeze(1)) ** 2

	L = cholesky(K_yy, lower=True)
	L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
	K_inv = L_inv.dot(L_inv.T)
	y_var = kernel.diag(X)
	y_var -= np.einsum("ij,ij->i", np.dot(K_xy, K_inv), K_xy)

	return y_var