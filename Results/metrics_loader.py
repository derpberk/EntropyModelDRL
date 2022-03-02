from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from scipy.interpolate import RBFInterpolator
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas

class metric_constructor:

	def __init__(self, path, temporal = False):

		self.path = path
		# Metrics
		self.R = []
		self.DH = []
		self.A = []
		self.XI = []
		self.DI = []
		self.MSE_GP = []
		self.MSE_SVR = []
		self.temporal = temporal


		if not self.temporal:
			kernel = C(1.0) * RBF(length_scale=5.0, length_scale_bounds=(1e-7, 1e7)) + WhiteKernel(noise_level=0.00001,noise_level_bounds=(1e-7, 1e7))
		else:
			kernel = C(1.0) * RBF(length_scale=[5.0, 5.0, 1], length_scale_bounds=(1e-7, 1e7)) + WhiteKernel(noise_level=0.00001, noise_level_bounds=(1e-7, 1e7))

		self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1E-5)
		self.svr = SVR(C=1E6)
		self.data_vector = []


	def record_new(self):

		self.R = []
		self.DH = []
		self.A = []
		self.XI = []
		self.DI = []
		self.MSE_GP = []
		self.MSE_SVR = []

	def record_step(self, r, dh, a, xi, di, meas_locs, meas_vals, visitable_locs, true_vals, horizon=0, sample_times=None):

		self.R.append(r)
		self.DH.append(dh)
		self.A.append(a)
		self.XI.append(xi)
		self.DI.append(di)

		# Fit and Obstain model #
		if self.temporal:

			expanded_meas_locs = np.hstack((meas_locs[-horizon:], sample_times[-horizon:]))

			self.gp.fit(expanded_meas_locs, meas_vals[-horizon:])
			self.svr.fit(expanded_meas_locs, meas_vals[-horizon:])

			expanded_visitable_locs = np.hstack((visitable_locs, sample_times[-1] * np.ones(shape=(len(visitable_locs), 1))))

			self.MSE_GP.append(mean_squared_error(y_true=true_vals, y_pred=self.gp.predict(expanded_visitable_locs)))
			self.MSE_SVR.append(mean_squared_error(y_true=true_vals, y_pred=self.svr.predict(expanded_visitable_locs)))

		else:

			self.gp.fit(meas_locs[horizon:], meas_vals[horizon:])
			self.svr.fit(meas_locs[horizon:], meas_vals[horizon:])

			self.MSE_GP.append(mean_squared_error(y_true=true_vals, y_pred=self.gp.predict(visitable_locs)))
			self.MSE_SVR.append(mean_squared_error(y_true=true_vals, y_pred=self.svr.predict(visitable_locs)))



	def record_finish(self, t):

		data = {'Run': t,
		        'Reward': self.R,
		        'Entropy': self.DH,
		        'A_vector': self.A,
		        'DetectionRate': self.XI,
		        'Distance': self.DI,
		        'MSE_GP': self.MSE_GP,
		        'MSE_SVR': self.MSE_SVR}

		dataframe = pandas.DataFrame.from_dict(data)

		self.data_vector.append(dataframe)

	def record_save(self, path=None):

		if path is None:
			path = self.path

		dataframe_final = pandas.concat(self.data_vector, ignore_index=True)
		dataframe_final.to_csv(path)


