from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import copy
import pandas

class metric_constructor:

	def __init__(self, path):

		self.path = path
		# Metrics
		self.R = []
		self.DH = []
		self.A = []
		self.XI = []
		self.DI = []
		self.MSE_GP = []
		self.MSE_SVR = []

		kernel = C(1.0) * RBF(length_scale=5, length_scale_bounds=(1e-7, 1e7)) + WhiteKernel(noise_level=0.00001,noise_level_bounds=(1e-7, 1e7))
		self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1E-5)
		self.svr = SVR(C=1E5)
		self.data_vector = []


	def record_new(self):

		self.R = []
		self.DH = []
		self.A = []
		self.XI = []
		self.DI = []
		self.MSE_GP = []
		self.MSE_SVR = []

	def record_step(self, r, dh, a, xi, di, meas_locs, meas_vals, visitable_locs, true_vals):

		self.R.append(r)
		self.DH.append(dh)
		self.A.append(a)
		self.XI.append(xi)
		self.DI.append(di)

		# Fit and Obstain model #
		self.gp.fit(meas_locs[-60:], meas_vals[-60:])
		self.svr.fit(meas_locs[-60:], meas_vals[-60:])
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


