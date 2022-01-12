from Environment.EntropyMinimization import BaseEntropyMinimization

import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

seed = 777

np.random.seed(seed)
random.seed(seed)

# train

navigation_map = np.genfromtxt('./ypacarai_map_middle.csv')
initial_position = [[30, 38]]

svr = SVR(C = 1E4, gamma=10)
kernel = C(1) * RBF(length_scale=10)
regresor = GaussianProcessRegressor(kernel=kernel,optimizer=None)

def compute_MSE(x_sampled, y_sampled, x_all, y_true):

	regresor.fit(x_sampled, y_sampled.squeeze(axis=1))

	y_pred = regresor.predict(x_all)

	return mse(y_true = y_true, y_pred = y_pred, squared=True)



# Create Parallel Environment

KSIZE = np.arange(2,8)
H = []
MSE = []
DISTANCE = np.linspace(100, 400, 10)
for k in tqdm(KSIZE):
	for d in tqdm(DISTANCE):

		environment_args = {'navigation_map': navigation_map,
		                    'number_of_agents': 1,
		                    'initial_positions': initial_position,
		                    'movement_length': 3,
		                    'density_grid': 0.15,
		                    'noise_factor': 1E-5,
		                    'lengthscale': k,
		                    'initial_seed': 0,
		                    'collision_penalty': -1,
		                    'max_distance': d,
		                    'number_of_trials': 5,
		                    'random_init_point': False,
		                    'termination_condition': False,
		                    'discrete': True
		                    }
		env = BaseEntropyMinimization(**environment_args)
		Tmean = []
		MSEmean = []

		h_step = []
		mse_step = []

		kernel = C(1) * RBF(length_scale=k)
		regresor = GaussianProcessRegressor(kernel=kernel, optimizer=None)

		for i in range(20):
			env.reset()
			done = False
			action = env.action_space.sample()
			T = 0
			while not done:

				valid = env.valid_action(action)
				while not valid:
					action = env.action_space.sample()
					valid = env.valid_action(action)

				_,_,done,_ = env.step(action)

			Tmean.append(env.trace/env.tr0)
			MSEmean.append(compute_MSE(env.measured_locations,
			                           env.measured_values,
			                           env.visitable_locations,
			                           env.GroundTruth_field[env.visitable_locations[:,0], env.visitable_locations[:,1]]))

		H.append(np.mean(Tmean))
		MSE.append(np.mean(MSEmean))






