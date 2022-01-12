import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO
import numpy as np
from utils import plot_trajectory

from EntropyMinimization import BaseEntropyMinimization

navigation_map = np.genfromtxt('../../Environment/ypacarai_map.csv')
initial_position = [[67, 73]]
environment_args = {'navigation_map': navigation_map,
	                    'number_of_agents': 1,
	                    'initial_positions': initial_position,
	                    'movement_length': 6,
	                    'density_grid': 0.2,
	                    'noise_factor': 0.01,
	                    'lengthscale': 10,
	                    'initial_seed': 0,
	                    'collision_penalty': -1,
	                    'max_distance': 400}


env = BaseEntropyMinimization(navigation_map=environment_args['navigation_map'],
		                              number_of_agents=environment_args['number_of_agents'],
		                              initial_positions=environment_args['initial_positions'],
		                              movement_length=environment_args['movement_length'],
		                              density_grid=environment_args['density_grid'],
		                              noise_factor=environment_args['noise_factor'],
		                              lengthscale=environment_args['lengthscale'],
		                              initial_seed=environment_args['initial_seed'],
		                              collision_penalty=environment_args['collision_penalty'],
		                              max_distance=environment_args['max_distance'],
		                              )

model = PPO.load(r'G:\Mi unidad\SharedFolder\EntropyModelDRL\Learning\EntropyMinimizationLearning\Results\Baseline_PPO\best_model.zip')

R = 0
d = False
s = env.reset()
env.render()
plt.pause(0.5)

while not d:

	a = model.predict(s, deterministic=False)
	s, r, d, _ = env.step(a[0])
	R += r
	print('Recomensa: ', r)
	# env.render()
	# plt.pause(0.5)

print('Recomensa acumulada: ', R)
env.render()
plot_trajectory(ax=plt.gca(), x =env.waypoints[:, 0, 1], y=env.waypoints[:, 0, 0], num_of_points = 200, plot_waypoints=True)
plt.pause(1000)