import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization
from EnvironmentUtils import AStarPlanner
from utils import plot_trajectory

seed = 232

np.random.seed(seed)
random.seed(seed)

# train
navigation_map = np.genfromtxt('../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]

""" ----------------------- EXPERIMENT SETUP ----------------------- """

environment_args = {'navigation_map': navigation_map,
                    'number_of_agents': 1,
                    'initial_positions': initial_position,
                    'movement_length': 3,
                    'density_grid': 0.2,
                    'noise_factor': 1E-5,
                    'lengthscale': 5,
                    'initial_seed': 0,
                    'collision_penalty': -1,
                    'max_distance': 200,
                    'number_of_trials': 5,
                    'number_of_actions': 8,
                    'random_init_point': True,
                    'termination_condition': False
                    }

env = BaseEntropyMinimization(**environment_args)
env.reset()
planner = AStarPlanner(env.navigation_map, 1, 0.5)
# N executions of the algorithm
N = 10


def greedy_std_action(std_map, vehicle_position, angle_set):
	highest_std_location = np.array(np.unravel_index(np.argmax(std_map), shape=std_map.shape))
	vector = highest_std_location - vehicle_position
	angle = np.arctan2(vector[1], vector[0])
	angle_diff = np.abs(angle_set - angle)
	return np.argmin(angle_diff)


def reverse_action(a):
	return (a + 4) % 8


for i in range(N):

	s = env.reset()
	env.render()
	a = env.action_space.sample()
	new_a = a
	done = False

	goal_achieved = True
	T = 0
	while not done:

		T += 1
		if goal_achieved:
			highest_std_location = np.array(np.unravel_index(np.argmax(s[-1]), shape=s[-1].shape))
			path = np.asarray(planner.planning(list(highest_std_location), list(env.fleet.vehicles[0].position))).T
			sliced_path = path[3::3]
			path_index = 0
			goal_achieved = False

		s, r, done, _ = env.step_to_position(sliced_path[path_index])
		print(T)
		path_index += 1

		if path_index == len(sliced_path):
			goal_achieved = True

		env.render()

"""
	plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)
"""
