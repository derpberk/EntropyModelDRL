import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization
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
# N executions of the algorithm
N = 10


def reverse_action(a):
	return (a + 4) % 8


for i in range(N):
	env.reset()

	if random.random() > 0.7:
		if env.fleet.vehicles[0].position[0] > env.navigation_map.shape[0] / 2:
			direction_UPDOWN = 3
		else:
			direction_UPDOWN = 7
		direction_LEFTRIGHT = 1
	else:
		if env.fleet.vehicles[0].position[0] > env.navigation_map.shape[0] / 2:
			direction_UPDOWN = 4
		else:
			direction_UPDOWN = 0
		direction_LEFTRIGHT = 2


	done = False
	action = direction_LEFTRIGHT
	vertical_need_flag = False

	while not done:

		horizontal_valid = env.valid_action(direction_LEFTRIGHT)
		vertical_valid = env.valid_action(direction_UPDOWN)

		# Si estamos retrocediendo porque no podemos subir
		if vertical_valid and vertical_need_flag:
			action = direction_UPDOWN
			vertical_need_flag = False
		else:
			# Si no es valida lateralmente:
			if not horizontal_valid:
				# Comprobamos si es valido verticalmente
				if vertical_valid:
					# Si lo es, tomamos un paso en vertical y cambiamos la direccion
					action = direction_UPDOWN
					direction_LEFTRIGHT = reverse_action(direction_LEFTRIGHT)
				else:
					# Si no lo es, cambiamos la direccion
					direction_LEFTRIGHT = reverse_action(direction_LEFTRIGHT)
					action = direction_LEFTRIGHT
					vertical_need_flag = True
			else:
				action = direction_LEFTRIGHT

		_, r, done, _ = env.step(action)
		env.render()

"""
	plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)
"""