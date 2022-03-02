import numpy as np
import random
from Environment.EntropyMinimization import BaseTemporalEntropyMinimization
from Results.metrics_loader import metric_constructor
from utils import plot_trajectory
import matplotlib.pyplot as plt

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
                    'max_distance': 500,
                    'number_of_trials': 5,
                    'number_of_actions': 8,
                    'random_init_point': True,
                    'termination_condition': False,
                    'dt': 0.03,
                    }

# Create environment #
env = BaseTemporalEntropyMinimization(**environment_args)
env.is_eval = True
env.reset()
# N executions of the algorithm
N = 20

metric_recorder = metric_constructor('../Results/EntropyMinimizationResults/TemporalLawnMowerResults.csv', temporal=True)
draw = False

def reverse_action(a):
	return (a + 4) % 8


for t in range(N):

	env.reset()
	metric_recorder.record_new()

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
	vertical_count = 0

	while not done:

		horizontal_valid = env.valid_action(direction_LEFTRIGHT)
		vertical_valid = env.valid_action(direction_UPDOWN)

		# Si estamos retrocediendo porque no podemos subir
		if vertical_valid and vertical_need_flag:
			action = direction_UPDOWN
			vertical_need_flag = False
			vertical_count = 0

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
					vertical_count += 1
					direction_LEFTRIGHT = reverse_action(direction_LEFTRIGHT)
					action = direction_LEFTRIGHT
					vertical_need_flag = True
			else:
				action = direction_LEFTRIGHT

		if vertical_count >= 2:
			vertical_count = 0

			if direction_UPDOWN > 0.7:
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


		s, r, done, i = env.step(action)

		metric_recorder.record_step(r,
		                            i['Entropy'],
		                            i['Area'],
		                            i['DetectionRate'],
		                            np.sum(env.fleet.get_distances()),
		                            env.measured_locations,
		                            np.asarray(env.measured_values).squeeze(1),
		                            env.visitable_locations,
		                            env.GroundTruth_field[env.visitable_locations[:, 0], env.visitable_locations[:, 1]],
		                            horizon=60,
		                            sample_times=env.sample_times)

	if draw:
		with plt.style.context('seaborn-dark'):
			fig,ax = plt.subplots(1,1)
			ax.imshow(s[-1], cmap='bone', vmin=0, vmax=1,interpolation='bicubic', zorder = 2)
			mask_nav = env.navigation_map
			mask_nav[np.where(mask_nav==1)] = np.nan
			ax.imshow(mask_nav, cmap='gray_r', vmin=0, vmax=1, zorder = 3)
			plt.tick_params(left=False, right=False, labelleft=False,
			                labelbottom=False, bottom=False)

			plot_trajectory(ax, env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
			                z=None,
			                colormap='jet',
			                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
			plt.show(block=True)

	metric_recorder.record_finish(t=t)

metric_recorder.record_save()
