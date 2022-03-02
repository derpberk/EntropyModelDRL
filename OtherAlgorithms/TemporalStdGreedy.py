import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization
from EnvironmentUtils import AStarPlanner
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
planner = AStarPlanner(env.navigation_map, 1, 0.5)
# N executions of the algorithm
N = 20


def reverse_action(a):
	return (a + 4) % 8


metric_recorder = metric_constructor('../Results/EntropyMinimizationResults/Temporal_std_greedy.csv', temporal=True)
draw = False
masksize = 10
for t in range(N):

	s = env.reset()
	metric_recorder.record_new()
	done = False
	goal_achieved = True

	while not done:

		mask = np.zeros_like(s[-1])
		mask[int(env.fleet.vehicles[0].position[0])-masksize:int(env.fleet.vehicles[0].position[0])+masksize+1,
		int(env.fleet.vehicles[0].position[1])-masksize:int(env.fleet.vehicles[0].position[1])+masksize+1] = 1
		highest_std_location = np.array(np.unravel_index(np.argmax(s[-1]*mask), shape=s[-1].shape))
		path = np.asarray(planner.planning(list(highest_std_location), list(env.fleet.vehicles[0].position))).T

		distance_mask = np.where(np.linalg.norm(path-env.fleet.vehicles[0].position, axis=1) < 4)[0][-1]
		next_position = path[distance_mask]

		angle_dif = np.arctan2(next_position[1]-env.fleet.vehicles[0].position[1], next_position[0]-env.fleet.vehicles[0].position[0])
		angle_dif = 2*np.pi + angle_dif if angle_dif < 0 else angle_dif

		a = np.argmin(np.abs(angle_dif-env.angle_set))

		while not env.valid_action(a):
			a = env.action_space.sample()

		s, r, done, i = env.step(a)

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
			fig, ax = plt.subplots(1, 1)
			ax.imshow(s[-1], cmap='bone', vmin=0, vmax=1, interpolation='bicubic', zorder=2)
			mask_nav = env.navigation_map
			mask_nav[np.where(mask_nav == 1)] = np.nan
			ax.imshow(mask_nav, cmap='gray_r', vmin=0, vmax=1, zorder=3)
			plt.tick_params(left=False, right=False, labelleft=False,
			                labelbottom=False, bottom=False)

			plot_trajectory(ax, env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
			                z=None,
			                colormap='jet',
			                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
			plt.show(block=True)

	metric_recorder.record_finish(t=t)

metric_recorder.record_save()


