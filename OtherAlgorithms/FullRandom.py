import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization
from Results.metrics_loader import metric_constructor
import matplotlib.pyplot as plt
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
                    'random_init_point': False,
                    'termination_condition': False
                    }

env = BaseEntropyMinimization(**environment_args)
env.reset()
# N executions of the algorithm
N = 100
draw = True

class FullRandomAgent:

	def __init__(self, action_space, valid_action_function):
		self.action_space = action_space
		self.valid_action_function = valid_action_function

	def safe_select_action(self):

		valid = False
		while not valid:
			a = self.action_space.sample()
			valid = self.valid_action_function(a)

		return a

agent = FullRandomAgent(env.action_space, env.valid_action)
metric_recorder = metric_constructor('../Results/EntropyMinimizationResults/StaticFullRandom.csv')

for t in range(N):

	env.reset()
	done = False
	metric_recorder.record_new()

	while not done:

		a = agent.safe_select_action()

		s, r, done, i = env.step(a)

		metric_recorder.record_step(r,
		                            i['Entropy'],
		                            i['Area'],
		                            i['DetectionRate'],
		                            np.sum(env.fleet.get_distances()),
		                            env.measured_locations,
		                            env.measured_values.squeeze(1),
		                            env.visitable_locations,
		                            env.GroundTruth_field[env.visitable_locations[:, 0], env.visitable_locations[:, 1]])

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

"""
	plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)
"""





