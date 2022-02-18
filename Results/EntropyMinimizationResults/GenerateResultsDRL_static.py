from Algorithms.RainbowDQL.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
from Environment.EntropyMinimization import BaseEntropyMinimization
from Results.metrics_loader import metric_constructor
from utils import plot_trajectory
import matplotlib.pyplot as plt

navigation_map = np.genfromtxt('../../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]

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

agent_args = {'env': env,
              'memory_size': 100000,
              'batch_size': 64,
              'target_update': 100,
              'soft_update': True,
              'tau': 0.001,
              'epsilon_values': [1.0, 0.1],
              'epsilon_interval': [0.1, 0.8],
              'learning_starts': 10,
              'gamma': 0.99,
              'lr': 1e-4,
              'alpha': 0.5,
              'beta': 0.4,
              'prior_eps': 1e-6,
              'noisy': True,
              'logdir': None,
              'log_name': "Experiment",
              'safe_actions': True}

agent = DuelingDQNAgent(**agent_args)
agent.epsilon = 0
policy_path = '/Users/samuel/Desktop/runs/Baselines_Noisy/BestPolicy.pth'

agent.load_model(policy_path)

number_of_trials = 100

metric_recorder = metric_constructor('./DRL_noisy_static.csv')
draw = True

for t in range(number_of_trials):

	# Reset
	s = env.reset()
	d = False
	metric_recorder.record_new()

	while not d:
		# Take action
		a = agent.safe_select_action(s)
		agent.dqn.reset_noise()

		s, r, d, i = env.step(a)

		metric_recorder.record_step(r,
		                            i['Entropy'],
		                            i['Area'],
		                            i['DetectionRate'],
		                            np.sum(env.fleet.get_distances()),
		                            env.measured_locations,
		                            env.measured_values.squeeze(1),
		                            env.visitable_locations,
		                            env.GroundTruth_field[env.visitable_locations[:,0], env.visitable_locations[:,1]])


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
