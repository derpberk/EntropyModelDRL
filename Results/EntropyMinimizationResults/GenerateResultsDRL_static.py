from Algorithms.RainbowDQL.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization
from Results.metrics_loader import metric_constructor
from utils import plot_trajectory
import matplotlib.pyplot as plt


parameters = {
	'epsilongreedy': {'noisy': False, 'safe_actions': False},
	'noisy': {'noisy': True, 'safe_actions': True},
	'safe': {'noisy': True, 'safe_actions': True},
}

#ALGORITHM = 'noisy'
ALGORITHM = 'safe'
# ALGORITHM = 'epsilongreedy'

navigation_map = np.genfromtxt('../../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]

# Environment parameters #
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
# Agent parameter #
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
              'noisy': parameters[ALGORITHM]['noisy'],
              'logdir': None,
              'log_name': "Experiment",
              'safe_actions': parameters[ALGORITHM]['safe_actions']}

# Create agent #
agent = DuelingDQNAgent(**agent_args)
agent.epsilon = 0.05
# Load policy #
policy_path = f'/Users/samuel/Desktop/runs/Temporal_Noisy/BestPolicy.pth'
agent.load_model(policy_path)

number_of_trials = 20
metric_recorder = metric_constructor(f'./DRL_temporal_{ALGORITHM}.csv', temporal=True)
draw = False

for t in range(number_of_trials):

	print(f"Run {t}")
	s = env.reset()
	d = False
	metric_recorder.record_new()

	while not d:
		# Take action
		a = agent.select_action(s)

		s, r, d, i = env.step(a)

		metric_recorder.record_step(r,
		                            i['Entropy'],
		                            i['Area'],
		                            i['DetectionRate'],
		                            np.sum(env.fleet.get_distances()),
		                            env.measured_locations,
		                            np.asarray(env.measured_values).squeeze(1),
		                            env.visitable_locations,
		                            env.GroundTruth_field[env.visitable_locations[:,0], env.visitable_locations[:,1]],
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
