import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization
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
                    'max_distance': 1000,
                    'number_of_trials': 5,
                    'number_of_actions': 8,
                    'random_init_point': True,
                    'termination_condition': False,
                    'dt': 0.05
                    }

env = BaseTemporalEntropyMinimization(**environment_args)
env.reset()
# N executions of the algorithm
N = 10


def reverse_action(a):
    return (a + 4) % 8


for i in range(N):
    env.reset()
    a = env.action_space.sample()
    new_a = a
    done = False

    while not done:

        valid = env.valid_action(a)
        while not valid:
            new_a = env.action_space.sample()
            while new_a == reverse_action(a):
                new_a = env.action_space.sample()
            valid = env.valid_action(new_a)
        a = new_a

        _, r, done, m = env.step(a)
        print("Random peaks detection: ", m["DetectionRate"])
        env.render()
        env.axs[2].plot(env.random_peaks[:, 1], env.random_peaks[:, 0], 'xb')
        env.axs[3].plot(env.GroundTruth.A[:,1]*navigation_map.shape[1], env.GroundTruth.A[:,0]*navigation_map.shape[0], 'xr')

"""
	plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)
"""
