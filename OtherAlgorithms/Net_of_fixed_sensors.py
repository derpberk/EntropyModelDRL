import matplotlib.pyplot as plt
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

# Number of sensors #
N_of_sensors = 25
# Position of sensors #
mesh = np.meshgrid(np.linspace(0, navigation_map.shape[0], N_of_sensors, endpoint=False),
                   np.linspace(0, navigation_map.shape[1], N_of_sensors, endpoint=False))
all_locations = np.vstack((mesh[0].flatten(), mesh[1].flatten())).T.astype(int)
sensor_positions = []
# Select those values that are visitable
for position in all_locations:
    if navigation_map[position[0], position[1]] == 1:
        sensor_positions.append(list(position))

sensor_positions = np.asarray(sensor_positions)

environment_args = {'navigation_map': navigation_map,
                    'number_of_agents': sensor_positions.shape[0],
                    'initial_positions': sensor_positions,
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
env.render()
plt.plot(sensor_positions[:, 1], sensor_positions[:, 0], 'xr')
plt.show(block=True)

"""
    plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
                    colormap='jet',
                    num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)
"""
