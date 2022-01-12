from Algorithms.RainbowDQL.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
import random
from Environment.EntropyMinimization import BaseEntropyMinimization


seed = 777

np.random.seed(seed)
random.seed(seed)

# train

navigation_map = np.genfromtxt('../../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]

# Create Parallel Environment
environment_args = {'navigation_map': navigation_map,
                    'number_of_agents': 1,
                    'initial_positions': initial_position,
                    'movement_length': 3,
                    'density_grid': 0.15,
                    'noise_factor': 1E-5,
                    'lengthscale': 5,
                    'initial_seed': 0,
                    'collision_penalty': -1,
                    'max_distance': 200,
                    'number_of_trials': 5,
                    'random_init_point': True,
                    'termination_condition': False,
                    'discrete': True
                    }

env = BaseEntropyMinimization(**environment_args)

agent = DuelingDQNAgent(env=env,
                        memory_size=100000,
                        batch_size=64,
                        target_update=100,
                        soft_update=True,
                        tau=0.0001,
                        epsilon_values=[1.0, 0.05],
                        epsilon_interval=[0.1, 0.5],
                        learning_starts=10,
                        gamma=0.99,
                        lr=1e-4,
                        alpha=0.5,
                        beta=0.4,
                        prior_eps=1e-6,
                        logdir=None,
                        noisy=True,
                        log_name="Experiment"
                        )

agent.train(episodes=15000)

