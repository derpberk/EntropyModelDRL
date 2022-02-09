from Algorithms.RainbowDQL.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization

navigation_map = np.genfromtxt('../../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]
# Create Parallel Environment
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
                    'dt': 0.05,
                    }

env = BaseTemporalEntropyMinimization(**environment_args)


agent = DuelingDQNAgent(env=env,
                        memory_size=100000,
                        batch_size=64,
                        target_update=100,
                        soft_update=True,
                        tau=0.001,
                        epsilon_values=[1.0, 0.1],
                        epsilon_interval=[0.1, 0.8],
                        learning_starts=10,
                        gamma=0.99,
                        lr=1e-4,
                        alpha=0.5,
                        beta=0.4,
                        prior_eps=1e-6,
                        noisy=False,
                        logdir=None,
                        log_name="Experiment"
                        )

agent.load_model(r'/Users/samuel/Desktop/runs/Temporal_PER_dueling_epsilongreedy/BestPolicy.pth')
agent.epsilon = 0
env.reset()
L1 = agent.evaluate_policy(30, render=False, safe=True)

env = BaseTemporalEntropyMinimization(**environment_args)
agent = DuelingDQNAgent(env=env,
                        memory_size=100000,
                        batch_size=64,
                        target_update=100,
                        soft_update=True,
                        tau=0.001,
                        epsilon_values=[1.0, 0.1],
                        epsilon_interval=[0.1, 0.8],
                        learning_starts=10,
                        gamma=0.99,
                        lr=1e-4,
                        alpha=0.5,
                        beta=0.4,
                        prior_eps=1e-6,
                        noisy=True,
                        logdir=None,
                        log_name="Experiment"
                        )

agent.load_model(r'/Users/samuel/Desktop/runs/Temporal_noisy/BestPolicy.pth')
agent.epsilon = 0
env.reset()
L2 = agent.evaluate_policy(30, render=False, safe=True)

env = BaseTemporalEntropyMinimization(**environment_args)
agent = DuelingDQNAgent(env=env,
                        memory_size=100000,
                        batch_size=64,
                        target_update=100,
                        soft_update=True,
                        tau=0.001,
                        epsilon_values=[1.0, 0.1],
                        epsilon_interval=[0.1, 0.8],
                        learning_starts=10,
                        gamma=0.99,
                        lr=1e-4,
                        alpha=0.5,
                        beta=0.4,
                        prior_eps=1e-6,
                        noisy=True,
                        logdir=None,
                        log_name="Experiment"
                        )

agent.load_model(r'/Users/samuel/Desktop/runs/Safe_Noisy_Temporal/BestPolicy.pth')
agent.epsilon = 0
env.reset()
L3 = agent.evaluate_policy(30, render=False, safe=True)


np.savetxt('ENTROPY_VALUES.csv', np.array((L1, L2, L3)))