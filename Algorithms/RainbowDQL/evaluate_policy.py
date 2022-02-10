from Algorithms.RainbowDQL.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
from Environment.EntropyMinimization import BaseEntropyMinimization, BaseTemporalEntropyMinimization

navigation_map = np.genfromtxt('../../Environment/ypacarai_map_middle.csv')
initial_position = [[30, 38]]
# Create Parallel Environment

experiment_set = 2

if experiment_set == 1:

    noisy_list = [False, True, True]
    safe_list = [False, False, True]
    names = ['Baselines_epsilon', 'Baselines_noisy', 'Baselines_safe']

    L = []

    ########### DYNAMIC CASE ###########
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
                                 'termination_condition': False,
                                 }

    for i in range(len(names)):
        collisions = 0
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
                               'noisy': noisy_list[i],
                               'logdir': None,
                               'log_name': "Experiment",
                               'safe_actions': safe_list[i]}
        agent = DuelingDQNAgent(**agent_args)

        agent.load_model(r'/home/samuel/Escritorio/runs/' + names[i] + r'/BestPolicy.pth')
        agent.epsilon = 0
        env.reset()
        L.append(agent.evaluate_policy(30, render=False))
        collisions += env.fleet.fleet_collisions

        print(f"N_OF_COLLISIONS FOR {names[i]}: {collisions}")

    np.savetxt('ENTROPY_BASE.csv', np.array(L))


elif experiment_set == 2:

    noisy_list = [False, True, True]
    safe_list = [False, False, True]
    names = ['Temporal_epsilon', 'Temporal_noisy', 'Temporal_safe']

    L = []

    ########### DYNAMIC CASE ###########
    temporal_environment_args = {'navigation_map': navigation_map,
                                 'number_of_agents': 1,
                                 'initial_positions': initial_position,
                                 'movement_length': 3,
                                 'density_grid': 0.2,
                                 'noise_factor': 1E-5,
                                 'lengthscale': 5,
                                 'initial_seed': 0,
                                 'collision_penalty': -1,
                                 'max_distance': 1000,
                                 'number_of_trials': 24,
                                 'number_of_actions': 8,
                                 'random_init_point': True,
                                 'termination_condition': False,
                                 'dt': 0.05,
                                 }

    for i in range(len(names)):
        collisions = 0
        env = BaseTemporalEntropyMinimization(**temporal_environment_args)
        temporal_agent_args = {'env': env,
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
                               'noisy': noisy_list[i],
                               'logdir': None,
                               'log_name': "Experiment",
                               'safe_actions': safe_list[i]}
        agent = DuelingDQNAgent(**temporal_agent_args)

        agent.load_model(r'/home/samuel/Escritorio/runs/' + names[i] + r'/BestPolicy.pth')
        agent.epsilon = 0
        env.reset()
        L.append(agent.evaluate_policy(30, render=False))
        collisions += env.fleet.fleet_collisions
        print(f"N_OF_COLLISIONS FOR {names[i]}: {collisions}")


    np.savetxt('ENTROPY_TEMPORAL.csv',np.array(L))