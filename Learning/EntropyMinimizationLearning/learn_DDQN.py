import sys
from stable_baselines3.dqn import DQN
from stable_baselines3.common.monitor import Monitor

import os
import torch as th
import numpy as np

from NeuralNetworks.Baselines3 import CustomCNNforSensing
from utils import SaveOnBestTrainingRewardCallback

from Environment.EntropyMinimization import BaseEntropyMinimization

from datetime import datetime

if __name__ == '__main__':

    policy_kwargs = dict(
        features_extractor_class=CustomCNNforSensing,
        features_extractor_kwargs=dict(features_dim=1024),
        activation_fn=th.nn.ReLU,
        net_arch=[256, 128, 128, 64, 64],

    )

    # Create log dir
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_dir = "./Results/DQN_Results_" + date_time
    os.makedirs(log_dir, exist_ok=True)

    # Navigation Map and initial positions #
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

    # Create Monitoring Environment
    monitor_env = Monitor(env, log_dir)

    # Create Training Model and parameters

    num_of_steps = 5E5


    def learning_rate_schedule(remaining_progress):

        lr_max = 1E-4
        lr_min = 1E-5
        if remaining_progress > 0.9:
            return lr_max
        elif remaining_progress < 0.1:
            return lr_min
        else:
            return ((lr_max - lr_min) / (0.9 - 0.1)) * (remaining_progress - 0.1) + lr_min


    learning_rate = learning_rate_schedule

    model = DQN(
        policy='CnnPolicy',
        env=monitor_env,
        learning_rate=learning_rate,
        buffer_size=100_000,  # 1e6
        learning_starts=5000,
        batch_size=32,
        tau=0.0001,
        gamma=0.99,
        train_freq=1,
        gradient_steps=-1,
        optimize_memory_usage=True,
        target_update_interval=100,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=0,
    )

    # Write experiment descriptor
    with open(log_dir + 'experiment_descriptor.log', 'w') as f:

        msg = f""" 
        *************************** Experiment ****************************** 
        *********************************************************************
        Environment parameters:
        {str(environment_args)}
        ********************************************************************* 
        """

        f.write(msg)
        f.close()

    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    # Learn!
    model.learn(total_timesteps=int(num_of_steps),
                tb_log_name="./ExperimentDQN_" + date_time,
                callback=callback)
