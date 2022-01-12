import sys
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, ActionNoise

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
        net_arch=dict(pi=[256, 128, 128, 64, 64], qf=[256, 128, 128, 64, 64]),

    )

    # Create log dir
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_dir = "./Results/SAC_Results_" + date_time
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
                        }

    env = BaseEntropyMinimization(navigation_map=environment_args['navigation_map'],
                                  number_of_agents=environment_args['number_of_agents'],
                                  initial_positions=environment_args['initial_positions'],
                                  movement_length=environment_args['movement_length'],
                                  density_grid=environment_args['density_grid'],
                                  noise_factor=environment_args['noise_factor'],
                                  lengthscale=environment_args['lengthscale'],
                                  initial_seed=environment_args['initial_seed'],
                                  collision_penalty=environment_args['collision_penalty'],
                                  max_distance=environment_args['max_distance'],
                                  number_of_trials=environment_args['number_of_trials'],
                                  termination_condition=environment_args['termination_condition'],
                                  random_init_point=environment_args['random_init_point']
                                  )

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


    class NormalActionNoiseDecay(ActionNoise):

        def __init__(self, mean: np.ndarray, sigma: np.ndarray, decay_rate: float):
            self._decay_rate = decay_rate
            self._sigma = sigma
            self._mu = mean
            super(ActionNoise, self).__init__()

        def __call__(self) -> np.ndarray:
            self._sigma = np.clip(self._sigma - self._decay_rate, 0.05, np.inf)
            return np.random.normal(self._mu, self._sigma)


    # Constant Action Noise #
    action_noise = NormalActionNoiseDecay(mean=np.zeros((1,)), sigma=0.2 * np.ones((1,)),
                                          decay_rate=(0.2 - 0.05) / 200000)

    learning_rate = 3E-4

    th.cuda.empty_cache()

    model = SAC(
        policy='CnnPolicy',
        env=monitor_env,
        learning_rate=learning_rate,
        buffer_size=100_000,  # 1e6
        learning_starts=0,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=action_noise,
        optimize_memory_usage=True,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        seed=0
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
                tb_log_name="./ExperimentSAC_" + date_time,
                callback=callback)
