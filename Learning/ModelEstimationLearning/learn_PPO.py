from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import gym
import os
import torch as th
import numpy as np

from NeuralNetworks.Baselines3 import CustomCNNforSensing
from utils import SaveOnBestTrainingRewardCallback

from Environment.ModelEstimationScenario import GaussianProcessModeling

from datetime import datetime

def make_env(**kwargs):
    def _init() -> gym.Env:
        env = GaussianProcessModeling(navigation_map=kwargs['navigation_map'],
                                      number_of_agents=kwargs['number_of_agents'],
                                      initial_positions=kwargs['initial_positions'],
                                      movement_limits=kwargs['movement_limits'],
                                      noise_factor=kwargs['noise_factor'],
                                      lengthscale=kwargs['lengthscale'],
                                      initial_seed=kwargs['initial_seed'],
                                      collision_penalty=kwargs['collision_penalty'],
                                      max_distance=kwargs['max_distance'],
                                      )
        env.seed(kwargs['initial_seed'] * 400)

        return env

    set_random_seed(kwargs['initial_seed'])

    return _init


if __name__ == '__main__':

    policy_kwargs = dict(
        features_extractor_class=CustomCNNforSensing,
        features_extractor_kwargs=dict(features_dim=256),
        activation_fn=th.nn.LeakyReLU,
        net_arch=[128, 128, 128, dict(pi=[64, 64, 64], vf=[64, 64, 64])],

    )

    # Create log dir
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_dir = "./Results/PPO_Results_" + date_time
    os.makedirs(log_dir, exist_ok=True)

    # Navigation Map and initial positions #
    navigation_map = np.genfromtxt('../../Environment/ypacarai_map.csv')
    initial_position = np.array([67, 73])

    # Create Parallel Environment
    environment_args = {'navigation_map': navigation_map,
                        'number_of_agents': 1,
                        'initial_positions': [[65, 75]],
                        'movement_limits': [6, 20],
                        'noise_factor': 1E-5,
                        'lengthscale': 10,
                        'initial_seed': 0,
                        'collision_penalty': -5,
                        'max_distance': 400}

    vec_env = SubprocVecEnv([make_env(environment_args) for i in range(12)])

    # Create Monitoring Environment
    monitor_env = VecMonitor(vec_env, log_dir)

    # Create Training Model and parameters

    num_of_steps = 1E6


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

    model = PPO("CnnPolicy", monitor_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
                n_steps=512,  # The number of steps to run for each environment per update - LA PROFUNDIDAD DEL EPISODIO
                learning_rate=learning_rate,  # The learning rate
                target_kl=5,  # The limiting value of KL divergence
                ent_coef=0.01,  # Entropy coefficient for the loss computing
                batch_size=64,  # Mini-batch Size
                n_epochs=10,  # Number of epochs when optimizing the surrogate loss
                gamma=0.95,  # Discount factor
                gae_lambda=0.95,  # Trade-off between bias/variance in GAE
                clip_range=0.2,  # Clip parameter for the fraction
                max_grad_norm=0.5,  # Maximum gradient
                seed=42,  # Seed
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
                tb_log_name="ExperimentPPO_" + date_time,
                callback=callback)
