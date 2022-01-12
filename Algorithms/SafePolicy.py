from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from torch import nn
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
from collections import namedtuple
import torch
from typing import Union
import gym.spaces

bound = namedtuple('bound', 'lower upper')

class SafeActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(SafeActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        self.safe_action_strategy = SafeActionDistribution(bound_map = kwargs['navigation_map'],
                                                           movement_size = kwargs['movement_size'])


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Reimplementation of the forward but in a safe explorative way

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = self.safe_action_strategy.get_actions(distribution = distribution,
                                                        position = obs['position'].cpu().detach().numpy(),
                                                        deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob



class SafeActionDistribution:

    def __init__(self, bound_map: np.ndarray, movement_size: float):

        """ Safe/Bounded Gaussian Distribution for safe-exploration """

        self.bound_map = bound_map  # Constant map of enviorn
        self.movement_size_ = movement_size
        self.action_set = np.linspace(-1.0, 1.0, 50, dtype=np.float32)

    def get_actions(self, distribution, position, deterministic):

        if deterministic:
            return distribution.mode()
        else:
            proposed = distribution.sample()
            bounds = self.compute_invalid_bounds(position)
            safe_act, v = self.return_safe_action(distribution, bounds, proposed_action=proposed)
            return safe_act

    def compute_invalid_bounds(self, position: np.ndarray):
        """
        Returns an array of named tuples (bound type) with non-permited actions.

        :param bound_map: The navigation map.
        :param position: The current position of the agent.
        :param movement_size: The movement in pixels
        :return: numpy array with bounds of forbidden actions
        """

        valid_set = np.zeros_like(self.action_set, dtype=bool)
        movement_size = self.movement_size_ + 1

        for i, action in enumerate(self.action_set):

            # Compute the movement vector in XY #
            movement = np.array([movement_size * np.cos(np.pi * action), movement_size * np.sin(np.pi * action)], dtype=np.float32)

            # Next intended position given the aforementioned movement #
            next_intended_position = position + movement

            next_intended_position = np.clip(next_intended_position, a_min=(0, 0), a_max=(self.bound_map.shape[0] - 1, self.bound_map.shape[1] - 1))

            # Compute if there is a collision #

            if self.bound_map[next_intended_position[0].astype(int), next_intended_position[1].astype(int)] == 1:
                valid_set[i] = True

        bound_list = self.zero_runs(valid_set)

        bounds = [bound(-np.inf, -1)]

        for interval in bound_list:
            ext = 3
            extended_interval_left = int(np.clip(interval[0] - ext, 0, np.inf))
            extended_interval_right = int(np.clip(interval[1] - 1 + ext, -np.inf, len(self.action_set) - 1))
            bounds.append(bound(self.action_set[extended_interval_left],
                                self.action_set[extended_interval_right]))

        bounds.append(bound(1, np.inf))

        return bounds

    @staticmethod
    def check_bounds(set_of_bounds, sample):

        for b in set_of_bounds:

            if (b.lower <= sample[0] <= b.upper).any().item():
                return False

        return True

    def return_safe_action(self, distribution: Union[torch.distributions.Distribution, gym.spaces.Space],
                           set_of_bounds, maxnum_of_iterations=1E3, proposed_action=None):

        it = 0
        valid = False
        sample = proposed_action

        if proposed_action is not None:
            valid = self.check_bounds(set_of_bounds, proposed_action)
            sample = proposed_action

        while not valid:

            sample = distribution.sample()
            it += 1
            valid = self.check_bounds(set_of_bounds, sample)

            if it == maxnum_of_iterations and not valid:
                print('Maxnum of iterations reached. No safe action encountered. Consider why.')
                return sample, False

        return sample, valid

    @staticmethod
    def zero_runs(arr):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges








