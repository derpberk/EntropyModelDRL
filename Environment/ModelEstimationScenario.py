from abc import ABC
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import gym
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from EnvironmentUtils import GroundTruth


class GaussianProcessModeling(gym.Env, ABC):

    def __init__(self,
                 navigation_map,
                 number_of_agents: int,
                 initial_positions: list,
                 movement_limits: list,
                 noise_factor: float,
                 lengthscale: float,
                 initial_seed=0,
                 collision_penalty=-1,
                 max_distance=400,
                 ):

        self.navigation_map = navigation_map
        self.number_of_agents = number_of_agents
        self.initial_positions = np.asarray(initial_positions)
        self.movement_min_length = movement_limits[0]
        self.movement_max_length = movement_limits[1]
        self.lengthscale = lengthscale
        self.noise_factor = noise_factor
        self.seed = initial_seed
        self.collision_penalty = collision_penalty
        self.max_distance = max_distance
        self.max_num_of_points = 30


        """ Positions where is it possible to measure """
        self.visitable_locations = np.vstack(np.where(self.navigation_map != 0)).T

        """ Check every initial position """
        for i, pos in enumerate(self.initial_positions):
            assert self.navigation_map[pos[0], pos[1]] == 1, f"Impossible position for drone {i}."

        """ Gym Parameters """
        # Action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.number_of_agents, 2))
        # Observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2 + self.number_of_agents,
                                                                          self.navigation_map.shape[0],
                                                                          self.navigation_map.shape[1]))

        """ Kernel"""
        self.kernel = C(constant_value=1.0) * Matern(length_scale=self.lengthscale)  # RBF(length_scale=self.lengthscale)

        self.GP = GaussianProcessRegressor(kernel=self.kernel, alpha=self.noise_factor)

        """ Ground Truth Generator """
        self.GroundTruth = GroundTruth(navigation_map=self.navigation_map,
                                       function_type='shekel',
                                       initial_seed=self.seed)
        # Sample a new GT
        self.GroundTruth_field = self.GroundTruth.sample_gt()

        """ Other attributes """
        self.positions = None
        self.measured_locations = None
        self.measured_values = None
        self.state = None
        self.estimated_mu = None
        self.estimated_std = None
        self.distance = 0

    def reset(self):

        """ Reset the environment """

        """ Reset distance """
        self.distance = 0
        """ New ground truth """
        self.GroundTruth.reset_gt()
        self.GroundTruth_field = self.GroundTruth.sample_gt()
        """ Starting positions """
        self.positions = np.copy(self.initial_positions)
        """ Reset the measurements """
        self.measured_locations = np.copy(self.initial_positions)
        """ Take new measurements """
        self.measured_values = self.measure()
        """ Update gaussian process """
        self.GP.fit(self.measured_locations, self.measured_values.squeeze(axis=1))
        self.estimated_mu, self.estimated_std = self.GP.predict(self.visitable_locations, return_std=True)
        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def update_state(self):

        state = np.zeros((2 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))


        """ The position of the vehicles """
        for i in range(0, self.number_of_agents):
            state[i, int(self.positions[i, 0]), int(self.positions[i, 1])] = 1.0

        """ Std """
        uncertainty = self.estimated_std
        uncertainty = (uncertainty - np.min(uncertainty))/(np.max(uncertainty) - np.min(uncertainty))
        state[-2, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        """ Mu """
        mu = self.estimated_mu
        mu = (mu - np.min(mu)) / (np.max(mu) - np.min(mu))
        state[-1] = -1.0
        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = mu

        return state

    def measure(self, positions=None):
        """
        Take a measurement in the given N positions
        :param positions: Given positions.
        :return: An numpy array with dims (N,2)
        """

        if positions is None:
            positions = self.positions

        values = []
        for pos in positions:
            values.append([self.GroundTruth_field[int(pos[0]), int(pos[1])]])

        return np.asarray(values)

    def action2movement(self, action):

        l = (self.movement_max_length - self.movement_min_length)/2 * (action[1] + 1) + self.movement_min_length

        return [l * np.cos(np.pi * action[0]), l * np.sin(np.pi * action[1])]

    def clip_position_within_boundaries(self, actual_positions, intended_positions):

        clipped_positions = []

        for position, intended_position in zip(actual_positions, intended_positions):

            clipped_position = [position[0], position[1]]

            distance = np.linalg.norm(intended_position - position)
            segments = int(distance // self.movement_min_length)

            if distance == 0:
                clipped_position = [actual_positions[0], actual_positions[1]]

            else:

                x_vector = np.linspace(intended_position[0], position[0], segments)
                y_vector = np.linspace(intended_position[1], position[1], segments)

                for i in range(len(x_vector)):
                    if self.navigation_map[int(x_vector[i]), int(y_vector[i])] == 1:
                        clipped_position = [x_vector[i], y_vector[i]]
                        break

            clipped_positions.append(clipped_position)

        return np.asarray(clipped_positions)

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The angle of the movement [-1,1] and the length of the movement
        :return: next state, reward, done, info
        """

        done = False

        clipped_actions = np.clip(action, self.action_space.low[0], self.action_space.high[0])

        movement = np.array([self.action2movement(action) for action in clipped_actions])

        # Next intended position given the aforementioned movement #
        next_intended_positions = np.clip(self.positions + movement, a_min=(0,0),
                                          a_max=(self.navigation_map.shape[0]-1, self.navigation_map.shape[1]-1))

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        if collision:

            next_intended_positions = self.clip_position_within_boundaries(self.positions, next_intended_positions)


        # Update the positions
        self.positions = next_intended_positions

        self.distance += np.sum(np.linalg.norm(movement, axis=1))

        if self.distance >= self.max_distance:
            done = True

        """ Take new measurements """
        new_measurements = self.measure()
        self.measured_values = np.vstack((self.measured_values, new_measurements))
        self.measured_locations = np.vstack((self.measured_locations, self.positions))

        """ Update GP """
        self.GP.fit(self.measured_locations, self.measured_values.squeeze(axis=1))
        self.estimated_mu, self.estimated_std = self.GP.predict(self.visitable_locations, return_std=True)

        """ Compute reward """
        reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        if len(self.measured_values) > self.max_num_of_points * self.number_of_agents:
            done = True

        return self.state, reward, done, {}

    def reward(self):
        """
        The reward function

        :return: The information gain defined as Tr{t} - Tr{t+1}
        """

        mse = mean_squared_error(y_true = self.GroundTruth_field[self.visitable_locations[:,0], self.visitable_locations[:,1]],
                                 y_pred = self.estimated_mu)

        return np.clip(-1 * mse, a_min=-1.0, a_max=0.0)

    def check_collision(self, next_positions):

        for position in next_positions:

            if self.navigation_map[int(position[0]), int(position[1])] == 0:
                return True

        return False

    def render(self, mode="human", first=True):

        if first:
            fig, self.axs = plt.subplots(1, 2)

        if mode == 'human':

            colors = matplotlib.cm.get_cmap('jet', self.number_of_agents)
            colors = colors(range(self.number_of_agents))
            self.axs[0].imshow(self.state[-1])
            self.axs[0].set_title("Uncertainty")
            self.axs[1].imshow(self.state[-2])
            self.axs[1].set_title("Model")

            for color, pos in zip(colors, self.positions):
                self.axs[0].plot(pos[1], pos[0], 'x', color=color)
                self.axs[1].plot(pos[1], pos[0], 'x', color=color)

if __name__ == '__main__':

    plt.ion()

    navigation_map = np.genfromtxt('./ypacarai_map.csv')
    env = GaussianProcessModeling(navigation_map=navigation_map,
                                  number_of_agents=1,
                                  initial_positions=[[65, 75]],
                                  movement_limits=[6, 20],
                                  noise_factor=1E-5,
                                  lengthscale = 10,
                                  initial_seed = 0,
                                  collision_penalty = -1,
                                  max_distance = 400,
                                  )

    env.reset()

    env.render(first=True)
    plt.pause(0.5)

    N = 200
    for i in range(1):

        env.reset()

        for k in range(N):

            act = env.action_space.sample()

            s, r, d, _ = env.step(act)

            print(f"{i}.{k} - Reward: ", r)
            env.render(first=False)
            plt.pause(0.5)
            if d:
                break

    plt.pause(1)
