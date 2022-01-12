from abc import ABC
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gym
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from EnvironmentUtils import GroundTruth
import time
from MathUtils import conditioning_cov_matrix, conditioning_std, conditioning_cov_matrix_with_time, \
    conditioning_std_with_time


class BaseEntropyMinimization(gym.Env, ABC):

    def __init__(self,
                 navigation_map,
                 number_of_agents: int,
                 initial_positions: list,
                 movement_length: int,
                 density_grid: float = 0.1,
                 noise_factor: float = 0.0001,
                 lengthscale: float = 10,
                 initial_seed=0,
                 collision_penalty=-1,
                 number_of_trials=1,
                 number_of_actions=8,
                 max_distance=400,
                 random_init_point = False,
                 termination_condition = True,
                 discrete = True,
                 ):

        self.navigation_map = navigation_map
        self.number_of_agents = number_of_agents
        self.initial_positions = np.asarray(initial_positions)
        self.movement_length = movement_length
        self.density_grid = density_grid
        self.lengthscale = lengthscale
        self.noise_factor = noise_factor
        self.initial_seed = initial_seed
        self.collision_penalty = collision_penalty
        self.max_distance = max_distance
        self.random_initial_position = random_init_point
        self.termination_condition = termination_condition
        self.number_of_trials = number_of_trials
        self.discrete = discrete

        """ Create the measurement/evaluation points """
        density = np.round(1 / density_grid)

        """ Positions where is it possible to measure """
        self.visitable_locations = np.vstack(np.where(self.navigation_map != 0)).T

        # Create a mesh with the specified density #
        mesh = np.meshgrid(np.arange(0, navigation_map.shape[0], step=int(density)),
                           np.arange(0, navigation_map.shape[1], step=int(density)))
        all_locations = np.vstack((mesh[0].flatten(), mesh[1].flatten())).T.astype(int)

        evaluation_list = []

        # Select those values that are visitable
        for position in all_locations:
            if self.navigation_map[position[0], position[1]] == 1:
                evaluation_list.append(list(position))

        self.evaluation_locations = np.asarray(evaluation_list)

        """ Check every initial position """
        for i, pos in enumerate(self.initial_positions):
            assert self.navigation_map[pos[0], pos[1]] == 1, f"Impossible position for drone {i}."

        """ Gym Parameters """
        # Action space
        if self.discrete:
            # Discrete action space divided in 8 different actions
            # TODO: Not applicable with multi-agent
            assert self.number_of_agents == 1, "Not implemented discrete actions for multi-agent."
            self.action_space = gym.spaces.Discrete(number_of_actions)
            self.angle_set = np.linspace(0,2*np.pi, number_of_actions, endpoint=False)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.number_of_agents, ))

        # Observation space
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 + self.number_of_agents,
                                                                          self.navigation_map.shape[0],
                                                                          self.navigation_map.shape[1]))
        """ Kernel for conditioning """
        self.kernel = C(constant_value=1.0) * RBF(length_scale=self.lengthscale)

        """ Ground Truth Generator """
        self.GroundTruth = GroundTruth(navigation_map=self.navigation_map,
                                       function_type='shekel',
                                       initial_seed=self.initial_seed)
        # Sample a new GT
        self.GroundTruth_field = self.GroundTruth.sample_gt()

        """ Other attributes """
        self.positions = None
        self.heading_angles = None
        self.waypoints = None
        self.trajectories = None
        self.measured_locations = None
        self.measured_values = None
        self.covariance_matrix = None
        self.trace = None
        self.trace_ant = None
        self.state = None
        self.tr0 = None
        self.norm_rew_term = None
        self.distance = 0
        self.figure, self.axs = None, None
        self.number_of_collisions = 0

        self.seed(self.initial_seed)

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):

        """ Reset the environment """

        """ Reset distance """
        self.distance = 0
        self.number_of_collisions = 0
        """ New ground truth """
        self.GroundTruth.reset_gt()
        self.GroundTruth_field = self.GroundTruth.sample_gt()
        """ Starting positions """

        if self.random_initial_position:
            self.initial_positions = self.visitable_locations[np.random.choice(np.arange(0,len(self.visitable_locations)),
                                                                               self.number_of_agents, replace=False)]

        self.positions = np.copy(self.initial_positions)

        """ Heading angles """
        self.heading_angles = np.zeros((self.number_of_agents,))
        center = np.array(self.navigation_map.shape, dtype = int)/2
        self.heading_angles = np.asarray([np.arctan2(center[1]-pos[1], center[0]-pos[0]) for pos in self.positions])
        self.waypoints = np.expand_dims(np.copy(self.initial_positions), 0)  # Trajectory of the agent #
        self.trajectories = np.copy(self.waypoints)
        """ Reset the measurements """
        self.measured_locations = np.copy(self.initial_positions)
        """ Take new measurements """
        self.measured_values = self.measure()
        """ Update the covariance matrix and the trace"""
        self.tr0 = np.sum(np.real(np.linalg.eigvals(self.kernel(self.evaluation_locations))))
        self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations, self.kernel, alpha=self.noise_factor)
        self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

        self.trace_ant = self.trace
        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def update_state(self):

        state = np.zeros((2 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.copy(self.navigation_map)

        """ The position of the vehicles """

        for k in range(0, self.number_of_agents):
            w = np.linspace(0, 1, len(self.trajectories[0]))
            state[k+1, self.trajectories[k, :, 0], self.trajectories[k, :, 1]] = w

        uncertainty = conditioning_std(self.visitable_locations, self.measured_locations, self.kernel)

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

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

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The angle of the movement [-1,1]
        :return: next state, reward, done, info
        """

        done = False

        if self.discrete:
            angles = np.asarray([self.angle_set[action]])
        else:
            move_angles = np.clip(action, self.action_space.low[0], self.action_space.high[0])
            angles = move_angles * np.pi

        self.heading_angles = angles

        movement = np.array([[self.movement_length * np.cos(angle),
                              self.movement_length * np.sin(angle)] for angle in angles])

        # Next intended position given the aforementioned movement #
        next_intended_positions = self.positions + movement

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        self.distance += self.movement_length

        if collision:

            self.number_of_collisions += 1

            if self.termination_condition or self.number_of_collisions == self.number_of_trials:
                done = True

            reward = -np.abs(self.collision_penalty)

        else:

            # Update the positions and waypoints
            self.positions = next_intended_positions
            self.waypoints = np.vstack((self.waypoints, [self.positions]))


            new_trajectories = None
            for k in range(self.number_of_agents):

                if new_trajectories is None:
                    new_trajectories = self.compute_trajectory(self.waypoints[-2, k], self.waypoints[-1, k])
                else:
                    new_trajectories = np.hstack(new_trajectories, self.compute_trajectory(self.waypoints[-2, k], self.waypoints[-1, k]))


            self.trajectories = np.hstack((self.trajectories, [new_trajectories]))

            if self.distance >= self.max_distance:
                done = True

            """ Take new measurements """
            new_measurements = self.measure()
            self.measured_values = np.vstack((self.measured_values, new_measurements))
            self.measured_locations = np.vstack((self.measured_locations, self.positions))

            """ Update the covariance matrix """
            self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations,
                                                             self.kernel, alpha=self.noise_factor)

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        return self.state, reward, done, {}

    @staticmethod
    def compute_trajectory(p1, p2):

        trajectory = None

        p = p1.astype(int)
        d = p2.astype(int) - p1.astype(int)
        N = np.max(np.abs(d))
        s = d / N

        for ii in range(0, N):
            p = p + s
            if trajectory is None:
                trajectory = np.array([np.rint(p)])
            else:
                trajectory = np.vstack((trajectory, [np.rint(p)]))

        return trajectory.astype(int)

    def reward(self):
        """
        The reward function

        :return: The information gain defined as Tr{t} - Tr{t+1}
        """

        information_gain = self.trace_ant - self.trace
        reward = -0.5 if information_gain < 0.01 else information_gain

        return reward

    def check_collision(self, next_positions):

        for position in next_positions:

            if self.navigation_map[int(position[0]), int(position[1])] == 0:
                return True

        return False

    def valid_action(self, action):

        if self.discrete:
            angles = np.asarray([self.angle_set[action]])
        else:
            move_angles = np.clip(action, self.action_space.low[0], self.action_space.high[0])
            angles = move_angles * np.pi

        self.heading_angles = angles

        movement = np.array([[self.movement_length * np.cos(angle),
                              self.movement_length * np.sin(angle)] for angle in angles])

        # Next intended position given the aforementioned movement #
        next_intended_positions = self.positions + movement

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        return not collision


    def render(self, mode="human"):

        plt.ion()

        if self.figure is None:
            self.figure, self.axs = plt.subplots(1,3)
            self.s0 = self.axs[0].imshow(self.state[0], cmap = 'gray')
            self.axs[0].plot(self.evaluation_locations[:,1], self.evaluation_locations[:,0], 'r.', alpha=0.3)
            self.s1 = self.axs[1].imshow(self.state[1], cmap = 'gray', vmin = 0.0, vmax=1.0)
            self.s2 = self.axs[2].imshow(self.state[2], cmap = 'coolwarm')

        else:

            self.s0.set_data(self.state[0])
            self.s1.set_data(self.state[1])
            self.s2.set_data(self.state[2])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        plt.pause(0.1)


class NonHomogeneousEntropyMinimization(BaseEntropyMinimization):

    def __init__(self,
                 navigation_map,
                 number_of_agents: int,
                 initial_positions: list,
                 movement_length: int,
                 density_grid: float,
                 noise_factor: float,
                 lengthscale: float,
                 initial_seed=0,
                 collision_penalty=-1,
                 max_distance=400,
                 ):

        super().__init__(navigation_map=navigation_map,
                         number_of_agents=number_of_agents,
                         initial_positions=initial_positions,
                         movement_length=movement_length,
                         density_grid=density_grid,
                         noise_factor=noise_factor,
                         lengthscale=lengthscale,
                         initial_seed=initial_seed,
                         collision_penalty=collision_penalty,
                         max_distance=max_distance)

        """ New observation space for the non-homogeneous case """
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3 + self.number_of_agents,
                                                                          self.navigation_map.shape[0],
                                                                          self.navigation_map.shape[1]))

        """ Support Vector Regressor for inference """
        self.Regressor = SVR(kernel="rbf", C=1E3, gamma='scale')

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
        """ Fit the regression model """
        self.Regressor.fit(self.measured_locations, self.measured_values.squeeze(axis=1))
        """ Estimate the model """
        estimated_values = self.Regressor.predict(self.visitable_locations)
        self.estimated_model = np.copy(self.navigation_map)
        self.estimated_model[self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = estimated_values
        """ Update the covariance matrix and the trace"""
        self.tr0 = np.sum(np.real(np.linalg.eigvals(self.kernel(self.evaluation_locations))))
        self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations,
                                                         self.kernel)
        self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))
        self.trace_ant = self.trace
        self.norm_rew_term = self.tr0 - self.trace
        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def step(self, action):
        """
            Move the vehicles according to the action
            :param action: The angle of the movement [-1,1]
            :return: next state, reward, done, info
            """

        done = False

        angles = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        movement = np.array([[self.movement_length * np.cos(np.pi * angle),
                              self.movement_length * np.sin(np.pi * angle)] for angle in angles])

        # Next intended position given the aforementioned movement #
        next_intended_positions = self.positions + movement

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        if collision:
            done = True
            reward = -np.abs(self.collision_penalty)
        else:

            # Update the positions
            self.positions = next_intended_positions

            self.distance += self.movement_length

            if self.distance >= self.max_distance:
                done = True

            """ Take new measurements """
            new_measurements = self.measure()
            self.measured_values = np.vstack((self.measured_values, new_measurements))
            self.measured_locations = np.vstack((self.measured_locations, self.positions))

            """ Fit the regression model """
            self.Regressor.fit(self.measured_locations, self.measured_values.squeeze(axis=1))

            """ Estimate the model """
            estimated_values = self.Regressor.predict(self.visitable_locations)

            self.estimated_model = np.copy(self.navigation_map)
            self.estimated_model[self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = estimated_values

            """ Update the covariance matrix """
            self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations,
                                                             self.kernel)

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        return self.state, reward, done, {}

    def update_state(self):

        state = np.zeros((3 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.clip(np.copy(self.navigation_map), 0.0, 1.0)

        """ The position of the vehicles """
        for i in range(0, self.number_of_agents):
            state[i + 1, int(self.positions[i, 0]), int(self.positions[i, 1])] = 1.0

        uncertainty = conditioning_std(self.visitable_locations, self.measured_locations, self.kernel)

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        estimated_model_normalized = (self.estimated_model - np.min(self.estimated_model)) / (
                np.max(self.estimated_model) - np.min(self.estimated_model))

        state[-2] = self.navigation_map - 1 + estimated_model_normalized

        return state

    def reward(self):
        """
        The reward function

        :return: The information gain defined as Tr{t} - Tr{t+1}
        """

        information_gain = self.trace_ant - self.trace

        """ Computed as: w = yt/max(y)"""
        normalized_y = (self.measured_values - np.min(self.measured_values)) / (
                np.max(self.measured_values) - np.min(self.measured_values))
        information_weight = np.clip(np.mean(normalized_y[-self.number_of_agents:]), 0.2, 1.0)

        return information_gain * information_weight / self.norm_rew_term

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


class BaseTemporalEntropyMinimization(BaseEntropyMinimization):

    def __init__(self,
                 navigation_map,
                 number_of_agents: int,
                 initial_positions: list,
                 movement_length: int,
                 density_grid: float = 0.1,
                 noise_factor: float = 0.0001,
                 lengthscale: float = 10,
                 initial_seed=0,
                 collision_penalty=-1,
                 number_of_trials=1,
                 number_of_actions=8,
                 max_distance=400,
                 random_init_point=False,
                 termination_condition=True,
                 discrete=True,
                 dt = 0.1,
                 ):

        super().__init__(navigation_map=navigation_map,
                         number_of_agents=number_of_agents,
                         initial_positions=initial_positions,
                         movement_length=movement_length,
                         density_grid=density_grid,
                         noise_factor=noise_factor,
                         lengthscale=lengthscale,
                         initial_seed=initial_seed,
                         collision_penalty=collision_penalty,
                         max_distance=max_distance)

        self.GroundTruth.dt = dt
        self.sample_times = None
        self.dt = dt

    def reset(self):

        """ Reset the environment """

        """ Reset distance """
        self.distance = 0
        self.number_of_collisions = 0
        """ New ground truth """
        self.GroundTruth.reset_gt()
        self.GroundTruth_field = self.GroundTruth.sample_gt()
        """ Starting positions """

        if self.random_initial_position:
            self.initial_positions = self.visitable_locations[
                np.random.choice(np.arange(0, len(self.visitable_locations)),
                                 self.number_of_agents, replace=False)]

        self.positions = np.copy(self.initial_positions)

        """ Heading angles """
        self.heading_angles = np.zeros((self.number_of_agents,))
        center = np.array(self.navigation_map.shape, dtype=int) / 2
        self.heading_angles = np.asarray([np.arctan2(center[1] - pos[1], center[0] - pos[0]) for pos in self.positions])
        self.waypoints = np.expand_dims(np.copy(self.initial_positions), 0)  # Trajectory of the agent #
        self.trajectories = np.copy(self.waypoints)
        """ Reset the measurements """
        self.measured_locations = np.copy(self.initial_positions)
        """ Take new measurements """
        self.measured_values = self.measure()
        self.sample_times = np.zeros_like(self.measured_values)
        """ Update the covariance matrix and the trace"""
        self.tr0 = np.sum(np.real(np.linalg.eigvals(self.kernel(self.evaluation_locations))))
        self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations, self.measured_locations,
                                                                   self.kernel, sample_times=self.sample_times,
                                                                   time=0.0, weights=1)
        self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

        self.trace_ant = self.trace

        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def update_state(self):

        state = np.zeros((2 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.copy(self.navigation_map)

        """ The position of the vehicles """

        for k in range(0, self.number_of_agents):
            w = np.linspace(0, 1, len(self.trajectories[0]))
            state[k + 1, self.trajectories[k, :, 0], self.trajectories[k, :, 1]] = w

        uncertainty = conditioning_std_with_time(self.visitable_locations, self.measured_locations, self.kernel,
                                                 sample_times=self.sample_times, time=np.max(self.sample_times), weights=1)

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        return state

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The angle of the movement [-1,1]
        :return: next state, reward, done, info
        """

        done = False

        if self.discrete:
            angles = np.asarray([self.angle_set[action]])
        else:
            move_angles = np.clip(action, self.action_space.low[0], self.action_space.high[0])
            angles = move_angles * np.pi

        self.heading_angles = angles

        movement = np.array([[self.movement_length * np.cos(angle),
                              self.movement_length * np.sin(angle)] for angle in angles])

        # Next intended position given the aforementioned movement #
        next_intended_positions = self.positions + movement

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        self.distance += self.movement_length

        if collision:

            self.number_of_collisions += 1

            if self.termination_condition or self.number_of_collisions == self.number_of_trials:
                done = True

            reward = -np.abs(self.collision_penalty)

        else:

            # Update the positions and waypoints
            self.positions = next_intended_positions
            self.waypoints = np.vstack((self.waypoints, [self.positions]))

            new_trajectories = None
            for k in range(self.number_of_agents):

                if new_trajectories is None:
                    new_trajectories = self.compute_trajectory(self.waypoints[-2, k], self.waypoints[-1, k])
                else:
                    new_trajectories = np.hstack(new_trajectories,
                                                 self.compute_trajectory(self.waypoints[-2, k], self.waypoints[-1, k]))

            self.trajectories = np.hstack((self.trajectories, [new_trajectories]))

            if self.distance >= self.max_distance:
                done = True

            """ Take new measurements """
            new_measurements = self.measure()
            self.measured_values = np.vstack((self.measured_values, new_measurements))
            self.measured_locations = np.vstack((self.measured_locations, self.positions))
            self.sample_times = np.vstack((self.sample_times, self.sample_times[-self.number_of_agents:] + self.dt))

            """ Update the covariance matrix """
            self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations,
                                                                       self.measured_locations,
                                                                       self.kernel, sample_times=self.sample_times,
                                                                       time=np.max(self.sample_times),weights=1)

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        self.GroundTruth.step()

        return self.state, reward, done, {}


class NonHomogeneousTemporalEntropyMinimization(BaseTemporalEntropyMinimization):

    def __init__(self,
                 navigation_map,
                 number_of_agents: int,
                 initial_positions: list,
                 movement_length: int,
                 density_grid: float,
                 noise_factor: float,
                 lengthscale: float,
                 initial_seed=0,
                 collision_penalty=-1,
                 max_distance=400,
                 dt=0.1
                 ):

        super().__init__(navigation_map=navigation_map,
                         number_of_agents=number_of_agents,
                         initial_positions=initial_positions,
                         movement_length=movement_length,
                         density_grid=density_grid,
                         noise_factor=noise_factor,
                         lengthscale=lengthscale,
                         initial_seed=initial_seed,
                         collision_penalty=collision_penalty,
                         max_distance=max_distance,
                         dt=dt)

        """ New observation space for the non-homogeneous case """
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3 + self.number_of_agents,
                                                                          self.navigation_map.shape[0],
                                                                          self.navigation_map.shape[1]))

        """ Support Vector Regressor for inference """
        self.Regressor = SVR(kernel="rbf", C=1E5, gamma='scale')

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
        self.sample_times = np.zeros_like(self.measured_values)
        """ Fit the regression model """
        self.Regressor.fit(self.measured_locations, self.measured_values.squeeze(axis=1))
        """ Estimate the model """
        self.estimated_model = self.Regressor.predict(self.visitable_locations)
        self.estimated_model_matrix = np.copy(self.navigation_map)
        self.estimated_model_matrix[
            self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = self.estimated_model

        """ Update the covariance matrix and the trace"""
        self.tr0 = np.sum(np.real(np.linalg.eigvals(self.kernel(self.evaluation_locations))))
        self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations, self.measured_locations,
                                                                   self.kernel, sample_times=self.sample_times,
                                                                   weights=np.array([1.0]),
                                                                   time=0.0)
        self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))
        self.trace_ant = self.trace
        self.norm_rew_term = self.tr0 - self.trace
        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def update_state(self):

        state = np.zeros((3 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.clip(np.copy(self.navigation_map), 0.0, 1.0)

        """ The position of the vehicles """
        for i in range(0, self.number_of_agents):
            state[i + 1, int(self.positions[i, 0]), int(self.positions[i, 1])] = 1.0

        weights = np.abs((self.measured_values - np.mean(self.measured_values)) / len(self.measured_values))
        if len(weights) > 1:
            weights = weights / np.linalg.norm(weights.squeeze(1))

        uncertainty = conditioning_std_with_time(self.visitable_locations, self.measured_locations, self.kernel,
                                                 sample_times=self.sample_times, time=np.max(self.sample_times),
                                                 weights=weights.squeeze(1))

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        estimated_model_normalized = (self.estimated_model - np.min(self.estimated_model) + 1E-5) / (
                np.max(self.estimated_model) - np.min(self.estimated_model) + 1E-5)

        state[-2] = np.copy(self.navigation_map) - 1
        state[-2, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = estimated_model_normalized

        return state

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The angle of the movement [-1,1]
        :return: next state, reward, done, info
        """

        done = False

        angles = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        movement = np.array([[self.movement_length * np.cos(np.pi * angle),
                              self.movement_length * np.sin(np.pi * angle)] for angle in angles])

        # Next intended position given the aforementioned movement #
        next_intended_positions = self.positions + movement

        # Compute if there is a collision #
        collision = self.check_collision(next_intended_positions)

        if collision:
            done = True
            reward = -np.abs(self.collision_penalty)
        else:

            # Update the positions
            self.positions = next_intended_positions

            self.distance += self.movement_length

            if self.distance >= self.max_distance:
                done = True

            """ Take new measurements """
            new_measurements = self.measure()
            self.measured_values = np.vstack((self.measured_values, new_measurements))
            self.measured_locations = np.vstack((self.measured_locations, self.positions))
            self.sample_times = np.vstack((self.sample_times, self.sample_times[-self.number_of_agents:] + self.dt))

            """ Fit the regression model """
            self.Regressor.fit(self.measured_locations, self.measured_values.squeeze(axis=1))

            """ Estimate the model """
            self.estimated_model = self.Regressor.predict(self.visitable_locations)
            self.estimated_model_matrix = np.copy(self.navigation_map)
            self.estimated_model_matrix[
                self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = self.estimated_model

            """ Update the covariance matrix """
            weights = np.abs((self.measured_values - np.mean(self.measured_values)) / len(self.measured_values))
            weights = weights / np.linalg.norm(weights.squeeze(1))
            self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations,
                                                                       self.measured_locations,
                                                                       self.kernel, sample_times=self.sample_times,
                                                                       weights=weights.squeeze(1),
                                                                       time=np.max(self.sample_times))

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        return self.state, reward, done, {}

    def render(self, mode="human", first=True):

        if first:
            fig, self.axs = plt.subplots(1, 2)

        if mode == 'human':

            colors = matplotlib.cm.get_cmap('jet', self.number_of_agents)
            colors = colors(range(self.number_of_agents))
            unc = self.state[-1]
            unc[self.navigation_map == 0] = np.nan
            self.axs[0].imshow(self.state[-1], cmap='gray_r')
            self.axs[0].set_title("Uncertainty")
            model = self.state[-2]
            model[self.navigation_map == 0] = np.nan
            self.axs[1].imshow(self.state[-2], cmap='coolwarm')
            self.axs[1].set_title("Model")

            for color, pos in zip(colors, self.positions):
                self.axs[0].plot(pos[1], pos[0], '.', color=color, markersize=1)
                self.axs[1].plot(pos[1], pos[0], '.', color=color, markersize=1)

    def compute_mse(self):

        mse = mean_squared_error(
            y_true=self.GroundTruth_field[self.visitable_locations[:, 0], self.visitable_locations[:, 1]],
            y_pred=self.estimated_model)

        return mse


if __name__ == '__main__':

    from utils import plot_trajectory


    np.random.seed(2)

    """ Create the environment """
    initial_position = [[30, 7]]
    navigation_map = np.genfromtxt('./ypacarai_map_middle.csv')
    env = BaseTemporalEntropyMinimization(navigation_map=navigation_map,
                                  number_of_agents=1,
                                  initial_positions=initial_position,
                                  movement_length=3,
                                  density_grid=0.2,
                                  noise_factor=1E-2,
                                  lengthscale=5,
                                  initial_seed=0,
                                  max_distance=1200,
                                  random_init_point=False,
                                  termination_condition = False,
                                  number_of_trials=5,
                                  discrete=True,
                                  dt = 0.05)

    """ Reset! """
    env.reset()

    # Render environment #
    env.render()
    plt.pause(0.3)

    r = -1
    d = False
    valid = False
    Racc = 0
    R = []

    actions = env.action_space.sample()

    while not d:

        # Compute next valid position #
        while not env.valid_action(actions):
            actions = env.action_space.sample()
        s, r, d, _ = env.step(actions)
        Racc += r
        R.append(Racc)
        print('Reward: ', r)
        env.render()
        plt.pause(0.1)

        # Render environment #

    # Final renders #
    env.render()
    plot_trajectory(env.axs[2], env.waypoints[:, 0, 1], env.waypoints[:, 0, 0], z=None, colormap ='jet', num_of_points = 200, linewidth = 1, k = 1, plot_waypoints=False, markersize = 0.5)

    plt.show(block=True)

    plt.plot(R, 'b-o')
    plt.grid()
    plt.show(block=True)



