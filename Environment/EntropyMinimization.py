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


class DiscreteVehicle:

    def __init__(self, initial_position, n_actions, movement_length, navigation_map):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.trajectory = np.copy(self.waypoints)

        self.distance = 0.0
        self.num_of_collisions = 0
        self.action_space = gym.spaces.Discrete(n_actions)
        self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
        self.movement_length = movement_length
        self.navigation_map = navigation_map

    def move(self, action):

        self.distance += self.movement_length
        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
        next_position = self.position + movement

        if self.check_collision(next_position):
            collide = True
            self.num_of_collisions += 1
        else:
            collide = False
            self.position = next_position
            self.waypoints = np.vstack((self.waypoints, [self.position]))
            self.update_trajectory()

        return collide

    def check_collision(self, next_position):

        if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
            return True
        return False

    def update_trajectory(self):

        p1 = self.waypoints[-2]
        p2 = self.waypoints[-1]

        mini_traj = self.compute_trajectory_between_points(p1, p2)

        self.trajectory = np.vstack((self.trajectory, mini_traj))

    @staticmethod
    def compute_trajectory_between_points(p1, p2):
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

    def reset(self, initial_position):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.trajectory = np.copy(self.waypoints)
        self.distance = 0.0
        self.num_of_collisions = 0

    def check_action(self, action):

        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
        next_position = self.position + movement

        return self.check_collision(next_position)

    def move_to_position(self, goal_position):

        """ Add the distance """
        assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
        self.distance += np.linalg.norm(goal_position - self.position)
        """ Update the position """
        self.position = goal_position


class DiscreteFleet:

    def __init__(self, number_of_vehicles, n_actions, initial_positions, movement_length, navigation_map):

        self.number_of_vehicles = number_of_vehicles
        self.initial_positions = initial_positions
        self.n_actions = n_actions
        self.movement_length = movement_length
        self.vehicles = [DiscreteVehicle(initial_position=initial_positions[k],
                                         n_actions=n_actions,
                                         movement_length=movement_length,
                                         navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def move(self, fleet_actions):

        collision_array = [self.vehicles[k].move(fleet_actions[k]) for k in range(self.number_of_vehicles)]

        self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

        return collision_array

    def measure(self, gt_field):

        """
        Take a measurement in the given N positions
        :param gt_field:
        :return: An numpy array with dims (N,2)
        """
        positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

        values = []
        for pos in positions:
            values.append([gt_field[int(pos[0]), int(pos[1])]])

        if self.measured_locations is None:
            self.measured_locations = positions
            self.measured_values = values
        else:
            self.measured_locations = np.vstack((self.measured_locations, positions))
            self.measured_values = np.vstack((self.measured_values, values))

        return self.measured_values, self.measured_locations

    def reset(self, initial_positions):

        for k in range(self.number_of_vehicles):
            self.vehicles[k].reset(initial_position=initial_positions[k])

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def get_distances(self):

        return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

    def check_collisions(self, test_actions):

        return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

    def move_fleet_to_positions(self, goal_list):
        """ Move the fleet to the given positions.
         All goal positions must ve valid. """

        goal_list = np.atleast_2d(goal_list)

        for k in range(self.number_of_vehicles):
            self.vehicles[k].move_to_position(goal_position=goal_list[k])


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
                 random_init_point=False,
                 termination_condition=True
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
        self.action_space = gym.spaces.Discrete(number_of_actions)
        self.angle_set = np.linspace(0, 2 * np.pi, number_of_actions, endpoint=False)

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

        """ Fleet attributes """
        self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
                                   n_actions=number_of_actions,
                                   initial_positions=initial_positions,
                                   movement_length=movement_length,
                                   navigation_map=navigation_map)

        """ Other variables """
        self.measured_locations = None
        self.measured_values = None
        self.covariance_matrix = None
        self.trace = None
        self.trace_ant = None
        self.state = None
        self.tr0 = None
        self.norm_rew_term = None
        self.figure, self.axs = None, None
        self.random_peaks = None
        self.seed(self.initial_seed)

    def seed(self, seed=None):
        np.random.seed(seed)

    def sample_action_space(self):

        return [self.fleet.vehicles[k].action_space.sample() for k in range(self.number_of_agents)]

    def reset(self):

        """ Reset the environment """

        """ New ground truth """
        self.GroundTruth.reset_gt()
        self.GroundTruth_field = self.GroundTruth.sample_gt()

        """ 10 random located events in the ground truth """
        self.random_peaks = self.visitable_locations[np.random.choice(np.arange(0, len(self.visitable_locations)),
                                                                      10,
                                                                      replace=False)]

        """ Starting positions """

        if self.random_initial_position:
            self.initial_positions = self.visitable_locations[
                np.random.choice(np.arange(0, len(self.visitable_locations)),
                                 self.number_of_agents, replace=False)]

        """ Reset fleet """
        self.fleet.reset(initial_positions=self.initial_positions)
        """ Take new measurements """
        self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)
        """ Update the covariance matrix and the trace"""
        self.tr0 = np.sum(np.real(np.linalg.eigvals(self.kernel(self.evaluation_locations))))
        self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations,
                                                         self.kernel, alpha=self.noise_factor)

        self.trace = self.covariance_matrix.trace()

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

            if len(self.fleet.vehicles[k].trajectory) > 1:
                w = np.linspace(0, 1, len(self.fleet.vehicles[k].trajectory))
            else:
                w = 1.0

            state[k + 1, self.fleet.vehicles[k].trajectory[:, 0], self.fleet.vehicles[k].trajectory[:, 1]] = w

        uncertainty = conditioning_std(self.visitable_locations, self.measured_locations, self.kernel)

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        return state

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The categorical movement
        :return: next state, reward, done, info
        """

        done = False

        if not isinstance(action, list):
            action = [action]

        collition_mask = self.fleet.move(action)

        if any(collition_mask):

            if self.termination_condition or self.fleet.fleet_collisions >= self.number_of_trials:
                done = True

            reward = -np.abs(self.collision_penalty)

        else:

            if any(np.array(self.fleet.get_distances()) >= self.max_distance):
                done = True

            """ Take new measurements """
            self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)

            """ Update the covariance matrix """
            self.covariance_matrix = conditioning_cov_matrix(self.evaluation_locations, self.measured_locations,
                                                             self.kernel, alpha=self.noise_factor)

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = self.covariance_matrix.trace()

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        return self.state, reward, done, self.metrics()

    def reward(self):
        """
        The reward function

        :return: The information gain defined as Tr{t} - Tr{t+1}
        """

        information_gain = self.trace_ant - self.trace
        reward = -0.5 if information_gain < 0.01 else information_gain

        return reward

    def render(self, mode="human"):

        assert self.is_eval, "Eval mode must be activated to render - env.is_eval must be True!"
        plt.ion()


        if self.figure is None:
            self.figure, self.axs = plt.subplots(1, 4)
            self.s0 = self.axs[0].imshow(self.state[0], cmap='gray')
            self.axs[0].plot(self.evaluation_locations[:, 1], self.evaluation_locations[:, 0], 'r.', alpha=0.3)
            self.s1 = self.axs[1].imshow(self.state[1], cmap='gray', vmin=0.0, vmax=1.0)
            self.s2 = self.axs[2].imshow(self.state[-1], cmap='coolwarm')
            self.s3 = self.axs[3].imshow(self.GroundTruth_field, cmap='viridis')
            self.s4 = self.axs[2].scatter(self.random_peaks[:, 1], self.random_peaks[:, 0], c='Black')
            self.s5 = self.axs[2].scatter(self.fleet.vehicles[0].position[1], self.fleet.vehicles[0].position[0], c='Yellow')

        else:

            self.s0.set_data(self.state[0])
            self.s1.set_data(self.state[1])
            self.s2.set_data(self.state[-1])
            self.s3.set_data(self.GroundTruth_field)
            peaks = np.copy(self.random_peaks)
            peaks[:,0] = self.random_peaks[:,1]
            peaks[:, 1] = self.random_peaks[:, 0]
            self.s4.set_offsets(peaks)
            self.s5.set_offsets((self.fleet.vehicles[0].position[1],self.fleet.vehicles[0].position[0]))
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        plt.pause(0.1)

    def valid_action(self, a):
        assert self.number_of_agents == 1, "Not implemented for Multi-Agent!"

        # Return the action valid flag #
        return not self.fleet.check_collisions([a])[0]

    def step_to_position(self, desired_positions):
        """ Travel to the given position and take a sample """

        done = False

        self.fleet.move_fleet_to_positions(desired_positions)

        """ Take new measurements """
        self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)

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

        if any(np.array(self.fleet.get_distances()) >= self.max_distance):
            done = True

        return self.state, reward, done, self.metrics()

    def metrics(self):

        metric_dict = {"Entropy": self.trace,
                       "Area": np.sum(
                           self.state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] < 0.05),
                       "DetectionRate": np.sum(
                           self.state[-1, self.random_peaks[:, 0].astype(int), self.random_peaks[:, 1].astype(int)] < 0.05) / 10
                       }

        return metric_dict

    def get_action_mask(self):

        mask = np.array(list(map(self.valid_action, np.arange(self.action_space.n))))
        return mask


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
                 dt=0.1,
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
                         number_of_trials=number_of_trials,
                         number_of_actions=number_of_actions,
                         max_distance=max_distance,
                         random_init_point=random_init_point,
                         termination_condition=termination_condition)

        self.random_peaks_detected = None
        self.GroundTruth.dt = 0.005
        self.sample_times = None
        self.dt = dt
        self.metrics_dict = None
        self.is_eval = False

    def reset(self):
        """ Reset the environment """

        """ New ground truth """
        self.GroundTruth.reset_gt()
        self.GroundTruth_field = self.GroundTruth.sample_gt()

        """ 10 random located events in the ground truth """
        self.random_peaks = np.copy(self.visitable_locations[np.random.choice(np.arange(0, len(self.visitable_locations)),10,replace=False)]).astype(float)
        
        

        self.random_peaks_detected = np.zeros((10,), dtype=np.bool)

        """ Starting positions """

        if self.random_initial_position:
            self.initial_positions = self.visitable_locations[
                np.random.choice(np.arange(0, len(self.visitable_locations)),
                                 self.number_of_agents, replace=False)]

        """ Reset fleet """
        self.fleet.reset(initial_positions=self.initial_positions)
        """ Take new measurements """
        self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)
        self.sample_times = np.zeros_like(self.measured_values)
        """ Update the covariance matrix and the trace"""
        self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations, self.measured_locations,
                                                                   self.kernel, sample_times=self.sample_times,
                                                                   time=0.0, weights=1)
        self.trace = self.covariance_matrix.trace()

        self.trace_ant = self.trace

        """ Produce new state """
        self.state = self.update_state()

        """ Initialize metrics to integrate """

        distance = np.linalg.norm(self.fleet.vehicles[0].position - self.random_peaks, axis=1)
        detected = np.logical_and(distance < 3, self.state[-1, self.random_peaks[:, 0].astype(int), self.random_peaks[:, 1].astype(int)] < 0.05)
        self.random_peaks_detected = np.logical_or(detected, self.random_peaks_detected)

        self.metrics_dict = {"Entropy": self.trace,
                             "Area": np.sum(
                                 self.state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] < 0.05),
                             "DetectionRate": self.random_peaks_detected.sum() / 10
                             }

        return self.state

    def update_state(self):

        state = np.zeros((2 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.copy(self.navigation_map)

        """ The position of the vehicles """
        for k in range(0, self.number_of_agents):

            if len(self.fleet.vehicles[k].trajectory) > 1:
                w = np.linspace(0, 1, len(self.fleet.vehicles[k].trajectory))
            else:
                w = 1.0

            state[k + 1, self.fleet.vehicles[k].trajectory[:, 0], self.fleet.vehicles[k].trajectory[:, 1]] = w

        uncertainty = conditioning_std_with_time(self.visitable_locations, self.measured_locations, self.kernel,
                                                 sample_times=self.sample_times, time=np.max(self.sample_times),
                                                 weights=1)

        state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] = uncertainty

        return state

    def step(self, action):

        """
        Move the vehicles according to the action
        :param action: The categorical movement
        :return: next state, reward, done, info
        """

        if not isinstance(action, list):
            action = [action]

        done = False

        collition_mask = self.fleet.move(action)

        if any(collition_mask):

            if self.termination_condition or self.fleet.fleet_collisions >= self.number_of_trials:
                done = True

            reward = -np.abs(self.collision_penalty)

        else:

            if any(np.array(self.fleet.get_distances()) >= self.max_distance):
                done = True

            """ Take new measurements with time """
            self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)
            self.sample_times = np.vstack((self.sample_times, self.sample_times[-self.number_of_agents:] + self.dt))

            """ Update the covariance matrix """
            self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations,
                                                                       self.measured_locations,
                                                                       self.kernel, sample_times=self.sample_times,
                                                                       time=np.max(self.sample_times), weights=1)

            """ Update the trace """
            self.trace_ant = self.trace
            self.trace = self.covariance_matrix.trace()

            """ Compute reward """
            reward = self.reward()

            """ Update ground truth dynamic """
            if self.is_eval:
                self.GroundTruth.step()
                self.GroundTruth_field = self.GroundTruth.sample_gt()

                for j in range(len(self.random_peaks)):

                    valid_pos = False
                    while not valid_pos:
                        new_position = self.random_peaks[j] + (2*np.random.rand(2) - 1.0) * 0.5
                        valid_pos = self.navigation_map[new_position[0].astype(int), new_position[1].astype(int)] == 1.0

                    self.random_peaks[j, :] = new_position


        """ Produce new state """
        self.state = self.update_state()

        if self.is_eval:
            metrics = self.update_metrics()
        else:
            metrics = {}

        return self.state, reward, done, metrics

    def step_to_position(self, desired_positions):
        """ Travel to the given position and take a sample """

        done = False

        self.fleet.move_fleet_to_positions(desired_positions)

        """ Take new measurements """
        self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)
        self.sample_times = np.vstack((self.sample_times, self.sample_times[-self.number_of_agents:] + self.dt))

        """ Update the covariance matrix """
        self.covariance_matrix = conditioning_cov_matrix_with_time(self.evaluation_locations,
                                                                   self.measured_locations,
                                                                   self.kernel, sample_times=self.sample_times,
                                                                   time=np.max(self.sample_times), weights=1)

        """ Update the trace """
        self.trace_ant = self.trace
        self.trace = np.sum(np.real(np.linalg.eigvals(self.covariance_matrix)))

        """ Compute reward """
        reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        if any(np.array(self.fleet.get_distances()) >= self.max_distance):
            done = True

        return self.state, reward, done, {}

    def reward(self):
        """
        The reward function

        :return: The information gain defined as Tr{t} - Tr{t+1}
        """

        information_gain = self.trace_ant - self.trace
        reward = -0.5 if information_gain < 0.01 else information_gain

        return reward

    def update_metrics(self):

        self.metrics_dict["Entropy"] = self.trace
        self.metrics_dict["Area"] += np.sum(self.state[-1, self.visitable_locations[:, 0], self.visitable_locations[:, 1]] < 0.05)
        distance = np.linalg.norm(self.fleet.vehicles[0].position - self.random_peaks, axis=1)
        detected = np.logical_and(distance < 3, self.state[
            -1, self.random_peaks[:, 0].astype(int), self.random_peaks[:, 1].astype(int)] < 0.05)
        self.random_peaks_detected = np.logical_or(detected, self.random_peaks_detected)
        self.random_peaks_detected = np.logical_or(detected, self.random_peaks_detected)
        self.metrics_dict["DetectionRate"] = self.random_peaks_detected.sum() / 10

        return self.metrics_dict


if __name__ == '__main__':

    from utils import plot_trajectory

    np.random.seed(2)

    """ Create the environment """
    initial_position = [[36, 29]]
    navigation_map = np.genfromtxt('./ypacarai_map_middle.csv')
    env = BaseTemporalEntropyMinimization(navigation_map=navigation_map,
                                  number_of_agents=1,
                                  number_of_actions=8,
                                  initial_positions=initial_position,
                                  movement_length=3,
                                  density_grid=0.2,
                                  noise_factor=1E-2,
                                  lengthscale=5,
                                  initial_seed=0,
                                  max_distance=400,
                                  random_init_point=False,
                                  termination_condition=False,
                                  number_of_trials=5,
                                  dt=0.03)

    env.is_eval = True

    """ Reset! """
    env.reset()
    env.reset()

    env.render()
    plt.pause(0.1)

    r = -1
    d = False
    valid = False
    Racc = 0
    R = []


    while not d:
        # Compute next valid position #

        actions = env.sample_action_space()

        s, r, d, m = env.step(env.action_space.sample())
        Racc += r
        R.append(m['Entropy'])
        print('Reward: ', r)
        env.render()
        plt.pause(0.1)

        # Render environment #

    # Final renders #
    env.render()
    plot_trajectory(env.axs[2], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0], z=None,
                    colormap='jet',
                    num_of_points=500, linewidth=4, k=3, plot_waypoints=False, markersize=0.5)

    plt.show(block=True)

    plt.plot(R, 'b-o')
    plt.grid()
    plt.show(block=True)
