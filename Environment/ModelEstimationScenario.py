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


class ContinuousVehicle:

    def __init__(self, initial_position, movement_length_interval, navigation_map):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.heading_angle = 0
        self.waypoints = np.expand_dims(np.copy(initial_position),0)
        self.trajectory = np.copy(self.waypoints)

        self.distance = 0.0
        self.num_of_collisions = 0
        self.action_space = gym.spaces.Box(low = -1, high = 1, shape=(2,))
        self.movement_length_interval = movement_length_interval
        self.navigation_map = navigation_map

    def move(self, action):

        # Longitud del movimiento
        length = self.movement_length_interval[0] + ((self.movement_length_interval[1] - self.movement_length_interval[0])/2) * (action[1] + 1)
        self.distance += length
        # Ángulo de desplazamiento
        angle = ((2*np.pi - 0)/2)*(action[0] + 1)
        self.heading_angle = angle

        movement = np.array([length * np.cos(angle), length * np.sin(angle)])
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

        next_position = np.clip(next_position, a_min=(0,0), a_max=np.array(self.navigation_map.shape)-1)
        if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
            return True
        return False

    def update_trajectory(self):

        p1 = self.waypoints[-2]
        p2 = self.waypoints[-1]

        mini_traj = self.compute_trajectory_between_points(p1,p2)

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

        # Longitud del movimiento
        length = self.movement_length_interval[0] + (
                    (self.movement_length_interval[1] - self.movement_length_interval[0]) / 2) * (action[1] + 1)
        # Ángulo de desplazamiento
        angle = ((2 * np.pi - 0) / 2) * (action[0] + 1)

        movement = np.array([length * np.cos(angle), length * np.sin(angle)])
        next_position = self.position + movement

        return self.check_collision(next_position)

    def move_to_position(self, goal_position):

        """ Add the distance """
        assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
        self.distance += np.linalg.norm(goal_position-self.position)
        """ Update the position """
        self.position = goal_position


class ContinuousFleet:

    def __init__(self, number_of_vehicles, initial_positions, movement_length_interval, navigation_map):

        self.number_of_vehicles = number_of_vehicles
        self.initial_positions = initial_positions

        self.movement_length_interval = movement_length_interval
        self.vehicles = [ContinuousVehicle(initial_position=initial_positions[k],
                                         movement_length_interval=movement_length_interval,
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


class GaussianProcessModeling(gym.Env, ABC):

    def __init__(self,
                 navigation_map,
                 number_of_agents,
                 initial_positions,
                 movement_length_interval,
                 noise_factor,
                 lengthscale,
                 initial_seed=0,
                 max_distance=1000,
                 random_init_point=False,
                 termination_condition=False,
                 number_of_trials=5,):

        self.navigation_map = navigation_map
        self.number_of_agents = number_of_agents
        self.initial_positions = initial_positions
        self.movement_length_interval = movement_length_interval
        self.noise_factor = noise_factor
        self.lengthscale = lengthscale
        self.initial_seed = initial_seed
        self.max_distance = max_distance
        self.random_init_point = random_init_point
        self.termination_condition = termination_condition
        self.number_of_trials = number_of_trials
        self.collision_penalty = -1
        self.figure = None

        """ Check every initial position """
        for i, pos in enumerate(self.initial_positions):
            assert self.navigation_map[pos[0], pos[1]] == 1, f"Impossible position for drone {i}."

        """ Positions where is it possible to measure """
        self.visitable_locations = np.vstack(np.where(self.navigation_map != 0)).T

        mesh = np.meshgrid(np.arange(0, navigation_map.shape[0]),
                           np.arange(0, navigation_map.shape[1]))
        self.all_locations = np.vstack((mesh[0].flatten(), mesh[1].flatten())).T

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

        """ Fleet attributes """
        self.fleet = ContinuousFleet(number_of_vehicles=self.number_of_agents,
                                     initial_positions=self.initial_positions,
                                     movement_length_interval=self.movement_length_interval,
                                     navigation_map=navigation_map)

        """ Gaussian Process Regressor """
        self.GP = GaussianProcessRegressor(kernel=self.kernel, alpha=self.noise_factor)

        self.measured_locations = None
        self.measured_values = None
        self.GroundTruth_field = None
        self.mse = 0
        self.mu = None
        self.sigma = None
        self.state = None

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
        """ Starting positions """

        if self.random_init_point:
            self.initial_positions = self.visitable_locations[
                np.random.choice(np.arange(0, len(self.visitable_locations)),self.number_of_agents, replace=False)]

        """ Reset fleet """
        self.fleet.reset(initial_positions=self.initial_positions)
        """ Take new measurements """
        self.measured_values, self.measured_locations = self.fleet.measure(gt_field=self.GroundTruth_field)
        """ Fit the GP """
        self.GP.fit(X=self.measured_locations, y=self.measured_values)
        """ Predict GP """
        self.mu, self.sigma = self.GP.predict(self.all_locations, return_std=True)
        """ Compute the mse """
        self.mse = mean_squared_error(y_true=self.GroundTruth_field[np.where(self.navigation_map==1)],
                                      y_pred=self.mu.reshape(self.navigation_map.shape)[np.where(self.navigation_map == 1)])
        """ Produce new state """
        self.state = self.update_state()

        return self.state

    def update_state(self):

        state = np.zeros((3 + self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        """ The boundaries of the map """
        state[0] = np.copy(self.navigation_map)

        """ The mean of the map """

        mu_img = self.mu.reshape((self.navigation_map.shape[1], self.navigation_map.shape[0])).T
        state[1] = (mu_img - self.mu.min())/(self.mu.max() - self.mu.min())

        """ The unct of the map """
        std_img = self.sigma.reshape((self.navigation_map.shape[1], self.navigation_map.shape[0])).T
        state[2] = (std_img - self.sigma.min()) / (self.sigma.max() - self.sigma.min())

        """ The position of the vehicles """
        for k in range(0, self.number_of_agents):

            if len(self.fleet.vehicles[k].trajectory) > 1:
                w = np.linspace(0, 1, len(self.fleet.vehicles[k].trajectory))
            else:
                w = 1.0

            state[k + 3, self.fleet.vehicles[k].trajectory[:, 0], self.fleet.vehicles[k].trajectory[:, 1]] = w


        return state

    def step(self, action):
        """
        Move the vehicles according to the action
        :param action: The continuous movement
        :return: next state, reward, done, info
        """

        done = False

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

            """ Fit the GP """
            self.GP.fit(X=self.measured_locations, y=self.measured_values)
            """ Predict GP """
            self.mu, self.sigma = self.GP.predict(self.all_locations, return_std=True)
            """ Compute the mse """
            self.mse = mean_squared_error(y_true=self.GroundTruth_field[np.where(self.navigation_map == 1)],
                                          y_pred=self.mu.reshape(self.navigation_map.shape)[np.where(self.navigation_map == 1)])

            """ Compute reward """
            reward = self.reward()

        """ Produce new state """
        self.state = self.update_state()

        return self.state, reward, done, {}

    def reward(self):

        return -self.mse

    def render(self, mode="human"):

        plt.ion()

        if self.figure is None:
            self.figure, self.axs = plt.subplots(1, 4)
            self.s0 = self.axs[0].imshow(self.state[0], cmap='gray', vmin=0.0, vmax=1.0)
            self.s1 = self.axs[1].imshow(self.state[1], cmap='gray', vmin=0.0, vmax=1.0)
            self.s2 = self.axs[2].imshow(self.state[2], cmap='gray', vmin=0.0, vmax=1.0)
            self.s3 = self.axs[3].imshow(self.state[3], cmap='gray', vmin=0.0, vmax=1.0)

        else:

            self.s0.set_data(self.state[0])
            self.s1.set_data(self.state[1])
            self.s2.set_data(self.state[2])
            self.s3.set_data(self.state[3])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        plt.pause(0.1)




if __name__ == '__main__':
    """ Create the environment """
    initial_position = [[36, 29],[36,19]]
    navigation_map = np.genfromtxt('./ypacarai_map_middle.csv')

    env =  GaussianProcessModeling(navigation_map = navigation_map,
                 number_of_agents = 2,
                 initial_positions = initial_position,
                 movement_length_interval = [3,20],
                 noise_factor=1E-5,
                 lengthscale=5,
                 initial_seed=0,
                 max_distance=100,
                 random_init_point=False,
                 termination_condition=False,
                 number_of_trials=5,)


    env.reset()
    env.render()

    done = False
    while not done:
        env.step(np.random.rand(2,2)*2 - 1)
        env.render()

    plt.show(block=True)
