from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

import EnvironmentUtils
from ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer
from Networks.network import VisualNetwork, NoisyVisualNetwork, DuelingVisualNetwork, NoisyDuelingVisualNetwork
import torch.nn.functional as F


class DuelingDQNAgent:

	def __init__(
			self,
			env: gym.Env,
			memory_size: int,
			batch_size: int,
			target_update: int,
			soft_update: bool = False,
			tau: float = 0.0001,
			epsilon_values: List[float] = [1.0, 0.0],
			epsilon_interval: List[float] = [0.0, 1.0],
			learning_starts: int = 10,
			gamma: float = 0.99,
			lr: float = 1e-4,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			# NN parameters
			number_of_features: int = 1024,
			noisy: bool = False,
			logdir=None,
			log_name="Experiment",
			safe_actions=False,
			train_every=1,
	):
		"""

		:param env: Environment to optimize
		:param memory_size: Size of the experience replay
		:param batch_size: Mini-batch size for SGD steps
		:param target_update: Number of episodes between updates of the target
		:param soft_update: Flag to activate the Polyak update of the target
		:param tau: Polyak update constant
		:param gamma: Discount Factor
		:param lr: Learning Rate
		:param alpha: Randomness of the sample in the PER
		:param beta: Bias compensating constant in the PER weights
		:param prior_eps: Minimal probability for every experience to be samples
		:param number_of_features: Number of features after the visual extractor
		:param logdir: Directory to save the tensorboard log
		:param log_name: Name of the tb log
		:param safe_actions: Safe action flag
		"""

		""" Logging parameters """
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None

		""" Observation space dimensions """
		obs_dim = env.observation_space.shape
		action_dim = env.action_space.n

		""" Agent embeds the environment """
		self.safe_action = safe_actions
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.soft_update = soft_update
		self.tau = tau
		self.gamma = gamma
		self.learning_rate = lr
		self.epsilon_values = epsilon_values
		self.epsilon_interval = epsilon_interval
		self.epsilon = self.epsilon_values[0]
		self.learning_starts = learning_starts
		self.noisy = noisy
		self.train_every = train_every

		""" Automatic selection of the device """
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		self.beta = beta
		self.prior_eps = prior_eps
		self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)

		""" Create the DQN and the DQN-Target (noisy if selected) """
		if self.noisy:
			self.dqn = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
		else:
			self.dqn = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)

		self.dqn_target.load_state_dict(self.dqn.state_dict())
		self.dqn_target.eval()

		""" Optimizer """
		self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		""" Actual list of transitions """
		self.transition = list()

		""" Evaluation flag """
		self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0

		# Sample new noisy parameters
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

	# TODO: Implement an annealed Learning Rate (see:
	#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)

	def select_action(self, state: np.ndarray) -> np.ndarray:
		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy:
			selected_action = self.env.action_space.sample()
		elif not self.safe_action:
			selected_action = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax()
			selected_action = selected_action.detach().cpu().numpy()
		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			mask = self.env.get_action_mask()
			q_values.squeeze(0)[mask == False] = -np.inf
			selected_action = np.argmax(q_values)

		return selected_action

	def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		next_state, reward, done, _ = self.env.step(action)

		return next_state, reward, done

	def update_model(self) -> torch.Tensor:
		"""Update the model by gradient descent."""

		# PER needs beta to calculate weights
		samples = self.memory.sample_batch(self.beta)
		weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
		indices = samples["indices"]

		# PER: importance sampling before average
		elementwise_loss = self._compute_dqn_loss(samples)
		loss = torch.mean(elementwise_loss * weights)

		# Compute gradients and apply them
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# PER: update priorities
		loss_for_prior = elementwise_loss.detach().cpu().numpy()
		new_priorities = loss_for_prior + self.prior_eps
		self.memory.update_priorities(indices, new_priorities)

		# Sample new noisy distribution
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		return loss.item()

	@staticmethod
	def anneal_epsilon(p, p_init=0.1, p_fin=0.9, e_init=1.0, e_fin=0.0):

		if p < p_init:
			return e_init
		elif p > p_fin:
			return e_fin
		else:
			return (e_fin - e_init) / (p_fin - p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init

	def train(self, episodes):
		""" Train the agent. """

		# Optimization steps #
		steps = 0
		# Create train logger #
		if self.writer is None:
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)
		# Agent in training mode #
		self.is_eval = False
		# Reset episode count #
		self.episode = 1
		# Reset metrics #
		episodic_reward_vector = []
		record = -np.inf

		for episode in range(1, int(episodes) + 1):

			done = False
			state = self.env.reset()
			score = 0
			length = 0
			losses = []

			# Initially sample noisy policy #
			if self.noisy:
				self.dqn.reset_noise()
				self.dqn_target.reset_noise()

			# PER: Increase beta temperature
			self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

			# Epsilon greedy annealing
			self.epsilon = self.anneal_epsilon(p=episode / episodes,
			                                   p_init=self.epsilon_interval[0],
			                                   p_fin=self.epsilon_interval[1],
			                                   e_init=self.epsilon_values[0],
			                                   e_fin=self.epsilon_values[1])

			# Run an episode #
			while not done:

				# Inrease the plaed steps #
				steps += 1

				# Select the action using the current policy
				action = self.select_action(state)

				# Perform the action and retrieve the next state. #
				if self.safe_action:
					safe_mask = self.env.get_action_mask()

				next_state, reward, done = self.step(action)

				if self.safe_action:
					safe_mask_ = self.env.get_action_mask()

				# Compute the safe mask for the transition #
				if self.safe_action:
					info = {'safe_mask': safe_mask,
					        'safe_mask_': safe_mask_}
				else:
					info = {}

				# Store the transition
				self.transition = [state, action, reward, next_state, done, info]
				self.memory.store(*self.transition)

				# Update the state
				state = next_state
				# Accumulate indicators
				score += reward
				length += 1

				# if episode ends
				if done:

					# Append loss metric #
					if losses:
						self.episodic_loss = np.mean(losses)

					# Compute average metrics #
					self.episodic_reward = score
					self.episodic_length = length
					episodic_reward_vector.append(self.episodic_reward)
					self.episode += 1

					# Log progress
					self.log_data()

					# Save policy if is better on average
					mean_episodic_reward = np.mean(episodic_reward_vector[-50:])
					if mean_episodic_reward > record:
						print(f"New best policy with mean reward of {mean_episodic_reward}")
						print("Saving model in " + self.writer.log_dir)
						record = mean_episodic_reward
						self.save_model(name='BestPolicy.pth')

				# If training is ready
				if len(self.memory) >= self.batch_size and episode >= self.learning_starts and steps % self.train_every == 0:

					# Update model parameters by backprop-bootstrapping #
					loss = self.update_model()
					# Append loss #
					losses.append(loss)

					# Update target soft/hard #
					if self.soft_update:
						self._target_soft_update()
					elif episode % self.target_update == 0 and done:
						self._target_hard_update()

		# Save the final policy #
		self.save_model(name='FINALPolicy.pth')

	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		state = torch.FloatTensor(samples["obs"]).to(device)
		next_state = torch.FloatTensor(samples["next_obs"]).to(device)
		action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
		reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise

		curr_q_value = self.dqn(state).gather(1, action)
		done_mask = 1 - done

		with torch.no_grad():

			if self.safe_action:
				info_dictionaries = samples["info"]
				unsafe_next_q_values = self.dqn_target(next_state)

				for mask_dictionary, q_values in zip(info_dictionaries, unsafe_next_q_values):

					q_values[mask_dictionary['safe_mask_'] == False] = -np.inf

				next_q_value = unsafe_next_q_values.max(dim=1, keepdim=True)[0]

			else:
				next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0]

			target = (reward + self.gamma * next_q_value * done_mask).to(self.device)

		# calculate element-wise dqn loss
		elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		return elementwise_loss

	def _target_hard_update(self):
		"""Hard update: target <- local."""
		print(f"Hard update performed at episode {self.episode}!")
		self.dqn_target.load_state_dict(self.dqn.state_dict())

	def _target_soft_update(self):
		"""Soft update: target_{t+1} <- local * tau + target_{t} * (1-tau)."""
		for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_target.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def log_data(self):

		if self.episodic_loss:
			self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

		self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
		self.writer.add_scalar('train/beta', self.beta, self.episode)

		self.writer.add_scalar('train/accumulated_reward', self.episodic_reward, self.episode)
		self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

		self.writer.flush()

	def load_model(self, path_to_file):

		self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth'):

		torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)
