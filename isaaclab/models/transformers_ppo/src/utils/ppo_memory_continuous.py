"""
PPO memory class that manages the memory and batch generation for the class
"""

import numpy as np


class PPOMemoryContinuous:
    def __init__(self, batch_size, N, observation_space, action_n):
        self.N = N
        self.observation_space = observation_space
        self.states = np.zeros((self.N, *self.observation_space))
        self.action_n = action_n
        self.probs = np.zeros(self.N)
        self.vals = np.zeros(self.N)
        self.actions = np.zeros((self.N, self.action_n))
        self.rewards = np.zeros(self.N)
        self.dones = np.zeros(self.N)
        self.batch_size = batch_size

        self.mem_counter = 0

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return (
            self.states,
            self.actions,
            self.probs,
            self.vals,
            self.rewards,
            self.dones,
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        idx = self.mem_counter % self.N
        self.states[idx] = state
        self.actions[idx] = action
        self.probs[idx] = probs
        self.vals[idx] = vals
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.mem_counter += 1

    def clear_memory(self):
        self.states = np.zeros((self.N, *self.observation_space))
        self.probs = np.zeros(self.N)
        self.actions = np.zeros((self.N, self.action_n))
        self.rewards = np.zeros(self.N)
        self.dones = np.zeros(self.N)
        self.vals = np.zeros(self.N)
