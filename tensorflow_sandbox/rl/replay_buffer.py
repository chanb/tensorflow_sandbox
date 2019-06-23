from datetime import datetime

import _pickle as pickle
import numpy as np
import os


class ReplayBuffer():
    def __init__(self, cache_dir, state_dim, max_transitions=10000,
                 batch_size=128, start_time=None):
        if not start_time:
            start_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self._max_transitions = max_transitions
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._cache_file = os.path.join(
            cache_dir, "replay_buffer_{}.dat".format(start_time))
        self._metadata_file = os.path.join(
            cache_dir, "metadata_{}.pkl".format(start_time))
        self._state_dim = state_dim
        self._flatten_state_dim = np.product(state_dim)

        state_storage_dim = [max_transitions]
        for dim in state_dim:
            state_storage_dim.append(dim)

        # Shape: (Num_transitions, s, a, r, s', d)
        # s: 4 dim (2 np.float32) * state_shape
        # a: 1 dim (1 np.uint8)
        # r: 4 dim (1 np.float32)
        # s': 4 dim (2 np.float32) * state_shape
        # d: 1 dim (1 np.bool)
        self._state_dim_in_uint8 = 4 * np.product(state_dim)
        self._fp = np.memmap(self._cache_file, mode='w+', dtype=np.uint8,
                             shape=(max_transitions,
                                    10 + self._state_dim_in_uint8 * 2))

        self._count = 0
        self._pointer = 0

    def add(self, curr_s, action, reward, next_s, terminated):
        self._fp[self._pointer,
                 :self._state_dim_in_uint8] = curr_s.reshape(
                     self._flatten_state_dim).view(np.uint8)
        self._fp[self._pointer,
                 self._state_dim_in_uint8] = action
        self._fp[self._pointer,
                 self._state_dim_in_uint8 + 1:
                 self._state_dim_in_uint8 + 5] = reward.view(np.uint8)
        self._fp[self._pointer,
                 self._state_dim_in_uint8 + 5:
                 2 * self._state_dim_in_uint8 + 5] = next_s.reshape(
                     self._flatten_state_dim).view(np.uint8)
        self._fp[self._pointer, -1] = terminated

        if self._count < self._max_transitions:
            self._count += 1
        self._pointer = (self._pointer + 1) % self._max_transitions

    def get_state(self, index):
        assert self._count < index, "out of bound"
        return self.curr_s[index % self._max_transitions]

    def sample(self):
        idx_range = np.min((self._count, self._max_transitions))
        batch_size = np.min((self._count, self._batch_size))
        random_idx = np.random.choice(range(idx_range), size=batch_size,
                                      replace=False)
        reshape_state_dim = [batch_size] + list(self._state_dim)

        curr_s = np.ascontiguousarray(
            self._fp[random_idx, :self._state_dim_in_uint8]
            ).view(np.float32).reshape(reshape_state_dim)
        action = self._fp[random_idx, self._state_dim_in_uint8]
        reward = np.ascontiguousarray(
            self._fp[random_idx, self._state_dim_in_uint8 + 1:
                     self._state_dim_in_uint8 + 5]
            ).view(np.float32)
        next_s = np.ascontiguousarray(
            self._fp[random_idx, self._state_dim_in_uint8 + 5:
                     2 * self._state_dim_in_uint8 + 5]
            ).view(np.float32).reshape(reshape_state_dim)
        done = np.expand_dims(
            np.ascontiguousarray(self._fp[random_idx, -1])
            .view(np.bool), axis=1)

        return curr_s, action, reward, next_s, done

    def save_metadata(self):
        with open(self._metadata_file, 'wb') as f:
            pickle.dump([self._max_transitions, self._fp.shape,
                         self._pointer, self._count], f)

if __name__ == "__main__":
    cache_dir = "/Users/bryan/Documents/sandbox/data/"
    replay_buffer = ReplayBuffer(cache_dir, (1, 1))
