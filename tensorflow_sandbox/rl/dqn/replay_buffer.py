from datetime import datetime

import _pickle as pickle
import numpy as np
import os


class ReplayBuffer():
    def __init__(self, cache_dir, max_transitions=10000, batch_size=128):
        start_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self._max_transitions = max_transitions
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._cache_file =
        os.path.join(cache_dir, "replay_buffer_{}.dat".format(start_time))
        self._metadata_file =
        os.path.join(cache_dir, "metadata_{}.pkl".format(start_time))

        # Shape: (Num_transitions, s, a, r, s', d)
        # s: 8 dim (2 np.float32)
        # a: 1 dim (1 np.uint8)
        # r: 1 dim (1 np.uint8)
        # s': 8 dim (2 np.float32)
        # d: 1 dim (1 np.bool)
        self._fp = np.memmap(self._cache_file, mode='w+', dtype=np.uint8,
                             shape=(max_transitions, 19))

        self.curr_s = np.ascontiguousarray(self._fp[:, :8]).view(np.float32)
        self.action = self._fp[:, 8]
        self.reward = self._fp[:, 9]
        self.next_s = np.ascontiguousarray(self._fp[:, 9:17]).view(np.float32)
        self.terminated = self._fp[:, -1]
        self._count = 0
        self._pointer = 0

    def add(self, curr_s, action, reward, next_s, terminated):
        self.curr_s[self._pointer] = curr_s
        self.action[self._pointer] = action
        self.reward[self._pointer] = reward
        self.next_s[self._pointer] = next_s
        self.terminated[self._pointer] = terminated

        if self._count < self._max_transitions:
            self._count += 1

        self._pointer = (self._pointer + 1) % self._max_transitions

    def get_state(self, index):
        assert self._count < index, "out of bound"
        return self.curr_s[index % self._max_transitions]

    def sample(self):
        idx_range = np.min(self._count, self._max_transitions)
        batch_size = np.min(self._count, self._batch_size)
        random_idx = np.random.choice(range(idx_range), size=batch_size,
                                      replace=False)
        return self._fp[random_idx]

    def save_metadata(self):
        with open(self._metadata_file, 'wb') as f:
            pickle.dump([self._max_transitions, self._fp.shape,
                         self._pointer, self._count], f)

if __name__ == "__main__":
    cache_dir = "/Users/bryan/Documents/sandbox/data/"
    replay_buffer = ReplayBuffer(cache_dir)
