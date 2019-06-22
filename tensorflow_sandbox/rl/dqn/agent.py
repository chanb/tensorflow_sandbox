import numpy as np
import tensorflow as tf

from tensorflow_sandbox.models.q_functions import FullyConnectedQFunction


class Agent():
    def __init__(self, sess, action_dim, state_dim, replay_buffer,
                 discount=0.99, lr=3e-4, eps=0.1, hidden_sizes=[128],
                 eps_decay=0.9999, eps_decay_threshold=20000,
                 train_writer=None):
        self._sess = sess
        self._replay_buffer = replay_buffer
        self._train_writer = train_writer
        self._discount = discount
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._lr = lr
        self._eps = eps
        self._hidden_sizes = hidden_sizes
        self._input_dim = [None]
        self._eps_decay = eps_decay
        self._eps_decay_threshold = eps_decay_threshold
        for dim in state_dim:
            self._input_dim.append(dim)

        self._build_dqn()

    def act(self, state):
        if np.random.rand() < self._eps:
            action = np.random.choice(self._action_dim)
        else:
            action = self._best_action.eval({self._state_input: state})[0]
        return action

    def _build_dqn(self):
        # Placeholder for input
        self._state_input = tf.placeholder(
            tf.float32, self._input_dim, name='state_input')

        # Construct train and target networks
        with tf.variable_scope("train"):
            self._train_network = FullyConnectedQFunction(
                input_size=np.product(self._state_dim),
                output_size=self._action_dim,
                hidden_sizes=self._hidden_sizes,
                name="training_network")

            self._train_q = self._train_network(self._state_input)
            self._best_action = tf.argmax(self._train_q, axis=1)

        with tf.variable_scope("target"):
            self._target_network = FullyConnectedQFunction(
                input_size=np.product(self._state_dim),
                output_size=self._action_dim,
                hidden_sizes=self._hidden_sizes,
                name="target_network")
            self._target_q = self._target_network(self._state_input)

        # Construct assign operations for updating target network
        self._target_parameter_inputs = dict()
        self._assign_ops = dict()

        self._target_parameters = self._target_network.parameters
        with tf.variable_scope("assign_to_target"):
            for param_name in self._target_parameters:
                self._target_parameter_inputs[param_name] = tf.placeholder(
                    tf.float32, self._target_parameters[param_name].shape,
                    name=param_name)
                self._assign_ops[param_name] = tf.assign(
                    self._target_parameters[param_name],
                    self._target_parameter_inputs[param_name])

        # Construct loss and optimizer
        with tf.variable_scope("loss_opt"):
            self._target_q_in = tf.placeholder(tf.float32, [None],
                                               name='target_q')
            self._action = tf.placeholder(tf.int64, [None], name='action')

            action_one_hot = tf.one_hot(self._action, self._action_dim,
                                        name='action_one_hot')
            q_on_a = tf.reduce_sum(self._train_q * action_one_hot,
                                   name='q_on_a')

            self._delta = self._target_q_in - q_on_a
            self._loss = tf.reduce_mean(tf.square(self._delta), name="loss")
            self._opt = tf.train.AdamOptimizer(self._lr).minimize(self._loss)

        # Logging
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self._loss)
            self._merged = tf.summary.merge_all()

        # Initialize variables
        tf.global_variables_initializer().run()

        self.update_target_network()

    def update(self, update_num):
        s, a, r, s_next, d = self._replay_buffer.sample()
        # print(r)
        q_next = self._target_q.eval({self._state_input: s_next})
        target = np.squeeze(r + (1 - d) * self._discount *
                            np.max(q_next, axis=1, keepdims=True), axis=1)

        _, train_summary, loss, q_val = self._sess.run(
            [self._opt, self._merged, self._loss, self._train_q],
            feed_dict={self._state_input: s,
                       self._target_q_in: target,
                       self._action: a})

        if update_num >= self._eps_decay_threshold:
            self._eps = np.max((0.1, self._eps * self._eps_decay))

        if self._train_writer:
            self._train_writer.add_summary(train_summary, update_num)

    def update_target_network(self):
        for param_name in self._target_parameters:
            evaled = self._train_network.parameters[param_name].eval()
            result = self._assign_ops[param_name].eval(
                {self._target_parameter_inputs[param_name]:
                 evaled})

if __name__ == "__main__":
    with tf.Session() as sess:
        agent = Agent(1, 1)
