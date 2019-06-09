import numpy as np
import tensorflow as tf

from tensorflow_sandbox.models.q_functions import FullyConnectedQFunction


class Agent():
    def __init__(self, action_dim, state_dim, history_length=1,
                 lr=3e-4, eps=0.1):
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._history_length = history_length
        self._lr = lr
        self._build_dqn()
        self._eps = eps

    def act(self, state):
        if np.random.rand() < self._eps:
            action = np.random.choice(self._action_dim)
        else:
            action = self.best_action.eval({self.state_input: state})[0]

        return action

    def _build_dqn(self):
        # Placeholder for input
        self.state_input = tf.placeholder(
            tf.float32, [None, self._state_dim, self._history_length],
            name='state_input')

        # Construct train and target networks
        with tf.variable_scope("train"):
            self._train_network = FullyConnectedQFunction(
                input_size=self._state_dim,
                output_size=self._action_dim,
                name="training_network")

            self.q = self._train_network(self.state_input)
            self.best_action = tf.argmax(self.q, axis=1)

        with tf.variable_scope("target"):
            self._target_network = FullyConnectedQFunction(
                input_size=self._state_dim,
                output_size=self._action_dim,
                name="target_network")

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
            self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
            self.action = tf.placeholder(tf.int64, [None], name='action')

            action_one_hot = tf.one_hot(self.action, self._action_dim,
                                        name='action_one_hot')
            q_on_a = tf.reduce_sum(self.q * action_one_hot, name='q_on_a')

            self.delta = self.target_q - q_on_a
            self.loss = tf.reduce_mean(self.delta, name="loss")
            self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

        # Initialize variables
        tf.global_variables_initializer().run()

        self.update_target_network()

    def update_target_network(self):
        for param_name in self._target_parameters:
            self._assign_ops[param_name].eval(
                {self._target_parameter_inputs[param_name]:
                 self._train_network.parameters[param_name].eval()})

if __name__ == "__main__":
    with tf.Session() as sess:
        agent = Agent(1, 1)
