import numpy as np
import tensorflow as tf

from tensorflow_sandbox.models.actor_critics import FullyConnectedActorCritic


class Agent():
    def __init__(self, action_dim, state_dim,
                 latent_size=128,
                 shared_hidden_sizes=[128],
                 actor_hidden_sizes=[128],
                 critic_hidden_sizes=[64],
                 discount=0.99, lr=3e-4,
                 train_writer=None):
        self._train_writer = train_writer
        self._discount = discount
        self._lr = lr

        self._latent_size = latent_size
        self._shared_hidden_sizes = shared_hidden_sizes
        self._actor_hidden_sizes = actor_hidden_sizes
        self._critic_hidden_sizes = critic_hidden_sizes

        self._action_dim = action_dim
        self._state_dim = state_dim
        self._input_dim = [None]
        self._input_dim.extend(state_dim)

        self._build_network()

    def act(self, state, stochastic=True):
        probs = self.prob_value_pair[0].eval({self.state_input: state})[0]
        value = self.prob_value_pair[1].eval({self.state_input: state})[0]
        if stochastic:
            action = np.random.choice(self._action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return action, probs, value

    def _build_network(self):
        # Placeholder for input
        self.state_input = tf.placeholder(
            tf.float32, self._input_dim, name='state_input')

        # Construct actor critic network
        with tf.variable_scope("actor_critic"):
            self._actor_critic = FullyConnectedActorCritic(
                input_size=np.product(self._state_dim),
                output_size=self._action_dim,
                latent_size=self._latent_size,
                shared_hidden_sizes=self._shared_hidden_sizes,
                actor_hidden_sizes=self._actor_hidden_sizes,
                critic_hidden_sizes=self._critic_hidden_sizes
            )

            self.logits_value_pair = self._actor_critic(self.state_input)
            self.prob_value_pair = (tf.nn.softmax(self.logits_value_pair[0]),
                                    self.logits_value_pair[1])


if __name__ == "__main__":
    import numpy as np

    with tf.Session() as sess:
        agent = Agent(2, [1])
        tf.global_variables_initializer().run()
        res = agent.act(np.array([[1.]]), False)
        print(res)
