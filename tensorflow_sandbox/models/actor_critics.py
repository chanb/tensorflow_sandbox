import numpy as np
import tensorflow as tf

from tensorflow_sandbox.layers import Conv2D, Flatten, FullyConnectedLayer
from tensorflow_sandbox.models.classifiers import FullyConnectedClassifier


class FullyConnectedActorCritic():
    def __init__(self, input_size, output_size, latent_size,
                 shared_hidden_sizes=None,
                 actor_hidden_sizes=None,
                 critic_hidden_sizes=None,
                 activation=tf.nn.relu,
                 name="fc_actor_critic"):
        self._shared = FullyConnectedClassifier(
            input_size=input_size,
            output_size=latent_size,
            hidden_sizes=shared_hidden_sizes,
            activation=activation,
            name="fc_shared_base"
        )

        self._actor = FullyConnectedClassifier(
            input_size=latent_size,
            output_size=output_size,
            hidden_sizes=actor_hidden_sizes,
            activation=activation,
            name="fc_actor"
        )

        self._critic = FullyConnectedClassifier(
            input_size=latent_size,
            output_size=1,
            hidden_sizes=critic_hidden_sizes,
            activation=activation,
            name="fc_critic"
        )

    def __call__(self, x):
        x = self._shared(x)
        relu = tf.nn.relu(x)
        logits = self._actor(x)
        val = self._critic(x)

        return logits, val


if __name__ == "__main__":
    import numpy as np
    state_input = tf.placeholder(
        tf.float32, [1, 1], name='state_input')

    model = FullyConnectedActorCritic(1, 2, 2)
    forward_call = model(state_input)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        res = sess.run(
            [forward_call],
            feed_dict={state_input: np.array([[1.]])}
        )
        print(res)
