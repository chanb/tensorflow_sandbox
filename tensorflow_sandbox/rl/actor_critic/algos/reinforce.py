import numpy as np
import tensorflow as tf


class REINFORCE():
    def __init__(self, sess, agent, storage, action_dim,
                 discount=0.99, lr=3e-4,
                 actor_coef=1., critic_coef=0.5, ent_coef=0.):
        self._sess = sess
        self._agent = agent
        self._storage = storage

        self._action_dim = action_dim

        self._discount = discount
        self._lr = lr

        self._actor_coef = actor_coef
        self._critic_coef = critic_coef
        self._ent_coef = ent_coef

        self._build_algo()

    def _build_algo(self):
        # Construct loss and optimizer
        with tf.variable_scope("loss_opt"):
            self._returns = tf.placeholder(
                tf.float32,
                [None],
                name="returns"
            )

            self._actions_taken = tf.placeholder(
                tf.int32,
                [None],
                name="actions_taken"
            )

            # Forward call
            self._logits = self._agent.logits_value_pair[0]
            self._values = self._agent.logits_value_pair[1]

            # Get log prob and policy entropy
            self._log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._actions_taken)

            self._entropies = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self._logits, labels=self._logits)

            # Compute actor, critic, and entropy losses
            self._critic_loss = tf.reduce_mean(
                tf.square(self._returns - self._values))
            self._actor_loss = tf.reduce_mean(
                self._log_probs * (self._returns - self._values))
            self._entropy_loss = tf.reduce_mean(
                self._entropies
            )

            self._loss = self._actor_coef * self._actor_loss + \
                self._critic_coef * self._critic_loss + \
                self._ent_coef * self._entropy_loss

            self._opt = tf.train.AdamOptimizer(self._lr).minimize(self._loss)

    def update(self):
        states, actions, rewards, policy_probs, dones = self._storage.get_all()

        states = np.array(states[:-1])
        actions = np.array(actions, dtype=np.int32)
        returns = np.zeros(shape=(len(rewards)))
        for idx, reward in enumerate(rewards[::-1]):
            returns[:-idx] += reward
            returns[:-idx] *= self._discount

        _, loss, actor_loss, critic_loss, entropy_loss = sess.run(
            [self._opt, self._loss, self._actor_loss, self._critic_loss,
                self._entropy_loss],
            feed_dict={
                agent.state_input: states,
                self._actions_taken: actions,
                self._returns: returns
                }
        )

        print("Loss: {}, Actor Loss: {}, Critic Loss: {}, Entropy: {} Return: {}".format(
            loss, actor_loss, critic_loss, entropy_loss, sum(rewards)))


if __name__ == "__main__":
    import gym

    from tensorflow_sandbox.rl.actor_critic.agent import Agent
    from tensorflow_sandbox.rl.actor_critic.storage import Storage

    # env_name = "MountainCar-v0"
    # action_dim = 3

    env_name = "CartPole-v0"
    action_dim = 2
    num_eps = 100000
    env = gym.make(env_name)
    done = False

    with tf.Session() as sess:
        storage = Storage()
        agent = Agent(action_dim=action_dim,
                      state_dim=env.observation_space.shape)
        algo = REINFORCE(sess=sess,
                         agent=agent,
                         storage=storage,
                         action_dim=action_dim,
                         lr=0.01,
                         critic_coef=0.7,
                         ent_coef=0.02)

        tf.global_variables_initializer().run()
        for _ in range(num_eps):
            state = env.reset()
            storage.states.append(state)
            while not done:
                env.render()
                action, probs, value = agent.act(np.expand_dims(state, axis=0))
                state, reward, done, info = env.step(action)
                storage.add(state, action, reward, probs, value, done)

            algo.update()
            storage.reset()
            done = False
