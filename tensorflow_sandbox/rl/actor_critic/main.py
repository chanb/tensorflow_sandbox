from tensorflow_sandbox.rl.actor_critic.agent import Agent
from tensorflow_sandbox.rl.actor_critic.algos.reinforce import REINFORCE


def main():
    with tf.Session() as sess:
        agent = Agent(1, 1)
        algo = REINFORCE()
        tf.global_variables_initializer().run()


if __name__ == "__main__":
    main()
