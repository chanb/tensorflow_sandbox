import gym
import numpy as np
import tensorflow as tf

from agent import Agent


def run():
    with tf.Session() as sess:
        agent = Agent(2, 1)
        while True:
            action = agent.act([[[np.random.normal(0, 1)]]])
            print(action)

if __name__ == "__main__":
    run()
