import gym
import numpy as np
import tensorflow as tf

from agent import Agent


def run():
    env = gym.make('MountainCar-v0')

    done = False
    with tf.Session() as sess:
        agent = Agent(action_dim=3, state_dim=2)

        state = env.reset()
        print(state.shape)
        while not done:
            action = agent.act(state.reshape(1, 2, 1))

            next_state, reward, done, info = env.step(action)
            env.render()

if __name__ == "__main__":
    run()
