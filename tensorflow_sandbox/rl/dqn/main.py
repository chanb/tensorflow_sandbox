import gym
import numpy as np
import os
import tensorflow as tf

from datetime import datetime

from agent import Agent
from replay_buffer import ReplayBuffer


def train(cache_dir, model_path, max_timesteps=1000, lr=3e-4, discount=0.99,
          eps=0.1, update_timestep=400, batch_size=128, max_transitions=100000,
          history_length=1):
    start_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    env = gym.make('MountainCar-v0')
    curr_episode = 1
    curr_timesteps = 0
    num_updates = 1

    state_shape = [history_length]
    for dim in env.observation_space.shape:
        state_shape.append(dim)
    state = np.zeros(shape=state_shape, dtype=np.float32)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            logdir="./runs/{}_train".format(start_time))
        replay_buffer = ReplayBuffer(cache_dir,
                                     state_dim=state.shape,
                                     batch_size=batch_size,
                                     max_transitions=max_transitions,
                                     start_time=start_time)
        agent = Agent(sess=sess, action_dim=3, state_dim=state.shape,
                      replay_buffer=replay_buffer,
                      lr=lr, discount=discount, eps=eps,
                      train_writer=train_writer)

        saver = tf.train.Saver()
        obs = env.reset().reshape([1] + state_shape[1:]).astype(np.float32)
        state = np.vstack((state[1:], obs))

        for timestep in range(1, max_timesteps + 1):
            env.render()
            action = agent.act(np.expand_dims(state, axis=0))

            next_obs, reward, done, info = env.step(action)
            reward = np.array(reward, dtype=np.float32).reshape((1, 1))

            next_state = np.vstack((state[1:], next_obs.reshape(
                [1] + state_shape[1:]).astype(np.float32)))
            replay_buffer.add(state, action, reward, next_state, done)

            agent.update(timestep)

            state = next_state
            curr_timesteps += 1

            if done:
                print("Episode {} completed. Lasted {} steps."
                      .format(curr_episode, curr_timesteps))

                state = np.zeros(shape=state_shape, dtype=np.float32)
                obs = env.reset().reshape(
                    [1] + state_shape[1:]).astype(np.float32)
                state = np.vstack((state[1:], obs))
                curr_episode += 1
                curr_timesteps = 0

            if timestep % update_timestep == 0:
                print("Replace target network #{} at timesteps: {}"
                      .format(num_updates, timestep))
                agent.update_target_network()
                save_path = saver.save(sess, model_path)
                replay_buffer.save_metadata()
                num_updates += 1


def run():
    env = gym.make('MountainCar-v0')

    done = False
    with tf.Session() as sess:
        agent = Agent(action_dim=3, state_dim=2)

        state = env.reset()
        while not done:
            action = agent.act(state.reshape(1, 2, 1))

            next_state, reward, done, info = env.step(action)
            env.render()

if __name__ == "__main__":
    # run()
    cache_dir = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "saved_models/tmp")
    train(cache_dir,
          model_path,
          discount=1,
          max_timesteps=1000000,
          update_timestep=10000,
          batch_size=10000,
          max_transitions=100000,
          eps=0.9,
          history_length=4)
