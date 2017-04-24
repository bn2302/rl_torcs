import os
import random
import numpy as np
import tensorflow as tf

from collections import deque
from networks import ActorNetwork, CriticNetwork
from gym_torcs_docker import TorcsDockerEnv, obs_to_state
from numpy.random import seed, randn


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class DDPG(object):

    def __init__(
            self, docker_client, name='worker', port=3101,
            model_path='../models/ddpg', log_path='../logs/ddpg'):

        self.state_size = 29
        self.action_size = 2

        self.docker_client = docker_client

        self.buffer_size = 100000
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.001  # Target Network HyperParameters
        self.lra = 0.0001  # Learning rate for Actor
        self.lrc = 0.001  # Lerning rate for Critic
        seed(6486)

        self.explore = 100000.
        self.episode_count = 2000
        self.max_steps = 10000
        self.epsilon = 1

        self.model_path = model_path
        self.port = port
        self.name = name

        if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        tf.reset_default_graph()

        self.summary_writer = tf.summary.FileWriter(log_path)

        self.actor = ActorNetwork(
            self.state_size, self.action_size,
            tf.train.AdamOptimizer(self.lra), self.tau)

        self.critic = CriticNetwork(
            self.state_size, self.action_size,
            tf.train.AdamOptimizer(self.lrc), self.tau)

        self.buff = ReplayBuffer(self.buffer_size)
        self.saver = tf.train.Saver()
        self._create_summary()

    def _create_summary(self):
        with tf.name_scope('summary'):
            self.loss_summary_op = tf.summary.scalar(
                'loss', self.critic.loss, collections=['loss'])

            self.reward_ph = tf.placeholder(
                shape=[None, ], name='reward', dtype=tf.float32)
            self.target_q_values_ph = tf.placeholder(
                shape=[None, self.action_size], name='target_q_values',
                dtype=tf.float32)
            self.y_t_ph = tf.placeholder(
                shape=[None, self.action_size], name='target_y_t',
                dtype=tf.float32)

            tf.summary.scalar(
                'reward', tf.reduce_mean(
                    self.reward_ph), collections=['reward'])
            tf.summary.scalar(
                'target_q_values', tf.reduce_mean(self.target_q_values_ph),
                collections=['reward'])
            tf.summary.scalar(
                'y_t', tf.reduce_mean(self.y_t_ph), collections=['reward'])

            self.reward_summary_op = tf.summary.merge_all('reward')

    @staticmethod
    def addOUNoise(a, epsilon):

        def ou_func(x, mu, theta, sigma):
            return theta * (mu - x) + sigma * randn(1)

        a_new = np.zeros(np.shape(a))
        noise = np.zeros(np.shape(a))

        noise[0] = (max(epsilon, 0) * ou_func(a[0], 0.0, 0.60, 0.30))
        noise[1] = (max(epsilon, 0) * ou_func(a[1], 0.2, 1.00, 0.10))

        a_new[0] = a[0] + noise[0]
        a_new[1] = a[1] + noise[1]

        return a_new

    def train(self, track_name='', check_stuck=True):

        all_steps = 0

        if track_name == '':
            env = TorcsDockerEnv(
                self.docker_client, self.name, self.port, training=True)
        else:
            env = TorcsDockerEnv(
                self.docker_client, self.name, self.port,
                track_name=track_name)

        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.episode_count):

                recent_rewards = np.ones(100) * 1e9
                print("Episode : " + str(i) + " Replay Buffer "
                      + str(self.buff.count()))

                if np.mod(i, 3) == 0:
                    observation = env.reset(relaunch=True)
                else:
                    observation = env.reset()

                state_t = obs_to_state(observation)
                total_reward = 0

                for j in range(self.max_steps):
                    loss = 0
                    self.epsilon -= 1.0 / self.explore

                    action_t = self.actor.predict(
                        sess, state_t.reshape(1, state_t.shape[0]))

                    observation, reward_t, done, _ = env.step(
                        DDPG.addOUNoise(action_t[0], self.epsilon))
                    state_t1 = obs_to_state(observation)

                    recent_rewards[j % 100] = reward_t

                    if check_stuck and np.median(recent_rewards) < 5.0:
                        break

                    self.buff.add(
                        state_t, action_t[0], reward_t, state_t1, done)
                    batch = self.buff.getBatch(self.batch_size)
                    states = np.asarray([e[0] for e in batch])
                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([e[2] for e in batch])
                    new_states = np.asarray([e[3] for e in batch])
                    dones = np.asarray([e[4] for e in batch])
                    y_t = np.asarray([e[1] for e in batch])

                    target_q_values = self.critic.target_predict(
                        sess, new_states,
                        self.actor.target_predict(sess, new_states))

                    for k in range(len(batch)):
                        if dones[k]:
                            y_t[k] = rewards[k]
                        else:
                            y_t[k] = (
                                rewards[k] + self.gamma * target_q_values[k])

                    loss += self.critic.train(sess, y_t, states, actions)
                    actions_for_grad = self.actor.predict(sess, states)
                    grads = self.critic.gradients(
                        sess, states, actions_for_grad)
                    self.actor.train(sess, states, grads)
                    self.actor.target_train(sess)
                    self.critic.target_train(sess)

                    all_steps += 1

                    if j % 50:

                        loss_summary, reward_summary = sess.run(
                            [self.loss_summary_op,
                             self.reward_summary_op],
                            feed_dict={
                                self.critic.expected_critic: y_t,
                                self.critic.state: states,
                                self.critic.action: actions,
                                self.reward_ph: rewards,
                                self.target_q_values_ph: target_q_values,
                                self.y_t_ph: y_t})

                        self.summary_writer.add_summary(
                            loss_summary, all_steps)
                        self.summary_writer.add_summary(
                            reward_summary, all_steps)
                        self.summary_writer.flush()

                    total_reward += reward_t
                    state_t = state_t1
                    print(
                        "Episode", i, "Step", all_steps, "Action",
                        action_t, "Reward", reward_t, "Loss", loss)
                    if done:
                        break

                print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " +
                      str(total_reward))
                print("Total Step: " + str(all_steps))
                print("")

                if np.mod(i, 50) == 0:
                    self.saver.save(
                        sess, self.model_path+'/model-{:d}.cptk'.format(i))
        env.end()


if __name__ == "__main__":
    import docker

    docker_client = docker.from_env()

    ddpg = DDPG(
        docker_client, 3101, '../models/ddpg_gtrack1', '../logs/ddpg_gtrack1')
    ddpg.train('g-track-1')

    ddpg = DDPG(
        docker_client, 3101, '../models/ddpg_traintracks',
        '../logs/ddpg_traintracks')
    ddpg.train()

    ddpg = DDPG(
        docker_client, 3101, '../models/ddpg_gtrack1_nostuck',
        '../logs/ddpg_gtrack1_nostuck')
    ddpg.train('g-track-1', False)

    ddpg.train()
