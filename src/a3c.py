import os
import threading
import docker
import numpy as np
import tensorflow as tf
import scipy.signal

from time import sleep
from gym_torcs_docker import TorcsDockerEnv
from numpy.random import seed, randn


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


def obs_to_state(obs):
    return np.hstack(
        (obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY,
         obs.speedZ, obs.wheelSpinVel / 100.0, obs.rpm))


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constan(out)

    return _initializer


def addOUNoise(a, epsilon):

    def ou_func(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * randn(1)

    a_new = np.zeros(np.shape(a))
    noise = np.zeros(np.shape(a))

    noise[0][0] = (max(epsilon, 0) * ou_func(a[0][0], 0.0, 0.60, 0.30))
    noise[0][1] = (max(epsilon, 0) * ou_func(a[0][1], 0.5, 1.00, 0.10))
    noise[0][2] = (max(epsilon, 0) * ou_func(a[0][2], -0.1, 1.00, 0.05))

    a_new[0][0] = a[0][0] + noise[0][0]
    a_new[0][1] = a[0][1] + noise[0][1]
    a_new[0][2] = a[0][2] + noise[0][2]

    return a_new


class AC_Network(object):

    HIDDEN1_UNITS = 300
    HIDDEN2_UNITS = 600

    def __init__(self, s_size, a_size, scope, trainer):

        self.is_training = False

        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(
                shape=[None, s_size], dtype=tf.float32)

            s_layer1 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=self.inputs, activation=tf.nn.relu,
                    units=AC_Network.HIDDEN1_UNITS),
                training=self.is_training, name='s_layer_1')

            s_layer2 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer1, activation=tf.nn.relu,
                    units=AC_Network.HIDDEN2_UNITS),
                training=self.is_training, name='s_layer_2')

            # Output layers for policy and value estimations
            steering = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer2, units=1, activation=tf.nn.tanh),
                training=self.is_training, name='steering')

            acceleration = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer2, units=1, activation=tf.nn.sigmoid),
                training=self.is_training, name='acceleration')

            brake = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer2, units=1, activation=tf.nn.sigmoid),
                training=self.is_training, name='brake')

            self.policy = tf.concat(
                [steering, acceleration, brake], name='policy', axis=1)

            self.value = tf.layers.batch_normalization(
                tf.layers.dense(inputs=s_layer2, units=1),
                training=self.is_training, name='value')

            if scope != 'global':

                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(
                    shape=[None], dtype=tf.float32)
                log_prob = normal_dist.log_prob(self.a_his)
                exp_v = log_prob * td 
                entropy = normal_dist.entropy()
                                                            # encourage
                                                            # exploration
                                                                                self.exp_v
                                                                                =
                                                                                ENTROPY_BETA
                                                                                *
                                                                                entropy
                                                                                +
                                                                                exp_v
                                                                                self.a_loss
                                                                                =
                                                                                tf.reduce_mean(-self.exp_v)
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value, [-1])))

                self.policy_loss = 0.5 * tf.reduce_sum(
                    tf.square(
                        self.advantages - self.policy))

                self.loss = self.value_loss + self.policy_loss

                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)

                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)

                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 40.0)

                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker(object):

    def __init__(self, s_size, a_size, number, trainer, global_episodes,
                 docker_client, model_path):

        self.s_size = s_size
        self.a_size = a_size
        self.number = number
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.docker_client = docker_client
        self.model_path = model_path

        self.name = 'worker_{}'.format(self.number)
        self.docker_port = 3101 + self.number

        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            'train_{}'.format(self.number))

        self.env = TorcsDockerEnv(
            self.docker_client, self.name, self.docker_port)

        self.local_AC = AC_Network(
            self.s_size, self.a_size, self.name, self.trainer)
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        self.local_AC.is_training = True
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        rewards = rollout[:, 2]
        values = rollout[:, 5]
        self.rewards_plus = np.asarray(
            rewards.tolist() + [bootstrap_value])

        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = (
            rewards + gamma * self.value_plus[1:] - self.value_plus[:-1])
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.advantages: advantages}

        v_l, p_l, g_n, v_n, _ = sess.run(
            [self.local_AC.value_loss, self.local_AC.policy_loss,
             self.local_AC.grad_norms, self.local_AC.var_norms,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)

        return (v_l/len(rollout), p_l/len(rollout), g_n, v_n)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        self.local_AC.is_training = False

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting {}".format(self.name))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = []
                episode_step_count = 0

                obs = self.env.reset(relaunch=True)
                s = obs_to_state(obs)
                done = False

                epsilon = 1

                while not done:

                    a, v = sess.run(
                        [self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [s]})

                    epsilon -= 1.0 / max_episode_length
                    a = addOUNoise(a, epsilon)

                    obs, r, done, _ = self.env.step(a[0])

                    if not done:
                        s1 = obs_to_state(obs)
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r

                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if (len(episode_buffer) == 30 and not done
                            and episode_step_count != max_episode_length-1):

                        v1 = sess.run(
                            self.local_AC.value,
                            feed_dict={self.local_AC.inputs: [s]})[0, 0]

                        v_l, p_l, g_n, v_n = self.train(
                            episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                        if done:
                            break

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(
                        np.mean(episode_values))

                    if len(episode_buffer) != 0:
                        v_l, p_l, g_n, v_n = self.train(
                            episode_buffer, sess, gamma, 0.0)

                    if episode_count % 5 == 0 and episode_count != 0:
                        if (episode_count % 250 == 0
                                and self.name == 'worker_0'):
                            saver.save(
                                sess,
                                self.model_path+'/model-{:d}.cptk'.format(
                                    episode_count))
                            print("Saved Model")

                        mean_reward = np.mean(self.episode_rewards[-5:])
                        mean_length = np.mean(self.episode_lengths[-5:])
                        mean_value = np.mean(self.episode_mean_values[-5:])
                        summary = tf.Summary()
                        summary.value.add(
                            tag='Perf/Reward',
                            simple_value=float(mean_reward))
                        summary.value.add(
                            tag='Perf/Length',
                            simple_value=float(mean_length))
                        summary.value.add(
                            tag='Perf/Value',
                            simple_value=float(mean_value))
                        summary.value.add(
                            tag='Losses/Value Loss',
                            simple_value=float(v_l))
                        summary.value.add(
                            tag='Losses/Policy Loss',
                            simple_value=float(p_l))
                        summary.value.add(
                            tag='Losses/Grad Norm',
                            simple_value=float(g_n))
                        summary.value.add(
                            tag='Losses/Var Norm',
                            simple_value=float(v_n))

                        self.summary_writer.add_summary(
                            summary, episode_count)

                        self.summary_writer.flush()

                    if self.name == 'worker_0':
                        sess.run(self.increment)
                        episode_count += 1


def play_game(num_workers):

    seed(6486)

    max_episode_length = 300
    gamma = .99
    load_model = False
    model_path = '../model'
    a_size = 3
    s_size = 29
    docker_client = docker.from_env()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    if not os.path.exists(model_path):
            os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(
            0, dtype=tf.int32, name='global_episodes', trainable=False)

        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(s_size, a_size, 'global', None)

        workers = []
        for i in range(num_workers):
            workers.append(
                Worker(
                    s_size, a_size, i, trainer, global_episodes,
                    docker_client, model_path))

        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            t = threading.Thread(
                target=(
                    lambda: worker.work(
                        max_episode_length, gamma, sess, coord, saver)))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)


if __name__ == "__main__":
    play_game(1)
