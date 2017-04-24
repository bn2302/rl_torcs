import os
import threading
import numpy as np
import tensorflow as tf
import scipy.signal

from time import sleep
from gym_torcs_docker import TorcsDockerEnv, obs_to_state
from networks import A3CNetwork


class Worker(object):

    def __init__(self, s_size, action_size, trainer, number, global_episodes,
                 docker_client, docker_port, modeldir, logdir):

        self.s_size = s_size
        self.action_size = action_size
        self.number = number
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.docker_client = docker_client
        self.modeldir = modeldir
        self.logdir = logdir

        self.name = 'worker_{}'.format(self.number)
        self.docker_port = docker_port

        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, 'train_{}'.format(self.number)))

        self.local_AC = A3CNetwork(
            self.s_size, self.action_size, self.trainer, self.name)
        self.update_local_ops = A3CNetwork.update_target_graph(
            'global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        def discount(x, gamma):
            return scipy.signal.lfilter(
                [1], [1, -gamma], x[::-1], axis=0)[::-1]

        self.local_AC.is_training = True
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = np.stack(rollout[:, 1], 0)[0][0]
        rewards = rollout[:, 2]
        values = rollout[:, 5]
        self.rewards_plus = np.asarray(
            rewards.tolist() + [bootstrap_value])

        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = (
            rewards + gamma * self.value_plus[1:] - self.value_plus[:-1])
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.actions: actions,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.advantages: advantages}
        value_loss, policy_loss, gradient_norm, value_norm, _ = sess.run(
            [self.local_AC.value_loss, self.local_AC.policy_loss,
             self.local_AC.grad_norms, self.local_AC.var_norms,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)
        self.local_AC.is_training = False

        return (value_loss/len(rollout), policy_loss/len(rollout),
                gradient_norm, value_norm)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        self.local_AC.is_training = False
        env = TorcsDockerEnv(
            self.docker_client, self.name, self.docker_port, training=True)

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting {}".format(self.name))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0

                observation = env.reset(relaunch=True)
                state_t = obs_to_state(observation)
                done = False

                epsilon = 1

                while not done:

                    action_t, value_t = sess.run(
                        [self.local_AC.action, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [state_t]})

                    epsilon -= 1.0 / max_episode_length

                    observation, reward_t, done, _ = env.step(action_t[0][0])

                    if not done:
                        state_t1 = obs_to_state(observation)
                        episode_frames.append(state_t1)
                    else:
                        state_t1 = state_t

                    episode_buffer.append(
                        [state_t, action_t, reward_t, state_t1, done,
                         value_t[0, 0]])
                    episode_values.append(value_t[0, 0])

                    episode_reward += reward_t

                    state_t = state_t1
                    total_steps += 1
                    episode_step_count += 1

                    if (len(episode_buffer) == 30 and not done
                            and episode_step_count != max_episode_length-1):

                        value_t1 = sess.run(
                            self.local_AC.value,
                            feed_dict={self.local_AC.inputs: [state_t]})[0, 0]

                        (value_loss, policy_loss, gradient_norm,
                            variable_norm) = self.train(
                                episode_buffer, sess, gamma, value_t1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(
                    np.mean(episode_values))

                if len(episode_buffer) != 0:
                    (value_loss, policy_loss, gradient_norm,
                     variable_norm) = self.train(
                        episode_buffer, sess, gamma, 0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    if (episode_count % 250 == 0
                            and self.name == 'worker_0'):
                        saver.save(
                            sess,
                            os.path.join(self.model_path,
                                         'model-{:d}.cptk'.format(
                                             episode_count)))

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    print(
                        "Worker", self.name, "Episode", episode_count,
                        "Reward", mean_reward, "value_Loss", value_loss,
                        "policy_loss", policy_loss)

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
                        simple_value=float(value_loss))
                    summary.value.add(
                        tag='Losses/Policy Loss',
                        simple_value=float(policy_loss))
                    summary.value.add(
                        tag='Losses/Grad Norm',
                        simple_value=float(gradient_norm))
                    summary.value.add(
                        tag='Losses/Var Norm',
                        simple_value=float(variable_norm))

                    self.summary_writer.add_summary(
                        summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                    episode_count += 1
        env.end()


class A3C(object):

    def __init__(
            self, docker_client, docker_start_port=3101,
            modeldir='../models/a3c', logdir='../logs/a3c'):

        self.docker_client = docker_client

        self.docker_start_port = docker_start_port

        self.max_episode_length = 300
        self.gamma = .99
        self.logdir = logdir
        self.modeldir = modeldir
        self.state_size = 29
        self.action_size = 2

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        tf.reset_default_graph()

        self.global_episodes = tf.Variable(
                0, dtype=tf.int32, name='global_episodes', trainable=False)

        if not os.path.exists(self.modeldir):
                os.makedirs(self.modeldir)

    def train(self, num_workers, load_model=False):
        with tf.device("/cpu:0"):

            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = A3CNetwork(
                self.state_size, self.action_size, None, 'global')

            workers = []
            for i in range(num_workers):
                workers.append(
                    Worker(
                        self.state_size, self.action_size, trainer, i,
                        self.global_episodes, self.docker_client,
                        self.docker_start_port + i,
                        self.modeldir, self.logdir))

            saver = tf.train.Saver(max_to_keep=5)

        with tf.Session(config=self.config) as sess:

            coord = tf.train.Coordinator()

            if load_model:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                t = threading.Thread(
                    target=(
                        lambda: worker.work(
                            self.max_episode_length, self.gamma, sess, coord,
                            saver)))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)


if __name__ == "__main__":
    import docker

    docker_client = docker.from_env()

    a3c = A3C(docker_client)
    a3c.train(1)
