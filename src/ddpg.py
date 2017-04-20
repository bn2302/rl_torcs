import docker

from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from gym_torcs_docker import TorcsDockerEnv
from numpy.random import seed, randn
import numpy as np
import tensorflow as tf


# Ornstein-Uhlenbeck Process
def ou_func(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * randn(1)


def play_game(train_indicator=1):

    checkpoint_file = '../weights/model.ckpt'
    logdir = '../logs/train'
    # 1 means Train, 0 means simply Run
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99
    tau = 0.001  # Target Network HyperParameters
    lra = 0.0001  # Learning rate for Actor
    lrc = 0.001  # Lerning rate for Critic

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    seed(1337)

    explore = 100000.
    episode_count = 2000
    max_steps = 10000
    done = False
    step = 0
    epsilon = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    actor = ActorNetwork(sess, state_dim, action_dim, tau, lra)
    critic = CriticNetwork(sess, state_dim, action_dim, tau, lrc)
    buff = ReplayBuffer(buffer_size)  # Create replay buffer
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    with tf.name_scope('summary'):
        loss_summary_op = tf.summary.scalar(
            'loss', critic.loss, collections=['loss'])

        reward_ph = tf.placeholder(
            shape=[None, ], name='reward', dtype=tf.float32)
        target_q_values_ph = tf.placeholder(
            shape=[None, 3], name='target_q_values', dtype=tf.float32)
        y_t_ph = tf.placeholder(
            shape=[None, 3], name='target_y_t', dtype=tf.float32)

        tf.summary.scalar(
            'reward', tf.reduce_mean(reward_ph), collections=['reward'])
        tf.summary.scalar(
            'target_q_values', tf.reduce_mean(target_q_values_ph),
            collections=['reward'])
        tf.summary.scalar(
            'y_t', tf.reduce_mean(y_t_ph), collections=['reward'])

        reward_summary_op = tf.summary.merge_all('reward')

    docker_client = docker.from_env()

    # Generate a Torcs environment
    env = TorcsDockerEnv(docker_client, "worker")

    # Now load the weight
    try:
        print("Now we load the weight")
        saver.restore(sess, checkpoint_file)
    except tf.errors.NotFoundError as e:
        print("{}: Weight not found".format(e))

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        recent_rewards = np.ones(50) * 1e9

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                         ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0,
                         ob.rpm))

        total_reward = 0.

        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / explore
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.predict(s_t.reshape(1, s_t.shape[0]))

            noise_t[0][0] = train_indicator * \
                max(epsilon, 0) * \
                ou_func(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * \
                max(epsilon, 0) * \
                ou_func(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * \
                max(epsilon, 0) * \
                ou_func(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, _ = env.step(a_t[0])

            recent_rewards[j % 50] = r_t

            if np.median(recent_rewards) < 5.0:
                break

            s_t1 = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,
                 ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(batch_size)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_predict(
                new_states, actor.target_predict(new_states))

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma * target_q_values[k]

            if (train_indicator):
                loss += critic.train(y_t, states, actions)
                a_for_grad = actor.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
                if step % 500:
                    loss_summary, reward_summary = sess.run(
                            [loss_summary_op,
                             reward_summary_op],
                            feed_dict={
                                critic.expected_critic: y_t,
                                critic.state: states,
                                critic.action: actions,
                                reward_ph: rewards,
                                target_q_values_ph: target_q_values,
                                y_t_ph: y_t})

                    train_writer.add_summary(loss_summary)
                    train_writer.add_summary(reward_summary)

            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action",
                  a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                saver.save(sess, checkpoint_file)

        print("TOTAL REWARD @ " + str(i) +
              "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    train_writer.close()
    print("Finish.")


if __name__ == "__main__":
    play_game()


