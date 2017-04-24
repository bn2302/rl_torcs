#!/usr/bin/python3
import docker
import tensorflow as tf

from gym_torcs_docker import TorcsDockerEnv, obs_to_state
from ddpg import DDPG


def testModelOnTrack(
        docker_client, sess, model, trackname, max_steps=1000,
        docker_port=3101):

    env = TorcsDockerEnv(
        docker_client, 'test_{}'.format(trackname), port=docker_port)
    observation = env.reset(relaunch=True)
    state_t = obs_to_state(observation)

    total_reward = 0

    for _ in range(max_steps):
        action_t = model.predict(sess, state_t.reshape(1, state_t.shape[0]))
        observation, reward_t, done, _ = env.step(action_t[0])
        state_t = obs_to_state(observation)
        total_reward += reward_t

        if done:
            break

    env.end()

    return total_reward


def main():
    docker_client = docker.from_env()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    test_tracks = ['g-track-3', 'e-track-6', 'alpine-2']
    tf.reset_default_graph()

    model = DDPG(docker_client)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state('../models/ddpg_alltracks')
        saver.restore(sess, ckpt.model_checkpoint_path)

        for track in test_tracks:
            total_reward = testModelOnTrack(
                docker_client, sess, model.actor, track, max_steps=1000,
                docker_port=3121)

            print('Track', track, 'Reward', total_reward)


if __name__ == "__main__":
    main()
